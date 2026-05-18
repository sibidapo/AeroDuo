"""
flow_matching.py — Flow-matching denoiser for Stage 1 AeroDuo training.

Architecture
------------
GR00T DiT-style interleaved bidirectional self-attention and cross-attention
stack with AdaLN τ-modulation at every self-attn and cross-attn norm.

τ convention: τ=0 → noise,  τ=1 → clean data
  x_τ = (1 − τ)·noise + τ·clean
  v_target = clean − noise

Self-attention sequence (GR00T-style joint state+action):
  state token   [1, D_g]   — low UAV pose projected + type tag (type 0)
  action tokens [H, D_g]   — τ-conditioned encoded x_τ + horizon pos + type tag (type 1)
  → concat      [1+H, D_g]

Cross-attention context (graph only):
  graph tokens  [T, D_g]   — z_graph from GraphEncoder + learned positional

Per layer:
  AdaLN → bidirectional self-attention over [state | actions] → residual
  AdaLN → cross-attention (K,V = graph tokens) → residual
  LayerNorm → FFN → residual

Readout: slice the action positions [1:] from the joint sequence output.

Output (DiT-style τ-modulated readout):
  v_θ [H, action_dim] — predicted flow-matching vector field

No batch dimension — trained one episode at a time.
Internally reshapes to [1, S, D] for diffusers Attention compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

try:
    from .config import AeroduoConfig
except ImportError:
    from config import AeroduoConfig


# ── Timestep encoder ──────────────────────────────────────────────────────────

class TimestepEncoder(nn.Module):
    """
    Encode a discrete τ bucket → dense embedding.

    Mirrors GR00T's TimestepEncoder: Timesteps (fixed sinusoidal) →
    TimestepEmbedding (learned MLP).

    timesteps : LongTensor [N]
    returns   : [N, embedding_dim]
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(timesteps_proj)   # [N, D]


# ── AdaLayerNorm (GR00T version) ──────────────────────────────────────────────

class AdaLayerNorm(nn.Module):
    """
    GR00T-style adaptive LayerNorm.  SiLU is applied to temb *before* the
    linear (unlike the previous AeroDuo version which had a bare linear).
    No zero-init on self.linear — stability comes from proj_out_2 zero-init.

    x    : [N, S, D]   (batch-first sequence)
    temb : [N, D]
    returns: [N, S, D]
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.silu   = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.norm   = nn.LayerNorm(embedding_dim, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # temb: [N, D] → [N, 2D]; scale/shift: [N, D] each
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=-1)
        # scale[:, None] broadcasts [N, D] → [N, 1, D] over seq dim S
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


# ── ActionEncoder ─────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """
    τ-conditioned action encoder.  Plain-embodiment analog of GR00T's
    MultiEmbodimentActionEncoder — the action tokens receive τ directly before
    entering the denoiser stack, so low-τ (noisy) and high-τ (clean) tokens
    are pre-conditioned.

    actions : [H, action_dim]
    temb    : [1, D]          (broadcast over H)
    returns : [H, D]
    """

    def __init__(self, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.W1 = nn.Linear(action_dim, hidden_size)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, actions: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        a_emb         = self.W1(actions)                         # [H, D]
        tau_broadcast = temb.expand(a_emb.shape[0], -1)          # [H, D]
        x = torch.cat([a_emb, tau_broadcast], dim=-1)            # [H, 2D]
        x = F.silu(self.W2(x))
        return self.W3(x)                                        # [H, D]


# ── FlowDenoiserBlock ─────────────────────────────────────────────────────────

class FlowDenoiserBlock(nn.Module):
    """
    Single DiT-style transformer block for the flow-matching denoiser:

        AdaLN → bidirectional self-attn      → residual
        AdaLN → cross-attn (K,V = context)  → residual
        LayerNorm → FFN                      → residual

    All tensors are batch-first [1, S, D]; the dummy batch dim is managed
    by FlowMatchingNetwork.forward.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = True,
        norm_eps: float = 1e-5,
        norm_elementwise_affine: bool = False,
    ) -> None:
        super().__init__()

        # 1. Self-attention (bidirectional)
        self.norm1 = AdaLayerNorm(dim, norm_elementwise_affine, norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
        )

        # 2. Cross-attention (Q = action tokens, K,V = z_graph + state)
        self.norm2 = AdaLayerNorm(dim, norm_elementwise_affine, norm_eps)
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            dropout=dropout,
            bias=attention_bias,
        )

        # 3. FFN — plain LayerNorm (GR00T does not use AdaLN on norm3)
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            inner_dim=ffn_mult * dim,
            bias=True,
        )

    def forward(
        self,
        x:       torch.Tensor,   # [1, 1+H, D] — [state | action] tokens
        context: torch.Tensor,   # [1, T, D]   — graph tokens (K, V only)
        temb:    torch.Tensor,   # [1, D]       — τ embedding
    ) -> torch.Tensor:           # [1, 1+H, D]

        h = self.norm1(x, temb)
        x = x + self.attn1(h)

        h = self.norm2(x, temb)
        x = x + self.attn2(h, encoder_hidden_states=context)

        h = self.norm3(x)
        x = x + self.ff(h)

        return x


# ── Pose projector ────────────────────────────────────────────────────────────

class _PoseProjector(nn.Module):
    """
    Project raw low UAV pose [4] = (x, y, z, heading_rad) to a single
    token [1, D] in the denoiser's embedding space.

    Heading is encoded as (sin, cos) → [5] before the linear layer.
    """

    def __init__(self, D: int) -> None:
        super().__init__()
        self.proj = nn.Linear(5, D)   # 5 = (x, y, z, sin_h, cos_h)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)              # [1, 4]
        xyz   = pose[:, :3]                       # [1, 3]
        sin_h = torch.sin(pose[:, 3:4])           # [1, 1]
        cos_h = torch.cos(pose[:, 3:4])           # [1, 1]
        enc   = torch.cat([xyz, sin_h, cos_h], dim=-1)   # [1, 5]
        return self.proj(enc)                     # [1, D]


# ── FlowMatchingNetwork ───────────────────────────────────────────────────────

class FlowMatchingNetwork(nn.Module):
    """
    Flow-matching denoiser for low UAV trajectory prediction (Stage 1).

    Predicts the vector field v_θ(x_τ, τ | z_graph, low_state).

    τ convention: τ=0 → noise, τ=1 → clean data
      x_τ = (1 − τ)·noise + τ·clean
      v_target = clean − noise

    No batch dimension — single episode at inference / training time.
    Internally reshapes to [1, S, D] for diffusers Attention compatibility;
    inputs and outputs remain 2D.

    Parameters read from cfg
    ------------------------
    D_g                         : model dimension (256)
    action_dim                  : trajectory state dim (4)
    action_horizon              : denoising horizon H (8)
    window_T                    : graph window T (5)
    flow_matching_layers        : denoiser depth L (4)
    flow_matching_heads         : attention heads (4)
    flow_matching_ffn_mult      : FFN inner-dim multiplier (4)
    num_timestep_buckets        : τ discretization (1000)
    flow_matching_attention_bias: Q/K/V projection bias (True)
    flow_matching_activation    : FFN activation (gelu-approximate)
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        D          = cfg.D_g
        H          = cfg.action_horizon
        T          = cfg.window_T
        L          = cfg.flow_matching_layers
        heads      = cfg.flow_matching_heads
        ffn_mult   = cfg.flow_matching_ffn_mult
        head_dim   = D // heads
        attn_bias  = getattr(cfg, "flow_matching_attention_bias", True)
        activation = getattr(cfg, "flow_matching_activation", "gelu-approximate")
        n_buckets  = getattr(cfg, "num_timestep_buckets", 1000)

        assert D % heads == 0, f"D_g ({D}) must be divisible by flow_matching_heads ({heads})"

        self._n_buckets = n_buckets

        # ── τ embedding ───────────────────────────────────────────────────────
        self.timestep_encoder = TimestepEncoder(embedding_dim=D)

        # ── Action sequence ───────────────────────────────────────────────────
        self.action_encoder = ActionEncoder(cfg.action_dim, D)

        # ── Positional embeddings (slice-safe: [:T] and [:H]) ─────────────────
        self.horizon_pos = nn.Parameter(torch.randn(H, D) * 0.02)   # [H, D]
        self.graph_pos   = nn.Parameter(torch.randn(T, D) * 0.02)   # [T, D]

        # ── Token-type embedding: 0 = state token, 1 = action token ────────
        self.type_embed = nn.Embedding(2, D)

        # ── State projector ───────────────────────────────────────────────────
        self.state_proj = _PoseProjector(D)

        # ── Denoiser blocks ───────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            FlowDenoiserBlock(
                dim=D,
                num_attention_heads=heads,
                attention_head_dim=head_dim,
                cross_attention_dim=D,
                ffn_mult=ffn_mult,
                activation_fn=activation,
                attention_bias=attn_bias,
            )
            for _ in range(L)
        ])

        # ── DiT-style output head ─────────────────────────────────────────────
        self.norm_out   = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(D, 2 * D)
        self.proj_out_2 = nn.Linear(D, cfg.action_dim)

        # Zero-init: v_pred ≈ 0 at init → L_flow ≈ E‖clean − noise‖² (finite)
        nn.init.zeros_(self.proj_out_2.weight)
        nn.init.zeros_(self.proj_out_2.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        z_graph:   torch.Tensor,   # [T, D_g]       — from GraphEncoder
        low_state: torch.Tensor,   # [4]             — (x, y, z, heading_rad)
        x_tau:     torch.Tensor,   # [H, action_dim] — noisy trajectory at time τ
        tau:       torch.Tensor,   # scalar ∈ [0, 1] — τ=0 noise, τ=1 clean
    ) -> torch.Tensor:             # [H, action_dim] — predicted vector field v_θ
        T      = z_graph.shape[0]
        H      = x_tau.shape[0]
        device = z_graph.device
        dtype  = next(self.parameters()).dtype

        z_graph = z_graph.to(device=device, dtype=dtype)
        low_state = low_state.to(device=device, dtype=dtype)
        x_tau = x_tau.to(device=device, dtype=dtype)
        tau = tau.to(device=device, dtype=dtype)

        # ── 1. τ embedding ────────────────────────────────────────────────────
        t_bucket = (tau * self._n_buckets).long().view(1)   # [1]
        temb     = self.timestep_encoder(t_bucket)          # [1, D]

        # ── 2. Cross-attention context: graph tokens only ─────────────────────
        context = (
            z_graph + self.graph_pos[:T]
        ).unsqueeze(0)   # [1, T, D]

        # ── 3. Self-attention sequence: [state token | action tokens] ─────────
        state_type  = torch.zeros(1, dtype=torch.long, device=device)   # type 0
        action_type = torch.ones(H,  dtype=torch.long, device=device)   # type 1

        state_tok = (
            self.state_proj(low_state)                  # [1, D]
            + self.type_embed(state_type)
        )
        action_toks = (
            self.action_encoder(x_tau, temb)            # [H, D]
            + self.horizon_pos[:H]
            + self.type_embed(action_type)
        )
        sa = torch.cat([state_tok, action_toks], dim=0).unsqueeze(0)   # [1, 1+H, D]

        # ── 4. Denoiser stack ─────────────────────────────────────────────────
        for block in self.blocks:
            sa = block(sa, context, temb)

        # ── 5. Slice action positions, then DiT-style readout ─────────────────
        a = sa[:, 1:, :]   # discard state slot → [1, H, D]
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)  # [1, D] each
        a = self.norm_out(a) * (1 + scale[:, None]) + shift[:, None]   # [1, H, D]
        v_pred = self.proj_out_2(a)                                     # [1, H, action_dim]
        return v_pred.squeeze(0)                                        # [H, action_dim]
