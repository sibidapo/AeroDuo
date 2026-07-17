from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch.distributions import Beta

try:
    from ..config.lowuavconfig import DITConfig, FeatureMergerConfig, LowUAVConfig
except ImportError:
    from config.lowuavconfig import DITConfig, FeatureMergerConfig, LowUAVConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding (B, T, embedding_dim) given timesteps (B, T)."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        timesteps = timesteps.float()
        B, T = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


def swish(x):
    return x * torch.sigmoid(x)


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype if compute_dtype is not None else torch.float32
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        timesteps_proj = self.time_proj(timesteps).to(self.compute_dtype)
        return self.timestep_embedder(timesteps_proj)


class AdaLayerNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        output_dim = embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        temb = self.linear(self.silu(temb))
        scale, shift = temb.chunk(2, dim=1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.norm_type = norm_type
        self.posiitonal_embeddings = positional_embeddings

        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positional_embeddings` must also be defined."
            )

        self.pos_embed = (
            SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
            if positional_embeddings == "sinusoidal"
            else None
        )

        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim)
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = FeedForward(
            dim=dim,
            dropout=dropout,
            activation_fn=activation_fn,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, temb)
        else:
            norm_hidden_states = self.norm1(hidden_states)

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )

        if self.final_dropout:
            attn_output = self.final_dropout(attn_output)

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)
        return hidden_states


class DiT(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 48,
        output_dim: int = 1024,
        num_layers: int = 16,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()

        self.inner_dim = attention_head_dim * num_attention_heads
        self.gradient_checkpointing = False

        _dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        raw = getattr(self.config, "compute_dtype", torch.float32)
        _compute_dtype = _dtype_map[raw] if isinstance(raw, str) else raw
        self.timestep_encoder = TimestepEncoder(embedding_dim=self.inner_dim, compute_dtype=_compute_dtype)

        all_blocks = []
        for idx in range(self.config.num_layers):
            use_self_attn = idx % 2 == 1 and interleave_self_attention
            curr_cross_attention_dim = cross_attention_dim if not use_self_attn else None

            all_blocks.append(
                BasicTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=self.config.max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=curr_cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(all_blocks)

        #Output blocks
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, self.config.output_dim)
        print(
            "Total number of DiT parameters: ",
            sum(p.numel() for p in self.parameters() if p.requires_grad),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor],
        return_all_hidden_states: bool = False,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ):
        temb = self.timestep_encoder(timestep).to(hidden_states.dtype)

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        all_hidden_states = [hidden_states]

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1 and self.config.interleave_self_attention:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    temb=temb,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    temb=temb,
                )
            all_hidden_states.append(hidden_states)

        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        if return_all_hidden_states:
            return self.proj_out_2(hidden_states), all_hidden_states
        return self.proj_out_2(hidden_states)


class FeatureMerger(nn.Module):
    def __init__(
        self,
        cfg: FeatureMergerConfig,
        d_vlm: int = 2048,
        d_graph: int = 1024,
    ):
        super().__init__()
        self.d_vlm = d_vlm
        self.d_hidden = cfg.attn_hidden_dim
        self.n_head = cfg.n_head
        self.head_dim = self.d_hidden // self.n_head
        assert self.d_hidden % self.n_head == 0, "d_hidden must be divisible by n_head"

        kv_dim = cfg.kv_dim
        dropout = cfg.dropout

        #Projections
        self.graph_projector = nn.Linear(d_graph, kv_dim)
        self.q_proj = nn.Linear(d_vlm, self.d_hidden)
        self.k_proj = nn.Linear(kv_dim, self.d_hidden)
        self.v_proj = nn.Linear(kv_dim, self.d_hidden)
        self.out_proj = nn.Linear(self.d_hidden, d_vlm)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_vlm)

    def forward(self, z_graph: torch.Tensor, vlm_feat: torch.Tensor) -> torch.Tensor:
        dtype = self.graph_projector.weight.dtype
        z_graph = z_graph.to(dtype=dtype)
        vlm_feat = vlm_feat.to(dtype=dtype)

        z_proj = self.graph_projector(z_graph)

        B, S, _ = vlm_feat.shape
        _, T, _ = z_proj.shape

        # Reshape for multi-head: (B N d_hidden) -> (B N n_head head_dim) -> (B n_head N head_dim)
        q = self.q_proj(vlm_feat).view(B, S, self.n_head, self.head_dim).transpose(1, 2) #(B, n_head, S, head_dim)
        k = self.k_proj(z_proj).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   #(B, n_head, T, head_dim)
        v = self.v_proj(z_proj).view(B, T, self.n_head, self.head_dim).transpose(1, 2)   #(B, n_head, T, head_dim)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1) * (self.head_dim ** -0.5)) #(B, n_head, S, T)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # Weighted sum over values
        attn_output = torch.matmul(attn_weights, v) #(B, n_head, S, head_dim)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous() # (B, S, n_head, head_dim)
        attn_output = attn_output.view(B, S, self.d_hidden)    # (B, S, d_hidden)

        output = self.out_proj(attn_output) # (B, S, d_vlm)
        output = self.dropout_out(output)

        # Residual Connection + LayerNorm
        return self.norm(vlm_feat + output)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions: shape (B, T, action_dim)
        timesteps: shape (B, ) -- a single scalar per batch item
        returns: shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,)")

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        x = swish(self.layer2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.layer3(x)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class LowUAVActionHead(nn.Module):
    def __init__(self, cfg: LowUAVConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_timestep_buckets = cfg.num_timestep_buckets
        self.noise_s = cfg.noise_s
        self.num_inference_timesteps = cfg.num_inference_timesteps
        self.action_horizon = cfg.action_horizon
        self.action_dim = cfg.action_dim

        self.model = DiT(
            output_dim=cfg.output_dim,
            compute_dtype=cfg.dtype,
            **dataclasses.asdict(cfg.dit_cfg),
        )
        self.use_zgraph = cfg.use_zgraph
        if cfg.use_zgraph:
            self.feat_merger = FeatureMerger(cfg.feat_merger_cfg, cfg.vlm_hidden_dim, cfg.D_g)
        else:
            # Standalone mode: no FeatureMerger. The truncated VLM emits un-normed
            # mid-layer states, and the merger's final LayerNorm is what the DiT
            # normally sees — so keep a LayerNorm in its place.
            self.vl_norm = nn.LayerNorm(cfg.vlm_hidden_dim)

        if cfg.add_pos_embed:
            self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.input_emb_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(cfg.noise_beta_alpha, cfg.noise_beta_beta)

        self.state_encoder = MLP(cfg.state_dim, cfg.hidden_dim, cfg.input_emb_dim)
        self.action_encoder = ActionEncoder(cfg.action_dim, cfg.input_emb_dim)
        self.action_decoder = MLP(cfg.output_dim, cfg.hidden_dim, cfg.action_dim)

    def _condition_feats(
        self, vl_embs: torch.Tensor, z_graph: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """DiT conditioning: vl_embs × z_graph via FeatureMerger, or vl_embs alone."""
        if self.use_zgraph:
            assert z_graph is not None, "use_zgraph=True but z_graph is None"
            return self.feat_merger(z_graph, vl_embs)  # [B, L, H]
        return self.vl_norm(vl_embs.to(dtype=self.vl_norm.weight.dtype))

    def _encode_heading(self, pose: torch.Tensor) -> torch.Tensor:
        xyz = pose[..., :3]
        sin_h = torch.sin(pose[..., 3]).unsqueeze(-1)
        cos_h = torch.cos(pose[..., 3]).unsqueeze(-1)
        return torch.cat([xyz, sin_h, cos_h], dim=-1)

    def sample_time(self, batch_size, device, dtype):
        return (1 - self.beta_dist.sample([batch_size]).to(device, dtype=dtype)) * self.noise_s

    def forward(
        self,
        vl_embs: torch.Tensor,               # [B, L, H]
        z_graph: Optional[torch.Tensor],     # [B, T_h, D_g], or None when use_zgraph=False
        low_state: torch.Tensor,             # [B, 4]
        low_actions: torch.Tensor,           # [B, T_l, 4]
    ) -> torch.Tensor:

        device = vl_embs.device
        dtype = next(self.parameters()).dtype

        low_state = low_state.to(device=device, dtype=dtype)
        if low_state.dim() == 2:
            low_state = low_state.view(low_state.shape[0], 1, -1)
        assert low_state.dim() == 3, f"Expected low_state (B, 1, D), got {tuple(low_state.shape)}"

        low_actions = low_actions.to(device=device, dtype=dtype)
        merged_feats = self._condition_feats(vl_embs, z_graph)  # (B, L, H)

        low_state   = self._encode_heading(low_state)    # [B, 1, 4] -> [B, 1, 5]
        low_actions = self._encode_heading(low_actions)  # [B, T, 4] -> [B, T, 5]

        # =======================Step 1: Sample noise and timestep===============================
        noise = torch.randn(low_actions.shape, device=low_actions.device, dtype=low_actions.dtype)
        # Sample timestep t ∈ [0, 1]
        t = self.sample_time(low_actions.shape[0], device=low_actions.device, dtype=low_actions.dtype)
        t = t[:, None, None] # shape (B, 1, 1) for broadcast

        # ======================Step 2: Create noisy trajectory==================================
        noisy_trajectory = t * low_actions + (1 - t) * noise

        # ======================Step 3: Compute ground truth velocity field
        velocity = low_actions - noise

        # ======================Step 4: Encode noisy trajectory==================================
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # ======================Step 5: Encode state=============================================
        state_features = self.state_encoder(low_state)

        # ======================Step 6: Add position embedding===================================
        if self.cfg.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # =====================Step 7: Concatenate state and action feats========================
        sa_embs = torch.cat((state_features, action_features), dim=1)

        # =====================Step 8: Flow matching forward pass================================
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=merged_feats,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )

        pred = self.action_decoder(model_output)

        # =====================Step 9: Extract predicted action samples==========================
        pred_actions = pred[:, -low_actions.shape[1]:]

        # =====================Step 10: Calculate loss -> velocity field prediction error========
        loss = F.mse_loss(pred_actions, velocity)

        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        z_graph: Optional[torch.Tensor],
        low_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate actions from random noise using the flow matchin diffusion process.

        Args:
            vlm_embs: egocentric vlm embeddings from low UAV [B, seq_len, vlm_emb_dim]
            z_graph: shared graph embeddings from high UAV [B, T_h, D_g], or None
                when use_zgraph=False (standalone low-UAV mode)
            low_state: low UAV state [B, 4] -> [x, y, z, heading]

        Returns:
            action_pred: [B, action_horizon, 5]

        """
        # =================Step 1: Initialize actions from pure noise==================================
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        dtype = vl_embs.dtype

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # =================Step 2: Prepare all input features ========================================
        merged_feats = self._condition_feats(vl_embs, z_graph)  # [B, L, H]

        low_state   = low_state.to(device=device, dtype=dtype)
        if low_state.dim() == 2:  # [B, 4] -> [B, 1, 4]
            low_state = low_state.view(low_state.shape[0], 1, -1)
        assert low_state.dim() == 3, \
            f"Expected low_state shape [B, 1, 4], got {tuple(low_state.shape)}"

        low_state = self._encode_heading(low_state) # [B, 1, 4] -> [B, 1, 5]
        state_features = self.state_encoder(low_state)

        # =================Step 3: ODE integration loop =============================================
        for t in range(num_steps):
            # =============Step 3a: Compute current continuous timestep ==============================
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # =============Step 3b: Encode current action sequence ==================================
            timestep_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timestep_tensor)

            # =============Step 3c: Add position embedding ==========================================
            if self.cfg.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # =============Step 3d: Concatenat state and action feats ===============================
            sa_embs = torch.cat((state_features, action_features), dim=1)

            # ============Step 3e: Model forward pass (predicts action samples) ======================
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=merged_feats,
                timestep=timestep_tensor,
            )
            pred = self.action_decoder(model_output)

            # ============Step 3f: Extract predicted action samples ================================
            pred_velocity = pred[:, -self.action_horizon:]

            # ============Step 3h: Euler integration ==============================================
            actions = actions + (dt * pred_velocity)

        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
