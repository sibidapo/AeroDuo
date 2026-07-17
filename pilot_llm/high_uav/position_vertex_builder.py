"""
position_vertex_builder.py — pose cross-attention fusion + PerceiverIO pooling.

Two stages per timestep:

1. PoseFeatureMerger — the SmolVLM2 hidden sequence [S, 2048] (image + text
   tokens only; S ≈ 189) cross-attends over a 2-token KV context built from
   the projected high and low UAV poses.  Q = VLM hidden states,
   K/V = [proj(high_pose), proj(low_pose)].  Residual + LayerNorm keeps the
   output at [S, 2048], so the sequence is "re-colored" with pose information
   without changing its length or width.

2. PerceiverIO — compresses the pose-fused sequence [S, 2048] into a single
   place node [D_g] via content-dependent cross-attention, producing [T, D_g].
   A single learned output query drives Perceiver IO's cross-attention output,
   so different parts of the sequence contribute proportionally to how
   relevant they are for the query.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from perceiver_pytorch import PerceiverIO

try:
    from .config import AeroduoConfig
except ImportError:
    from config import AeroduoConfig  # direct script execution


class PoseFeatureMerger(nn.Module):
    """
    Cross-attention fusion of VLM hidden states with high/low UAV poses,
    mirroring FeatureMerger in low_uav/model/lowuav_action_head.py.

    Q  : VLM hidden states [N, S, d_vlm]
    K/V: 2-token context [N, 2, kv_dim] — one token per UAV, each produced by
         its own projector: (x, y, z, heading) → (x, y, z, sin_h, cos_h) → Linear(5, kv_dim).
         Separate projectors let the model distinguish which token is which UAV.

    Output: LayerNorm(vlm_feat + attn_out) — same shape as the query [N, S, d_vlm].
    """

    def __init__(
        self,
        d_vlm: int = 2048,
        d_hidden: int = 2048,
        kv_dim: int = 2048,
        n_head: int = 8,
        dropout: float = 0.0,
        pose_dim: int = 5,
    ):
        super().__init__()
        self.d_vlm = d_vlm
        self.d_hidden = d_hidden
        self.n_head = n_head
        self.head_dim = d_hidden // n_head
        assert d_hidden % n_head == 0, "d_hidden must be divisible by n_head"

        # Projections
        self.high_pose_proj = nn.Linear(pose_dim, kv_dim)
        self.low_pose_proj = nn.Linear(pose_dim, kv_dim)
        self.q_proj = nn.Linear(d_vlm, d_hidden)
        self.k_proj = nn.Linear(kv_dim, d_hidden)
        self.v_proj = nn.Linear(kv_dim, d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_vlm)

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_out = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_vlm)

    def _encode_heading(self, pose: torch.Tensor) -> torch.Tensor:
        xyz = pose[..., :3]
        sin_h = torch.sin(pose[..., 3]).unsqueeze(-1)
        cos_h = torch.cos(pose[..., 3]).unsqueeze(-1)
        return torch.cat([xyz, sin_h, cos_h], dim=-1)  # [..., 5]

    def forward(
        self,
        vlm_feat: torch.Tensor,   # [N, S, d_vlm]
        high_pose: torch.Tensor,  # [N, 4] — (x, y, z, heading_rad)
        low_pose: torch.Tensor,   # [N, 4]
    ) -> torch.Tensor:
        dtype = self.q_proj.weight.dtype
        vlm_feat = vlm_feat.to(dtype=dtype)
        high_pose = high_pose.to(dtype=dtype)
        low_pose = low_pose.to(dtype=dtype)

        high_tok = self.high_pose_proj(self._encode_heading(high_pose)).unsqueeze(1)  # [N, 1, kv_dim]
        low_tok = self.low_pose_proj(self._encode_heading(low_pose)).unsqueeze(1)     # [N, 1, kv_dim]
        pose_ctx = torch.cat([high_tok, low_tok], dim=1)                               # [N, 2, kv_dim]

        B, S, _ = vlm_feat.shape
        _, T, _ = pose_ctx.shape

        # Reshape for multi-head: (B N d_hidden) -> (B N n_head head_dim) -> (B n_head N head_dim)
        q = self.q_proj(vlm_feat).view(B, S, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, S, head_dim)
        k = self.k_proj(pose_ctx).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = self.v_proj(pose_ctx).view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Scaled Dot-Product Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1) * (self.head_dim ** -0.5))  # (B, n_head, S, T)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_attn(attn_weights)

        # Weighted sum over values
        attn_output = torch.matmul(attn_weights, v)  # (B, n_head, S, head_dim)

        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, S, n_head, head_dim)
        attn_output = attn_output.view(B, S, self.d_hidden)     # (B, S, d_hidden)

        output = self.out_proj(attn_output)  # (B, S, d_vlm)
        output = self.dropout_out(output)

        # Residual Connection + LayerNorm
        return self.norm(vlm_feat + output)


class PositionVertexBuilder(nn.Module):
    """
    Fuse position vertices [T, S, smolvlm2_hidden_dim] with high/low UAV poses
    (PoseFeatureMerger), then pool → [T, D_g] via PerceiverIO.

    Each timestep is processed independently. The same merger, PerceiverIO
    weights and output_query are shared across all T timesteps (no temporal
    mixing here — that happens in the HGTConv graph encoder downstream).

    Parameters
    ----------
    cfg : AeroduoConfig
        Single source-of-truth config; the relevant fields are
        ``smolvlm2_hidden_dim``, ``D_g``, ``perceiver_*`` (pooling) and
        ``pose_merger_*`` (cross-attention fusion).
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        self.D_g = cfg.D_g

        self.pose_merger = PoseFeatureMerger(
            d_vlm=cfg.smolvlm2_hidden_dim,
            d_hidden=cfg.pose_merger_hidden_dim,
            kv_dim=cfg.pose_merger_kv_dim,
            n_head=cfg.pose_merger_n_head,
            dropout=cfg.pose_merger_dropout,
        )

        self.perceiver = PerceiverIO(
            dim=cfg.smolvlm2_hidden_dim,
            queries_dim=cfg.D_g,
            logits_dim=cfg.D_g,
            depth=cfg.perceiver_depth,
            num_latents=cfg.perceiver_M,
            latent_dim=cfg.perceiver_D_latent,
            cross_heads=1,
            latent_heads=cfg.perceiver_n_heads,
        )

        # Single learned output query → one [D_g] place node per timestep
        self.output_query = nn.Parameter(torch.randn(1, 1, cfg.D_g))

    def forward(
        self,
        position_vertices: torch.Tensor, ### vlm output feat
        high_poses: torch.Tensor,
        low_poses: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        position_vertices : Tensor [T, S, smolvlm2_hidden_dim]
                         or Tensor [B, T, S, smolvlm2_hidden_dim]
        high_poses : Tensor [T, 4] or [B, T, 4] — (x, y, z, heading_rad)
        low_poses  : Tensor [T, 4] or [B, T, 4]
        mask       : optional Tensor [T, S] or [B, T, S], True = valid token.
            Needed when frames are zero-padded to a common S (per-frame VLM
            sequence length varies with the tokenized goal-cue text), so
            PerceiverIO's pooling ignores the padding.

        Returns
        -------
        place_nodes : Tensor [T, D_g]  or  [B, T, D_g]
        """
        if position_vertices.dim() == 3:
            # Single-episode path: [T, S, dim]
            T = position_vertices.shape[0]
            fused = self.pose_merger(position_vertices, high_poses, low_poses)  # [T, S, dim]
            queries = self.output_query.expand(T, -1, -1)           # [T, 1, D_g]
            out = self.perceiver(fused, mask=mask, queries=queries)  # [T, 1, D_g]
            return out.squeeze(1)                                    # [T, D_g]

        # Batched path: [B, T, S, dim] → flatten B*T → merge → perceiver → restore
        B, T = position_vertices.shape[:2]
        flat = position_vertices.reshape(B * T, *position_vertices.shape[2:])  # [B*T, S, dim]
        fused = self.pose_merger(
            flat,
            high_poses.reshape(B * T, -1),
            low_poses.reshape(B * T, -1),
        )                                                            # [B*T, S, dim]
        flat_mask = mask.reshape(B * T, -1) if mask is not None else None    # [B*T, S]
        queries = self.output_query.expand(B * T, -1, -1)           # [B*T, 1, D_g]
        out = self.perceiver(fused, mask=flat_mask, queries=queries)         # [B*T, 1, D_g]
        return out.squeeze(1).reshape(B, T, self.D_g)               # [B, T, D_g]
