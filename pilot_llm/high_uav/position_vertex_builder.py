"""
position_vertex_builder.py — PerceiverIO attention pooling of position vertices.

Compresses each timestep's SmolVLM2 hidden sequence [191, 2048] into a single
place node [D_g] via content-dependent cross-attention, producing [T, D_g].

A single learned output query drives Perceiver IO's cross-attention output,
so different parts of the 191-token sequence (image patches, language tokens,
state tokens) contribute proportionally to how relevant they are for the query.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from perceiver_pytorch import PerceiverIO


class PositionVertexBuilder(nn.Module):
    """
    Pool position vertices [T, S, smolvlm2_hidden_dim] → [T, D_g] via PerceiverIO.

    Each timestep is processed independently. The same PerceiverIO weights and
    output_query are shared across all T timesteps (no temporal mixing here —
    that happens in the HGTConv graph encoder downstream).

    Parameters
    ----------
    smolvlm2_hidden_dim : int
        Input token dimension (2048 for SmolVLM2-2.2B).
    D_g : int
        Output graph node dimension (256).
    perceiver_M : int
        Number of Perceiver IO latent vectors. Controls capacity of the
        intermediate bottleneck.
    perceiver_D_latent : int
        Latent dimension inside the Perceiver.
    perceiver_depth : int
        Number of (cross-attention → latent self-attention) blocks.
    perceiver_n_heads : int
        Number of heads in latent self-attention.
    """

    def __init__(
        self,
        smolvlm2_hidden_dim: int = 2048,
        D_g: int = 256,
        perceiver_M: int = 64,
        perceiver_D_latent: int = 256,
        perceiver_depth: int = 2,
        perceiver_n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.D_g = D_g

        self.perceiver = PerceiverIO(
            dim=smolvlm2_hidden_dim,
            queries_dim=D_g,
            logits_dim=D_g,
            depth=perceiver_depth,
            num_latents=perceiver_M,
            latent_dim=perceiver_D_latent,
            cross_heads=1,
            latent_heads=perceiver_n_heads,
        )

        # Single learned output query → one [D_g] place node per timestep
        self.output_query = nn.Parameter(torch.randn(1, 1, D_g))

    def forward(self, position_vertices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        position_vertices : Tensor [T, S, smolvlm2_hidden_dim]
                         or Tensor [B, T, S, smolvlm2_hidden_dim]

        Returns
        -------
        place_nodes : Tensor [T, D_g]  or  [B, T, D_g]
        """
        if position_vertices.dim() == 3:
            # Single-episode path: [T, S, dim]
            T = position_vertices.shape[0]
            queries = self.output_query.expand(T, -1, -1)          # [T, 1, D_g]
            out = self.perceiver(position_vertices, queries=queries) # [T, 1, D_g]
            return out.squeeze(1)                                    # [T, D_g]

        # Batched path: [B, T, S, dim] → flatten B*T → perceiver → restore
        B, T = position_vertices.shape[:2]
        flat = position_vertices.reshape(B * T, *position_vertices.shape[2:])  # [B*T, S, dim]
        queries = self.output_query.expand(B * T, -1, -1)           # [B*T, 1, D_g]
        out = self.perceiver(flat, queries=queries)                  # [B*T, 1, D_g]
        return out.squeeze(1).reshape(B, T, self.D_g)               # [B, T, D_g]
