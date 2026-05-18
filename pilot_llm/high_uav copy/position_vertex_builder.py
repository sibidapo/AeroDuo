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
            Stacked SmolVLM2 hidden sequences for T timesteps.
            S = 191 (81 image + ~100 language + 2 state tokens + boundary tokens).

        Returns
        -------
        place_nodes : Tensor [T, D_g]
            One compressed place node per timestep.
        """
        T = position_vertices.shape[0]
        device = position_vertices.device

        # Broadcast single query across T timesteps for batched processing
        queries = self.output_query.expand(T, -1, -1)  # [T, 1, D_g]

        # PerceiverIO expects [B, N, dim] input and [B, M_out, queries_dim] queries
        out = self.perceiver(position_vertices, queries=queries)  # [T, 1, D_g]

        return out.squeeze(1)  # [T, D_g]
