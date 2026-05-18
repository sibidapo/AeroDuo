"""
graph_encoder.py — HGT-based heterogeneous graph encoder for AeroDuo.

Graph node types
----------------
  position    : T nodes, features [D_g]          — one per timestep in the sliding window
  observation : K_active nodes, features [D_g]   — unique detected categories (zero-padded
                                                    rows in the input tensor are skipped)

Graph edge types
----------------
  (position,    temporal,     position)    — bidirectional between consecutive timesteps
  (position,    observes,     observation) — position t → every category detected at t
  (observation, observed_by,  position)   — reverse: each obs node → positions that saw it

Both observation directions are included so that HGT message passing lets position nodes
absorb scene content from their observations AND obs nodes gain temporal context from
positions, all within the same multi-layer pass.

Output
------
z_graph : Tensor [T, D_g]
    Full position-node sequence after HGT enrichment.  NOT pooled to a single vector —
    the flow matching network cross-attends to the entire [T, D_g] sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv


_NODE_TYPES = ['position', 'observation']
_EDGE_TYPES = [
    ('position',    'temporal',     'position'),
    ('position',    'observes',     'observation'),
    ('observation', 'observed_by',  'position'),
]
_METADATA = (_NODE_TYPES, _EDGE_TYPES)


class GraphEncoder(nn.Module):
    """
    Three-layer HGT encoder over the AeroDuo heterogeneous observation graph.

    Per-layer architecture:
        HGTConv(heads=4) → residual add → LayerNorm
    Separate LayerNorm parameters for each node type.

    Parameters
    ----------
    D_g        : int — node feature dim, shared by both node types (default 256)
    num_layers : int — number of HGTConv layers (default 3)
    heads      : int — attention heads per layer (default 4); D_g must be divisible by heads
    """

    METADATA = _METADATA

    def __init__(self, D_g: int = 256, num_layers: int = 3, heads: int = 4) -> None:
        super().__init__()
        assert D_g % heads == 0, f"D_g ({D_g}) must be divisible by heads ({heads})"
        self.D_g = D_g

        self.convs = nn.ModuleList([
            HGTConv(
                in_channels=D_g,
                out_channels=D_g,
                metadata=_METADATA,
                heads=heads,
            )
            for _ in range(num_layers)
        ])

        # Separate norms per node type per layer
        self.norms_pos = nn.ModuleList([nn.LayerNorm(D_g) for _ in range(num_layers)])
        self.norms_obs = nn.ModuleList([nn.LayerNorm(D_g) for _ in range(num_layers)])

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(
        self,
        place_nodes: torch.Tensor,           # [T, D_g]
        observation_vertices: torch.Tensor,   # [T, K_max, D_g]
    ) -> HeteroData:
        T    = place_nodes.shape[0]
        K_max = observation_vertices.shape[1]
        device = place_nodes.device
        dtype  = place_nodes.dtype

        # ── Valid observation nodes ───────────────────────────────────────────
        # A row is active iff it has at least one non-zero element.
        obs_mask  = observation_vertices.abs().sum(dim=-1) > 0   # [T, K_max]
        obs_feats = observation_vertices[obs_mask]                # [K_active, D_g]
        K_active  = obs_feats.shape[0]

        # ── Temporal edges — bidirectional (t→t+1 and t+1→t) ─────────────────
        if T > 1:
            fwd = torch.arange(T - 1, device=device)
            bwd = torch.arange(1,   T, device=device)
            temporal_ei = torch.stack(
                [torch.cat([fwd, bwd]),
                 torch.cat([bwd, fwd])],
                dim=0,
            )  # [2, 2*(T-1)]
        else:
            temporal_ei = torch.zeros(2, 0, dtype=torch.long, device=device)

        # ── Observation edges ─────────────────────────────────────────────────
        # Iterate (t, k) pairs in the same order as obs_feats to assign global
        # obs node indices that match the K_active rows of obs_feats.
        pos_src, obs_dst = [], []
        obs_global = 0
        for t in range(T):
            for k in range(K_max):
                if obs_mask[t, k]:
                    pos_src.append(t)
                    obs_dst.append(obs_global)
                    obs_global += 1

        if pos_src:
            pos_t = torch.tensor(pos_src, dtype=torch.long, device=device)
            obs_t = torch.tensor(obs_dst, dtype=torch.long, device=device)
            observes_ei    = torch.stack([pos_t, obs_t], dim=0)  # [2, K_active]
            observed_by_ei = torch.stack([obs_t, pos_t], dim=0)  # [2, K_active]
        else:
            observes_ei    = torch.zeros(2, 0, dtype=torch.long, device=device)
            observed_by_ei = torch.zeros(2, 0, dtype=torch.long, device=device)

        # ── Assemble HeteroData ───────────────────────────────────────────────
        data = HeteroData()
        data['position'].x    = place_nodes
        # Always provide at least one obs node so HGTConv sees a consistent metadata
        # shape; the dummy node has no edges and does not affect position output.
        data['observation'].x = (
            obs_feats if K_active > 0
            else torch.zeros(1, self.D_g, device=device, dtype=dtype)
        )
        data['position',    'temporal',    'position'].edge_index    = temporal_ei
        data['position',    'observes',    'observation'].edge_index  = observes_ei
        data['observation', 'observed_by', 'position'].edge_index     = observed_by_ei

        return data

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        place_nodes: torch.Tensor,           # [T, D_g]
        observation_vertices: torch.Tensor,   # [T, K_max, D_g]
    ) -> torch.Tensor:                        # [T, D_g]
        """
        Run HGT message passing and return the enriched position-node sequence.

        Parameters
        ----------
        place_nodes          : Tensor [T, D_g]         — from PositionVertexBuilder
        observation_vertices : Tensor [T, K_max, D_g]  — from stack_episode_vertices;
                               zero-padded rows for timesteps with fewer detections

        Returns
        -------
        z_graph : Tensor [T, D_g]
        """
        data = self._build_graph(place_nodes, observation_vertices)

        x_dict = {
            'position':    data['position'].x,
            'observation': data['observation'].x,
        }
        edge_index_dict = {
            ('position',    'temporal',    'position'):   data['position', 'temporal', 'position'].edge_index,
            ('position',    'observes',    'observation'): data['position', 'observes', 'observation'].edge_index,
            ('observation', 'observed_by', 'position'):   data['observation', 'observed_by', 'position'].edge_index,
        }

        for i, conv in enumerate(self.convs):
            out = conv(x_dict, edge_index_dict)

            # Residual add + LayerNorm; fall back to input features if a node type
            # received no incoming edges and HGTConv omits it from `out`.
            x_dict = {
                'position': self.norms_pos[i](
                    out.get('position', x_dict['position']) + x_dict['position']
                ),
                'observation': self.norms_obs[i](
                    out.get('observation', x_dict['observation']) + x_dict['observation']
                ),
            }

        return x_dict['position']   # [T, D_g] — z_graph
