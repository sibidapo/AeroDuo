"""
state_projector.py — Learned linear projectors for UAV pose and state tokens.

Both projectors map a raw numeric vector into SmolVLM2's hidden dimension so
that it can be appended as a single extra token to the VLM input sequence.

Modules
-------
HighUAVPoseProjector
    Maps (x, y, z, heading) → 1-token embedding of size smolvlm2_hidden_dim.
    Heading is encoded as (sin, cos) to preserve its circular nature before
    concatenation, giving a 5-dim input (= config.high_uav_pose_dim).

LowUAVStateProjector
    Maps (x, y, z, heading, ...) → 1-token embedding.
    Also holds a ``null_token`` — a registered nn.Parameter initialised to
    zeros — which is substituted when the low UAV state is unavailable at
    inference time.  The null token is already in the VLM hidden space (dim
    smolvlm2_hidden_dim); no projection is needed for the null path.

Both modules can be used standalone or composed inside PositionVertexBuilder.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from .config import AeroduoConfig
except ImportError:
    from config import AeroduoConfig  # direct script execution


# ─── Heading encoder helper ───────────────────────────────────────────────────

def encode_heading(heading_rad: torch.Tensor) -> torch.Tensor:
    """
    Convert a heading angle in radians to a 2-dim (sin, cos) representation.

    Parameters
    ----------
    heading_rad : Tensor of shape [...] (scalar or batched)

    Returns
    -------
    Tensor of shape [..., 2]
    """
    return torch.stack([torch.sin(heading_rad), torch.cos(heading_rad)], dim=-1)


# ─── High UAV pose projector ──────────────────────────────────────────────────

class HighUAVPoseProjector(nn.Module):
    """
    Project the high UAV's (x, y, z, heading) into SmolVLM2 token space.

    Input  : Tensor of shape [B, 4] — (x, y, z, heading_rad)
    Output : Tensor of shape [B, 1, smolvlm2_hidden_dim]
             (the "1" makes it trivial to cat with other token sequences)
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # high_uav_pose_dim = 5: (x, y, z, sin_h, cos_h)
        self.proj = nn.Linear(cfg.high_uav_pose_dim, cfg.smolvlm2_hidden_dim)

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pose : Tensor [B, 4] — (x, y, z, heading_rad)

        Returns
        -------
        Tensor [B, 1, smolvlm2_hidden_dim]
        """
        # Separate spatial coords and heading
        xyz     = pose[..., :3]                      # [B, 3]
        heading = pose[..., 3]                       # [B]
        heading_enc = encode_heading(heading)        # [B, 2]

        encoded = torch.cat([xyz, heading_enc], dim=-1)  # [B, 5]
        token   = self.proj(encoded)                     # [B, smolvlm2_hidden_dim]
        return token.unsqueeze(1)                        # [B, 1, smolvlm2_hidden_dim]


# ─── Low UAV state projector ──────────────────────────────────────────────────

class LowUAVStateProjector(nn.Module):
    """
    Project the low UAV's state vector into SmolVLM2 token space.

    State vector layout expected at ``config.low_uav_state_dim``:
        (x, y, z, heading_rad, extra_0, extra_1, ...)

    The heading element is always at index 3 and is replaced by (sin, cos)
    before projection, so the projected dimension is independent of layout.

    When the low UAV state is unavailable (e.g. early inference), pass
    ``state=None`` to receive the learned ``null_token`` instead.

    Parameters
    ----------
    null_token : nn.Parameter of shape [1, 1, smolvlm2_hidden_dim]
        Initialised to zeros; registered so it is included in state_dict and
        updated by the optimiser during Stage 2 joint training.

    Input  : Tensor [B, low_uav_state_dim] or None
    Output : Tensor [B, 1, smolvlm2_hidden_dim]
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # The raw state has `low_uav_state_dim` elements, but heading (index 3)
        # is replaced by (sin, cos) → net projected dim = low_uav_state_dim + 1.
        encoded_dim = cfg.low_uav_state_dim + 1   # +1 for the extra cos term
        self.proj = nn.Linear(encoded_dim, cfg.smolvlm2_hidden_dim)

        # Learned null token — used when state is unavailable at inference.
        # Shape [1, 1, D] so it broadcasts over batch dimension.
        self.null_token = nn.Parameter(
            torch.zeros(1, 1, cfg.smolvlm2_hidden_dim)
        )

    def forward(self, state: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        state : Tensor [B, low_uav_state_dim] or None

        Returns
        -------
        Tensor [B, 1, smolvlm2_hidden_dim]
        """
        if state is None:
            # null_token is [1,1,D]; caller decides batch handling
            return self.null_token

        # Replace heading (index 3) with (sin, cos)
        pre_heading  = state[..., :3]                  # [B, 3]
        heading      = state[..., 3]                   # [B]
        post_heading = state[..., 4:]                  # [B, low_uav_state_dim - 4]

        heading_enc  = encode_heading(heading)         # [B, 2]
        encoded = torch.cat(
            [pre_heading, heading_enc, post_heading], dim=-1
        )                                              # [B, low_uav_state_dim + 1]

        token = self.proj(encoded)                     # [B, smolvlm2_hidden_dim]
        return token.unsqueeze(1)                      # [B, 1, smolvlm2_hidden_dim]
