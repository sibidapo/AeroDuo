"""
position_vertex.py — Build the position vertex V_t for the spatiotemporal graph.

V_t encodes joint scene understanding at a single timestep by fusing:
  • BEV image (processed by SmolVLM2's vision encoder + perceiver resampler)
  • Language instruction (tokenised by SmolVLM2's text tokeniser)
  • High UAV state token  (1 extra token — projected inside SmolVLM2Encoder)
  • Low UAV state token   (1 extra token — projected inside SmolVLM2Encoder)

Token layout fed to the decoder
---------------------------------
[BEV image tokens (variable)]  [language tokens (variable)]
    [high_state_token (1)]  [low_state_token (1)]
^-- from processor                ↑ appended in SmolVLM2Encoder.forward()

After SmolVLM2's truncated decoder (vlm_layer_cutoff = 12 / 24 layers):
  last_hidden_state shape: [1, T+2, hidden_size]

Readout:
  mean-pool across all T+2 token positions → [hidden_size]
  → vt_proj (Linear + LayerNorm) → V_t of shape [D_g]

The V_t readout projection (vt_proj) is trainable in Stage 2 joint training.
SmolVLM2 weights and pose_token_proj are frozen; vt_proj receives grads.
LoRA adapters are attached to SmolVLM2 in Stage 4 without refactoring.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from .config import AeroduoConfig
    from .smolvlm2_encoder import SmolVLM2Encoder
except ImportError:
    from config import AeroduoConfig  # direct script execution
    from smolvlm2_encoder import SmolVLM2Encoder


class PositionVertexBuilder(nn.Module):
    """
    Assemble the position vertex V_t for one timestep.

    Trainable sub-modules (Stage 2 joint training):
      • vt_proj — Linear + LayerNorm readout projection

    Frozen:
      • encoder (SmolVLM2Encoder) — VLM weights and pose_token_proj

    Usage
    -----
    builder = PositionVertexBuilder(cfg)
    vt = builder(
        bev_image=pil_image,           # PIL.Image, RGB
        language_text=instruction_str,
        high_uav_pose=pose_tensor,     # [4] or [1,4]: (x,y,z,heading_rad)
        low_uav_state=state_tensor,    # [>=4] or None
        device=torch.device("cuda"),
    )
    # vt: Tensor of shape [D_g]
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = SmolVLM2Encoder(cfg)

        self.vt_proj = nn.Sequential(
            nn.Linear(cfg.smolvlm2_hidden_dim, cfg.D_g),
            nn.LayerNorm(cfg.D_g),
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        bev_image,
        language_text: str,
        high_uav_pose: torch.Tensor,            # [4] or [1, 4]
        low_uav_state: Optional[torch.Tensor],  # [>=4] or [1, >=4] or None
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build V_t for one BEV frame.

        Parameters
        ----------
        bev_image     : PIL.Image or np.ndarray — the BEV frame
        language_text : str — the navigation instruction
        high_uav_pose : Tensor [4] or [1,4] — (x, y, z, heading_rad)
        low_uav_state : Tensor [>=4] or None
        device        : torch.device

        Returns
        -------
        V_t : Tensor [D_g]
        """
        if high_uav_pose.dim() == 1:
            high_uav_pose = high_uav_pose.unsqueeze(0)   # [1, 4]
        high_uav_pose = high_uav_pose.to(device)

        if low_uav_state is None:
            low_uav_state = torch.zeros(1, 4, device=device)
        else:
            if low_uav_state.dim() == 1:
                low_uav_state = low_uav_state.unsqueeze(0)
            low_uav_state = low_uav_state.to(device)

        # Shape: [1, T+2, hidden_size]
        hidden = self.encoder(
            bev_image=bev_image,
            lang_description=language_text,
            low_state=low_uav_state,
            high_state=high_uav_pose,
            device=device,
        )

        pooled = hidden.mean(dim=1)    # [1, hidden_size]
        vt = self.vt_proj(pooled)      # [1, D_g]
        return vt.squeeze(0)           # [D_g]
