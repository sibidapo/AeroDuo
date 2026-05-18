"""
smolvlm2_encoder.py — SmolVLM2 loading and mid-layer hidden-state extraction.

Key design goals
----------------
1. Load SmolVLM2 completely frozen (no grad on any VLM parameter).
2. Physically truncate the decoder to layer ``vlm_layer_cutoff`` (N//2 = 12)
   so the forward pass exits early and never wastes compute on upper layers.
3. Expose decoder layers directly for Stage 4 LoRA attachment.
4. No disk I/O beyond the initial model load; all operations are in-memory.

Token layout (documented here and as comments in forward())
-----------------------------------------------------------
After processor encoding, the input sequence contains:

    [BEV image tokens (variable length)]  [language tokens (variable length)]
    ^-- injected by SmolVLM2 processor as image patch tokens

Two extra tokens are appended before the decoder forward:

    [...image tokens...] [...language tokens...] [high_state_token] [low_state_token]
                                                  ^[1]               ^[1]

Causal attention means high_state_token attends to all image+language context;
low_state_token attends to everything including high_state_token.

Notes on model internals
------------------------
* SmolVLM2ForConditionalGeneration wraps a vision encoder (perceiver resampler)
  and a language model (LLaMA-3 style, 24 decoder layers).
* ``model.model.text_model.layers`` is the list of decoder layers.
* Attention projections live at:
      model.model.text_model.layers[i].self_attn.{q,k,v,o}_proj
  These are standard nn.Linear modules, directly LoRA-attachable.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from .config import AeroduoConfig
except ImportError:
    from config import AeroduoConfig  # direct script execution

logger = logging.getLogger(__name__)


class UAVPoseProjector(nn.Module):
    """
    Project a UAV state into SmolVLM2 token space.

    Uses the first four elements (x, y, z, heading_rad); heading is encoded
    as (sin, cos), giving a fixed 5-dim input to a linear layer regardless of
    the full state vector length.

    Input  : Tensor [B, >=4]
    Output : Tensor [B, 1, vlm_hidden_dim]
    """

    def __init__(self, uav_pose_dim: int = 5, vlm_hidden_dim: int = 2048):
        super().__init__()
        self.proj = nn.Linear(uav_pose_dim, vlm_hidden_dim)

    def _encode_heading(self, pose: torch.Tensor) -> torch.Tensor:
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
        xyz   = pose[:, :3]
        sin_h = torch.sin(pose[:, 3]).unsqueeze(-1)
        cos_h = torch.cos(pose[:, 3]).unsqueeze(-1)
        return torch.cat([xyz, sin_h, cos_h], dim=-1)  # [B, 5]

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        encoded = self._encode_heading(pose)       # [B, 5]
        token   = self.proj(encoded)               # [B, vlm_hidden_dim]
        return token.unsqueeze(1)                  # [B, 1, vlm_hidden_dim]


class SmolVLM2Encoder(nn.Module):
    """
    Thin wrapper around SmolVLM2 that:
      - loads the model and processor once, frozen
      - physically truncates the decoder to vlm_layer_cutoff layers
      - provides ``forward`` which fuses BEV image, language, and two UAV state
        tokens, then returns the decoder's last_hidden_state
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        self.cfg = cfg

        logger.info("Loading SmolVLM2 processor …")
        self.processor = AutoProcessor.from_pretrained(cfg.smolvlm2_model_name)
        self.processor.image_processor.do_image_splitting = False

        logger.info("Loading SmolVLM2 model (frozen) …")
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            cfg.smolvlm2_model_name,
            torch_dtype=torch.float32,
        ).to("cuda")
        self.vlm.eval()

        for param in self.vlm.parameters():
            param.requires_grad_(False)

        self._lm_layers = self.vlm.model.text_model.layers
        n = len(self._lm_layers)
        logger.info(f"SmolVLM2 has {n} decoder layers.")

        cutoff = cfg.vlm_layer_cutoff if cfg.vlm_layer_cutoff is not None else n // 2
        self.vlm_layer_cutoff = cutoff
        self.vlm.model.text_model.layers = self._lm_layers[:cutoff]
        logger.info(f"Truncated decoder to {cutoff} layers (N//2 = {n // 2}).")

        self.hidden_size = self.vlm.config.text_config.hidden_size
        self.pose_token_proj = UAVPoseProjector(uav_pose_dim=5, vlm_hidden_dim=self.hidden_size)

    # ── Property: decoder layers (for LoRA attachment in Stage 4) ─────────────
    @property
    def decoder_layers(self) -> nn.ModuleList:
        """Truncated decoder layers — the only layers that run during forward."""
        return self.vlm.model.text_model.layers

    # ── Embedding helpers ──────────────────────────────────────────────────────

    def embed_image(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(dtype=self.vlm.model.vision_model.dtype)
        return self.vlm.get_image_features(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.vlm.model.text_model.get_input_embeddings()(token_ids)

    # ── Processor helper ───────────────────────────────────────────────────────

    def build_processor_inputs(
        self,
        bev_image,
        language_text: str,
        device: torch.device = torch.device("cuda"),
    ) -> dict:
        text_parts = language_text.split(
            "The description of the target and its surrounding is shown below."
        )
        direction_text = text_parts[0].strip().split(
            "Compass north corresponds to the top of the bird's-eye-view image."
        )[-1].strip()
        target_text = text_parts[-1].strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": bev_image},
                    {
                        "type": "text",
                        "text": (
                            "North-aligned bird's-eye-view aerial image.\n"
                            f"Directionl prior: {direction_text}\n"
                            f"Target: {target_text}\n"
                            "Identify scene regions, landmarks and sturctures "
                            "relevant for locating the target and guiding navigation toward it."
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[bev_image], return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    # ── Core forward ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(
        self,
        bev_image,
        lang_description: str,
        low_state: torch.Tensor,
        high_state: torch.Tensor,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """
        Run SmolVLM2 with BEV image + language text, appending high/low UAV
        state tokens, and return the decoder's last_hidden_state.

        Token layout fed to the (truncated) decoder
        --------------------------------------------
        [BEV image tokens] [language tokens] [high_state_token] [low_state_token]

        Parameters
        ----------
        bev_image        : PIL.Image or np.ndarray (H,W,3) RGB
        lang_description : str
        low_state        : Tensor [B, >=4] — low UAV state (x,y,z,heading,...)
        high_state       : Tensor [B, >=4] — high UAV state (x,y,z,heading,...)
        device           : torch.device

        Returns
        -------
        Tensor [B, T+2, hidden_size] — last_hidden_state of the truncated decoder
        """
        low_state_token  = self.pose_token_proj(low_state).to(device)
        high_state_token = self.pose_token_proj(high_state).to(device)

        proc                 = self.build_processor_inputs(bev_image, lang_description, device)
        input_ids            = proc["input_ids"]
        pixel_values         = proc["pixel_values"]
        pixel_attention_mask = proc["pixel_attention_mask"]
        attn_mask            = proc["attention_mask"]

        inputs_embeds = self.embed_tokens(input_ids)
        image_embeds  = self.embed_image(pixel_values, pixel_attention_mask)

        inputs_embeds = self.vlm.model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_embeds,
        )

        # Append state tokens — high first so low_state attends to high context
        inputs_embeds = torch.cat(
            [inputs_embeds, high_state_token, low_state_token],
            dim=1,
        )

        attention_mask = torch.cat(
            [attn_mask, torch.ones((1, 2), dtype=attn_mask.dtype, device=device)],
            dim=1,
        )

        lm_out = self.vlm.model.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )

        return lm_out.last_hidden_state
