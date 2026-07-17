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

The VLM sees image + language only.  UAV pose conditioning happens downstream:
PositionVertexBuilder cross-attends the returned hidden states with projected
high/low UAV pose tokens (PoseFeatureMerger) before PerceiverIO pooling.

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

import gc
import logging
import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from .config import AeroduoConfig
except ImportError:
    from config import AeroduoConfig  # direct script execution

logger = logging.getLogger(__name__)


# ── Fuzzy goal verbalisation ───────────────────────────────────────────────────
#
# Hal-13k compass frame (validated against the dataset's own directional priors
# on all 5377 episodes, and against BEV pixel flow via phase correlation):
#   +x = north = top of the BEV image,  +y = east = right of the image.
# A goal offset is (Δnorth, Δeast) in metres from the high UAV's current
# position to the target, so the bearing is atan2(Δeast, Δnorth) from north.
# The BEV footprint is ≈95 m × 95 m, which the distance buckets align with:
# "very close"/"close" targets are in view, "far away" targets are off-image.

_COMPASS_8 = [
    "north", "north-east", "east", "south-east",
    "south", "south-west", "west", "north-west",
]


def fuzzy_goal_phrase(d_north: float, d_east: float) -> str:
    """
    Verbalise a goal offset (Δnorth, Δeast in metres) as a fuzzy compass cue,
    e.g. "to the south-west, a moderate distance away".
    """
    dist = math.hypot(d_north, d_east)
    if dist < 10.0:
        return "almost directly below you, near the centre of the image"
    wind = _COMPASS_8[int(((math.degrees(math.atan2(d_east, d_north)) + 22.5) % 360.0) // 45.0)]
    if dist < 30.0:
        qual = "very close"
    elif dist < 80.0:
        qual = "close"
    elif dist < 160.0:
        qual = "a moderate distance away"
    else:
        qual = "far away"
    return f"to the {wind}, {qual}"


class SmolVLM2Encoder(nn.Module):
    """
    Thin wrapper around SmolVLM2 that:
      - loads the model and processor once, frozen
      - physically truncates the decoder to vlm_layer_cutoff layers
      - provides ``forward`` which fuses BEV image and language, then returns
        the decoder's last_hidden_state (pose conditioning happens downstream
        in PositionVertexBuilder)
    """

    def __init__(self, cfg: AeroduoConfig) -> None:
        super().__init__()
        self.cfg = cfg

        logger.info("Loading SmolVLM2 processor …")
        self.processor = AutoProcessor.from_pretrained(cfg.smolvlm2_model_name)
        self.processor.image_processor.do_image_splitting = False

        _dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        load_dtype = _dtype_map.get(
            getattr(cfg, "smolvlm2_load_dtype", "bfloat16"), torch.bfloat16
        )
        logger.info("Loading SmolVLM2 model (frozen, dtype=%s) …", load_dtype)
        self.vlm = AutoModelForImageTextToText.from_pretrained(
            cfg.smolvlm2_model_name,
            torch_dtype=load_dtype,
        ).to("cuda")
        self.vlm.eval()

        for param in self.vlm.parameters():
            param.requires_grad_(False)

        # Keep the full layer list in a local only: storing it on self would
        # register all 24 layers as a submodule and pin the truncated upper
        # half (~half the decoder) in GPU memory for the whole run.
        full_layers = self.vlm.model.text_model.layers
        n = len(full_layers)
        logger.info(f"SmolVLM2 has {n} decoder layers.")

        cutoff = cfg.vlm_layer_cutoff if cfg.vlm_layer_cutoff is not None else n // 2
        self.vlm_layer_cutoff = cutoff
        self.vlm.model.text_model.layers = full_layers[:cutoff]
        # The slice creates a fresh ModuleList container after vlm.eval() ran;
        # sync its (inert) training flag with the rest of the frozen backbone.
        self.vlm.model.text_model.layers.eval()
        del full_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Truncated decoder to {cutoff} layers (N//2 = {n // 2}).")

        self.hidden_size = self.vlm.config.text_config.hidden_size

    def train(self, mode: bool = True):
        # Frozen backbone: parent .train() calls propagate here (e.g.
        # policy.train() in the Stage 1 loop) — ignore them so the VLM stays
        # in eval mode permanently.  Mirrors LowVLMEncoder in low_uav.
        return self

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
        goal_offset: Optional[Sequence[float]] = None,
    ) -> dict:
        """
        Build SmolVLM2 processor inputs for one BEV frame.

        goal_offset : optional (Δnorth, Δeast) in metres from the high UAV's
            *current* position to the target.  When given, the prompt carries a
            per-frame fuzzy compass cue; otherwise it falls back to the static
            start-based directional prior parsed from ``language_text``.
        """
        text_parts = language_text.split(
            "The description of the target and its surrounding is shown below."
        )
        direction_text = text_parts[0].strip().split(
            "Compass north corresponds to the top of the bird's-eye-view image."
        )[-1].strip()
        target_text = text_parts[-1].strip()

        if goal_offset is not None:
            direction_line = (
                "From your present position the target lies "
                f"{fuzzy_goal_phrase(float(goal_offset[0]), float(goal_offset[1]))}.\n"
            )
        else:
            direction_line = f"Directional prior: {direction_text}\n"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": bev_image},
                    {
                        "type": "text",
                        "text": (
                            "You are the high-altitude scout of a two-UAV team, "
                            "guiding a low-flying teammate to a ground target.\n"
                            "The image is your north-aligned bird's-eye view: north is "
                            "the top of the image, east is the right.\n"
                            f"Target: {target_text}\n"
                            f"{direction_line}"
                            "Identify scene regions, landmarks and structures relevant "
                            "for locating the target and guiding navigation toward it: "
                            "note road layout, open corridors and obstacles between you "
                            "and the target, and any structure that matches the target "
                            "description."
                        ),
                    },
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = self.processor(text=prompt, images=[bev_image], return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}

    # ── Core forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        bev_image,
        lang_description: str,
        device: torch.device = torch.device("cuda"),
        goal_offset: Optional[Sequence[float]] = None,
    ) -> torch.Tensor:
        """
        Run SmolVLM2 with BEV image + language text and return the decoder's
        last_hidden_state.  UAV poses are NOT fed here — PositionVertexBuilder
        cross-attends the returned hidden states with projected pose tokens.

        Token layout fed to the (truncated) decoder
        --------------------------------------------
        [BEV image tokens] [language tokens]

        Parameters
        ----------
        bev_image        : PIL.Image or np.ndarray (H,W,3) RGB
        lang_description : str
        device           : torch.device
        goal_offset      : optional (Δnorth, Δeast) metres from the high UAV's
                           current position to the target — see
                           ``build_processor_inputs``

        Returns
        -------
        Tensor [B, S, hidden_size] — last_hidden_state of the truncated decoder
        """
        vlm_dtype = next(self.vlm.model.text_model.parameters()).dtype

        proc                 = self.build_processor_inputs(bev_image, lang_description, device, goal_offset)
        input_ids            = proc["input_ids"]
        pixel_values         = proc["pixel_values"]
        pixel_attention_mask = proc["pixel_attention_mask"]
        attention_mask       = proc["attention_mask"]

        inputs_embeds = self.embed_tokens(input_ids)
        image_embeds  = self.embed_image(pixel_values, pixel_attention_mask)

        inputs_embeds = self.vlm.model.inputs_merger(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_hidden_states=image_embeds,
        )

        lm_out = self.vlm.model.text_model(
            inputs_embeds=inputs_embeds.to(vlm_dtype),
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
            use_cache=False,
        )

        return lm_out.last_hidden_state
