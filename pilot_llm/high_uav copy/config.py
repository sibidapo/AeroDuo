"""
config.py — Centralised configuration dataclass for the aeroduo two-UAV pipeline.

This is the *single source of truth* for all hyperparameters.  Every module
from Stage 2 through Stage 5 imports from here; nothing is hardcoded elsewhere.

Key design decisions
--------------------
* ``sam2_feature_dim`` and ``sam2_spatial_res`` are **derived from the SAM2
  checkpoint** rather than hardcoded: the SAM-2.1-hiera-large FPN neck
  projects all trunk scales to 256 channels.  The highest-resolution feature
  map stored in ``predictor._features["high_res_feats"][0]`` has spatial
  resolution 256×256 (for a 1024-px input image).
* ``smolvlm2_hidden_dim`` and ``vlm_layer_cutoff`` are derived from the
  SmolVLM2-2.2B-Instruct checkpoint config.json:
      hidden_size = 2048, num_hidden_layers = 24 → cutoff = 12.
* ``lora_rank`` is ``None`` until Stage 4 activates LoRA adapters.
* ``high_uav_pose_dim`` = 5 because heading is encoded as (sin, cos), giving
  a 5-dim vector: (x, y, z, sin_heading, cos_heading).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AeroduoConfig:
    # ── Graph embedding ────────────────────────────────────────────────────────
    # Shared output dimensionality for all graph nodes (V_t and o_k).
    D_g: int = 256

    # ── SmolVLM2 (position vertex encoder) ────────────────────────────────────
    smolvlm2_model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    # Hidden size of the LLaMA text decoder inside SmolVLM2-2.2B-Instruct.
    # Source: config.json → text_config → hidden_size.
    smolvlm2_hidden_dim: int = 2048

    # Layer index at which to read out V_t hidden states (0-indexed).
    # num_hidden_layers = 24 → cutoff = 24 // 2 = 12.
    vlm_layer_cutoff: int = 12

    # ── UAV state projectors ───────────────────────────────────────────────────
    # Heading is encoded as (sin, cos), so raw (x, y, z, heading) becomes a
    # 5-dim vector before projection: (x, y, z, sin_h, cos_h).
    high_uav_pose_dim: int = 5

    # (x, y, z, sin_h, cos_h, vx, vy) — adjust for your dataset; must match
    # the actual number of elements fed to LowUAVStateProjector at runtime.
    low_uav_state_dim: int = 7

    # ── SAM2 observation vertex features ──────────────────────────────────────
    # Number of channels in the SAM2 FPN-neck output feature maps.
    # Source: FpnNeck convs all project to 256 channels (confirmed for
    # sam2.1_hiera_large.pt).
    sam2_feature_dim: int = 256

    # Spatial resolution (H = W) of the highest-resolution feature map stored
    # in predictor._features["high_res_feats"][0] after set_image on a 1024-px
    # input.  Shape: [1, sam2_feature_dim, sam2_spatial_res, sam2_spatial_res].
    sam2_spatial_res: int = 256

    # ── Graph encoder ─────────────────────────────────────────────────────────
    # Number of timesteps in the sliding window fed to GraphEncoder.
    window_T: int = 5

    # ── Flow matching ─────────────────────────────────────────────────────────
    # Trajectory state dimensionality — matches normalized_state in the dataset:
    # (x, y, z, heading_rad).
    action_dim: int = 4

    # Denoising horizon H — number of future timesteps predicted jointly.
    action_horizon: int = 8

    # Number of FlowDenoiserBlock layers.
    flow_matching_layers: int = 4

    # Attention heads inside FlowDenoiserBlock; D_g must be divisible by this.
    flow_matching_heads: int = 4

    # FFN inner dim multiplier: FFN hidden dim = D_g * flow_matching_ffn_mult.
    flow_matching_ffn_mult: int = 4

    # ── Flow matching — additional DiT / GR00T-aligned options ──────────────
    # Discretization of τ ∈ [0,1] for diffusers.Timesteps (integer buckets).
    num_timestep_buckets: int = 1000

    # Q/K/V projection bias inside diffusers.Attention.
    flow_matching_attention_bias: bool = True

    # FFN activation; "gelu-approximate" matches GR00T default.
    flow_matching_activation: str = "gelu-approximate"

    # ── LoRA (Stage 4) ────────────────────────────────────────────────────────
    # Rank for LoRA adapters on SmolVLM2 attention projections.
    # None → no LoRA; set to an integer (e.g. 16) in Stage 4.
    lora_rank: Optional[int] = None
