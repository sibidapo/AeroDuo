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
    D_g: int = 1024

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

    # ── Flow matching (GR00T-style head — mirrors low_uav LowUAVActionHead) ──
    # Raw trajectory state from the dataset: (x, y, z, heading_rad).
    action_dim: int = 4

    # Flow-space dims after sin/cos heading encoding inside the head:
    # (x, y, z, sin_h, cos_h).  Noise, the ODE path and the MSE loss all live
    # in this 5-dim space; decode heading with atan2(sin_h, cos_h).
    flow_state_dim: int = 5
    flow_action_dim: int = 5

    # Denoising horizon H — number of future timesteps predicted jointly.
    action_horizon: int = 8

    # DiT denoiser — architecture mirrors low_uav DITConfig but sized down
    # (2026-07-14 rebalance): the head was 583M vs a 52M encoder (11:1), so
    # the DiT could absorb the trajectory prior and starve z_graph of
    # gradient.  Now ~80M head vs ~74M encoder (≈1:1), matched to Hal-13k.
    # Interleaved cross/self attention blocks (even idx: cross-attend z_graph,
    # odd idx: self-attend over [state | action] tokens).  Keep layers even.
    flow_matching_layers: int = 8

    flow_matching_heads: int = 16

    flow_matching_head_dim: int = 48

    # DiT inner dim; must equal flow_matching_heads * flow_matching_head_dim.
    flow_input_emb_dim: int = 768

    # DiT output dim → action_decoder MLP input.
    flow_output_dim: int = 512

    # Hidden dim of the state/action encoder and action decoder MLPs.
    flow_hidden_dim: int = 512

    # 0.2 regularized the old 583M head; on ~80M it would underfit.
    flow_matching_dropout: float = 0.1

    # Q/K/V projection bias inside diffusers.Attention.
    flow_matching_attention_bias: bool = True

    # FFN activation; "gelu-approximate" matches GR00T default.
    flow_matching_activation: str = "gelu-approximate"

    # Sinusoidal positional-embedding table inside each DiT block.
    flow_max_num_positional_embeddings: int = 512

    # Learned positional embedding added to action tokens before the DiT.
    flow_add_pos_embed: bool = True
    flow_max_seq_len: int = 1024

    # ── Flow matching — τ discretization and sampling (GR00T convention) ─────
    # Discretization of τ ∈ [0,1] for diffusers.Timesteps (integer buckets).
    num_timestep_buckets: int = 1000

    # τ = (1 − Beta(α, β)) · noise_s — biased toward the noisy end (τ ≈ 0).
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.99

    # Euler integration steps in FlowMatchingNetwork.predict_action.
    num_inference_timesteps: int = 4

    # Probability of omitting the low-UAV state token from the DiT input
    # sequence during training.  The current low-UAV pose is also present in
    # z_graph (low_uav_poses_window[t_end] is fused into the place nodes), so
    # dropping the direct state path forces the DiT to recover it through
    # cross-attention to the graph instead of shortcutting from the state
    # token.  0.0 disables; predict_action always keeps the state token.
    state_dropout_p: float = 0.25

    # ── Pose feature merger (position vertex pose conditioning) ───────────────
    # Cross-attention block inside PositionVertexBuilder: VLM hidden states
    # (queries) attend over a 2-token KV context built from the projected
    # high/low UAV poses.  Mirrors FeatureMerger in the low-UAV action head.
    pose_merger_hidden_dim: int = 2048

    # Pose projection dim (K/V input): each pose is sin/cos-encoded to 5-dim
    # then projected Linear(5, pose_merger_kv_dim).
    pose_merger_kv_dim: int = 2048

    pose_merger_n_head: int = 8

    pose_merger_dropout: float = 0.0

    # ── Perceiver IO (position vertex compression) ────────────────────────────
    # Widened in the 2026-07-14 rebalance: this is the narrowest point of the
    # pipeline ([S≈189, 2048] VLM tokens → one D_g place node per timestep).
    # Number of latent vectors inside the Perceiver IO bottleneck.
    perceiver_M: int = 128

    # Dimension of each latent vector.
    perceiver_D_latent: int = 512

    # Number of (cross-attention → latent self-attention) rounds.
    perceiver_depth: int = 3

    # Attention heads in latent self-attention.
    perceiver_n_heads: int = 8

    # ── Graph encoder ─────────────────────────────────────────────────────────
    graph_encoder_layers: int = 4
    graph_encoder_heads: int = 4

    # ── Observation vertices ──────────────────────────────────────────────────
    # Cap on unique detected categories per timestep (goal + up to 4 contextual).
    max_obs_vertices: int = 5

    # ── Training precision ────────────────────────────────────────────────────
    # Mixed-precision mode forwarded to Accelerate: "bf16", "fp16", or "no".
    mixed_precision: str = "bf16"

    # dtype used when loading SmolVLM2 weights.
    # "bfloat16" halves VRAM vs "float32" with no practical accuracy loss
    # for frozen inference.  Set to "float32" if you observe instability.
    smolvlm2_load_dtype: str = "bfloat16"

    # ── LoRA (Stage 4) ────────────────────────────────────────────────────────
    # Rank for LoRA adapters on SmolVLM2 attention projections.
    # None → no LoRA; set to an integer (e.g. 16) in Stage 4.
    lora_rank: Optional[int] = None
