"""
aeroduo_policy.py — AeroDuo Stage 1/2 policy model.

Self-contained: owns every frozen and trainable component of the high-UAV
pipeline.  The training loop passes only what AeroduoDataset.collate_fn returns
and gets back a scalar loss.

Full pipeline
-------------
Input from dataset / training loop
    bev_images            List[PIL.Image]  length T
    high_uav_poses        [T, 4]
    low_uav_poses_window  [T, 4]
    low_uav_pose_current  [4]
    instruction           str
    low_uav_traj_target   [H, 4]  (None at inference)

BEVEncoder  (frozen, not an nn.Module)
    GroundingDINO detect   — backbone once per window, per-prompt batched over T
    SAM2 set_image_batch   — ONE Hiera forward for all T frames
    SAM2 predict           — per-frame mask decode (cheap)
  → image_embeds [T, 256, 64, 64]   detached
  → masks_arrays  List[ndarray [N,H,W]]
  → detections_list List[List[dict]]

SmolVLM2Encoder  (frozen, nn.Module)
    VLM(BEV images, instruction)  — image + text only, no pose tokens;
                                    one forward batched over the T-frame window
  → hidden_states T × [S_t, 2048]  (ragged: goal cue varies per frame)

PositionVertexBuilder  (trainable — PoseFeatureMerger + PerceiverIO + output_query)
    hidden_states cross-attend over projected high/low UAV pose tokens,
    then PerceiverIO pools the fused sequence
  → place_nodes [T, D_g]

ObservationVertexBuilder  (trainable — obs_projector)
    image_embed[t:t+1] + masks + detections  →  List[ObsVertex]
  → obs_tensor [T, K_max, D_g]

GraphEncoder  (trainable — HGTConv × 3)
  → z_graph [T, D_g]

FlowMatchingNetwork  (trainable, Stage 1 only — GR00T-style, mirrors
                      low_uav LowUAVActionHead; DiT cross-attends to z_graph)
    z_graph + low_state + low_actions
  → noise + τ ~ (1 − Beta(α, β)) · noise_s sampled inside the head
  → L_flow = MSE(v_pred, clean − noise) in (x, y, z, sin_h, cos_h) space

Trainable / Frozen split
------------------------
Frozen (excluded from optimizer and trainable_state_dict):
    BEVEncoder  (GroundingDINO + SAM2 — plain Python objects, not nn.Parameters)
    SmolVLM2Encoder

Trainable:
    place_node_builder  PoseFeatureMerger + PerceiverIO + output_query
    obs_vertex_builder  obs_projector (Linear + LayerNorm)
    graph_encoder       HGTConv layers + LayerNorms
    flow_net            full DiT denoiser

τ convention (flow matching — sampled inside FlowMatchingNetwork)
-----------------------------------------------------------------
    τ = 0  →  pure noise
    τ = 1  →  clean data
    τ ~ (1 − Beta(α, β)) · noise_s
    x_τ = τ · clean + (1 − τ) · noise
    v_target = clean − noise
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    from .config import AeroduoConfig
    from .smolvlm2_encoder import SmolVLM2Encoder
    from .position_vertex_builder import PositionVertexBuilder
    from .observation_vertex import ObservationVertexBuilder, ObsVertex
    from .graph_encoder import GraphEncoder
    from .flow_matching import FlowMatchingNetwork
    from .bev_encoder import BEVEncoder
except ImportError:
    from config import AeroduoConfig
    from smolvlm2_encoder import SmolVLM2Encoder
    from position_vertex_builder import PositionVertexBuilder
    from observation_vertex import ObservationVertexBuilder, ObsVertex
    from graph_encoder import GraphEncoder
    from flow_matching import FlowMatchingNetwork
    from bev_encoder import BEVEncoder


class AeroDuoPolicy(nn.Module):
    """
    Stage 1 / Stage 2 AeroDuo policy.

    Parameters
    ----------
    cfg             : AeroduoConfig — single source-of-truth for hyperparameters
    sam2_predictor  : SAM2ImagePredictor — loaded on target device by the training loop
    grounding_model : GroundingDINO model — loaded on target device by the training loop

    Notes
    -----
    *  BEVEncoder is a plain Python object (not nn.Module): GroundingDINO and SAM2
       are never registered as submodules, so they do not appear in state_dict()
       or parameters(), and Accelerate does not try to move them.

    *  SmolVLM2Encoder is an nn.Module whose parameters are all frozen
       (requires_grad=False).  It is registered as a submodule so .to(device)
       moves it correctly, but the optimizer receives no gradients through it.

    *  Mixed precision is handled by Accelerate (accelerator.prepare).
       trainable_state_dict / load_trainable_state_dict enable checkpointing
       only the trainable subset — the large frozen models are excluded.
    """

    def __init__(
        self,
        cfg: AeroduoConfig,
        sam2_predictor,
        grounding_model,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # ── Frozen: GroundingDINO + SAM2  (plain Python, not nn.Module) ───────
        # Stored as a plain attribute so PyTorch never registers these weights.
        self.bev_encoder = BEVEncoder(sam2_predictor, grounding_model)

        # ── Frozen: SmolVLM2 body (nn.Module) — image + text only ─────────────
        self.vlm_encoder = SmolVLM2Encoder(cfg)

        # ── Trainable: pose cross-attention fusion + PerceiverIO compression ──
        # [T, S, 2048] + high/low poses → [T, D_g]
        self.place_node_builder = PositionVertexBuilder(cfg)

        # ── Trainable: SAM2 mask-pool → graph observation node features ────────
        self.obs_vertex_builder = ObservationVertexBuilder(cfg)

        # ── Trainable: HGT heterogeneous graph encoder ─────────────────────────
        self.graph_encoder = GraphEncoder(
            D_g=cfg.D_g,
            num_layers=cfg.graph_encoder_layers,
            heads=cfg.graph_encoder_heads,
        )

        # ── Trainable: flow-matching trajectory denoiser ───────────────────────
        self.flow_net = FlowMatchingNetwork(cfg)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _trainable_dtype(self) -> torch.dtype:
        try:
            return next(self.place_node_builder.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _stack_obs_vertices(
        self,
        all_obs: List[List[ObsVertex]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Pack per-timestep ObsVertex lists into zero-padded [T, K_max, D_g]."""
        T     = len(all_obs)
        K_max = self.cfg.max_obs_vertices
        D_g   = self.cfg.D_g
        obs_tensor = torch.zeros(T, K_max, D_g, device=device, dtype=dtype)
        for t, obs_t in enumerate(all_obs):
            for k, obs in enumerate(obs_t[:K_max]):
                obs_tensor[t, k] = obs.feature.to(device=device, dtype=dtype)
        return obs_tensor

    # ── Graph encoding (used by both training and Stage 2 inference) ───────────

    def encode_graph(
        self,
        bev_images:           List,              # [B][T] List[List[PIL]] or [T] List[PIL]
        high_uav_poses:       torch.Tensor,      # [B, T, 4] or [T, 4]
        low_uav_poses_window: torch.Tensor,      # [B, T, 4] or [T, 4]
        instruction,                             # List[str] length B or str
        device:               torch.device,
        goal_offsets:         Optional[List] = None,  # [B] × ([T, 2] or None), or [T, 2]
    ) -> torch.Tensor:                            # [B, T, D_g]
        """
        Run the full BEV → graph pipeline and return z_graph [B, T, D_g].

        Accepts both a single episode (str instruction, [T,4] poses, [T] images)
        and a batch of B episodes (List[str], [B,T,4], [B][T] images).
        Single-episode inputs are promoted to B=1 internally and the result is
        returned with the leading batch dimension kept (for consistency with
        AeroDuoPolicy.forward, which always returns [B, …]).

        Pipeline
        --------
        Frozen (loop over B):
          1. BEVEncoder     → image_embeds [T,256,64,64], masks, detections  (per episode)
          2. SmolVLM2       → hidden_states T × [S_t, 2048]  (image + text only,
                              one batched forward per episode)
        Trainable (batched over B):
          3. PoseFeatureMerger + PerceiverIO
                            → place_nodes  [B, T, D_g]  (hidden states cross-attend
                              over projected high/low poses, then pooled)
          4. ObsVertexBuild → obs_tensor   [B, T, K_max, D_g]
          5. GraphEncoder   → z_graph      [B, T, D_g]
        """
        # ── Normalise to batched form ─────────────────────────────────────────
        if isinstance(instruction, str):
            instruction = [instruction]
            bev_images  = [bev_images]
            high_uav_poses       = high_uav_poses.unsqueeze(0)
            low_uav_poses_window = low_uav_poses_window.unsqueeze(0)
            if goal_offsets is not None:
                goal_offsets = [goal_offsets]

        B     = len(instruction)
        T     = len(bev_images[0])
        dtype = self._trainable_dtype()

        # ── Step 1 & 2: frozen models — loop over B episodes ─────────────────
        all_image_embeds: List[torch.Tensor]       = []   # B × Tensor[T,256,64,64]
        all_masks:        List[List]               = []   # B × T × ndarray[N,H,W]
        all_detections:   List[List]               = []   # B × T × List[dict]
        all_hidden:       List[List[torch.Tensor]] = []   # B × T × Tensor[S_bt, 2048]

        for b in range(B):
            # BEVEncoder: one Hiera forward for all T frames in this episode
            img_embs_b, masks_b, dets_b = self.bev_encoder(
                bev_images[b], instruction[b], device
            )   # img_embs_b: [T, 256, 64, 64]
            all_image_embeds.append(img_embs_b)
            all_masks.append(masks_b)
            all_detections.append(dets_b)

            # SmolVLM2: one batched forward for all T frames of this episode.
            # Returned sequences are ragged: the goal-cue text
            # (fuzzy_goal_phrase) tokenizes to a different length depending on
            # distance/bearing, so frames within a window (and across the
            # batch) vary in S.
            go_b = goal_offsets[b] if goal_offsets is not None else None
            hs_b = self.vlm_encoder(
                bev_images=bev_images[b],
                lang_description=instruction[b],
                device=device,
                goal_offsets=go_b,
            )                                  # T × [S_bt, 2048]
            all_hidden.append(hs_b)

        # ── Step 3: pose fusion + PerceiverIO [B, T, S, 2048] → [B, T, D_g] ──
        # Right-pad every frame's hidden sequence to the batch-wide max S and
        # build a validity mask so PerceiverIO's pooling ignores the padding.
        S_max = max(h.shape[0] for hs_b in all_hidden for h in hs_b)
        position_vertices = torch.zeros(
            B, T, S_max, self.cfg.smolvlm2_hidden_dim, device=device, dtype=dtype
        )
        hidden_mask = torch.zeros(B, T, S_max, dtype=torch.bool, device=device)
        for b in range(B):
            for t in range(T):
                h = all_hidden[b][t]
                s = h.shape[0]
                position_vertices[b, t, :s] = h.to(device=device, dtype=dtype)
                hidden_mask[b, t, :s] = True

        place_nodes = self.place_node_builder(
            position_vertices,
            high_uav_poses.to(device=device, dtype=dtype),           # [B, T, 4]
            low_uav_poses_window.to(device=device, dtype=dtype),     # [B, T, 4]
            mask=hidden_mask,                                        # [B, T, S_max]
        )                                                             # [B, T, D_g]

        # ── Step 4: observation vertices [B, T, K_max, D_g] ──────────────────
        obs_list: List[List[List[ObsVertex]]] = []
        for b in range(B):
            obs_b: List[List[ObsVertex]] = []
            for t in range(T):
                obs_t = self.obs_vertex_builder(
                    sam2_predictor=None,
                    masks_array=all_masks[b][t],
                    detections=all_detections[b][t],
                    device=device,
                    image_embed=all_image_embeds[b][t:t + 1],  # [1, 256, 64, 64]
                )
                obs_b.append(obs_t)
            obs_list.append(obs_b)

        obs_tensor = torch.stack([
            self._stack_obs_vertices(obs_list[b], device=device, dtype=dtype)
            for b in range(B)
        ])   # [B, T, K_max, D_g]

        # ── Step 5: HGT graph encoding → z_graph [B, T, D_g] ─────────────────
        z_graph = self.graph_encoder(place_nodes, obs_tensor)
        return z_graph

    # ── Training / inference forward ───────────────────────────────────────────

    def forward(
        self,
        bev_images:            List,                            # [B][T] or [T]
        high_uav_poses:        torch.Tensor,                    # [B, T, 4] or [T, 4]
        low_uav_poses_window:  torch.Tensor,                    # [B, T, 4] or [T, 4]
        low_uav_pose_current:  torch.Tensor,                    # [B, 4] or [4]
        instruction,                                            # List[str] or str
        device:                torch.device,
        low_uav_traj_target:   Optional[torch.Tensor] = None,   # [B, H, 4] or [H, 4]
        goal_offsets:          Optional[List] = None,           # [B] × ([T, 2] or None), or [T, 2]
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1 training:  provide ``low_uav_traj_target``
            → {"loss": scalar, "z_graph": [B, T, D_g]}
            (noise and τ are sampled inside FlowMatchingNetwork, GR00T-style)

        Stage 2 inference: omit ``low_uav_traj_target``
            → {"z_graph": [B, T, D_g]}
            (sample trajectories with policy.flow_net.predict_action(z_graph,
            low_state) → [B, H, 5] = (x, y, z, sin_h, cos_h))

        Accepts both single-episode (B=1 sugar) and batched (B > 1) inputs.
        The training loop can call policy(**batch, device=accelerator.device)
        where batch comes from AeroduoDataset.collate_fn (which now returns
        [B, T, 4] tensors for any batch_size).
        """
        # ── Normalise single-episode sugar to batched form ────────────────────
        if isinstance(instruction, str):
            instruction           = [instruction]
            bev_images            = [bev_images]
            high_uav_poses        = high_uav_poses.unsqueeze(0)
            low_uav_poses_window  = low_uav_poses_window.unsqueeze(0)
            low_uav_pose_current  = low_uav_pose_current.unsqueeze(0)
            if low_uav_traj_target is not None:
                low_uav_traj_target = low_uav_traj_target.unsqueeze(0)
            if goal_offsets is not None:
                goal_offsets = [goal_offsets]

        z_graph = self.encode_graph(
            bev_images=bev_images,
            high_uav_poses=high_uav_poses,
            low_uav_poses_window=low_uav_poses_window,
            instruction=instruction,
            device=device,
            goal_offsets=goal_offsets,
        )   # [B, T, D_g]

        out: Dict[str, torch.Tensor] = {"z_graph": z_graph}

        if low_uav_traj_target is None:
            return out

        # ── Flow matching training ─────────────────────────────────────────────
        # Noise, τ ~ (1 − Beta(α, β)) · noise_s, the sin/cos heading encoding
        # and the MSE loss all live inside the head (GR00T-style, mirroring
        # LowUAVActionHead); the head also casts inputs to its own dtype.
        out["loss"] = self.flow_net(
            z_graph=z_graph,
            low_state=low_uav_pose_current,
            low_actions=low_uav_traj_target,
        )

        return out

    @torch.no_grad()
    def get_graph(
        self,
        bev_images:           List,              # [B][T] List[List[PIL]] or [T] List[PIL]
        high_uav_poses:       torch.Tensor,      # [B, T, 4] or [T, 4]
        low_uav_poses_window: torch.Tensor,      # [B, T, 4] or [T, 4]
        instruction,                             # List[str] length B or str
        device:               torch.device,
        goal_offsets:         Optional[List] = None,  # [B] × ([T, 2] or None), or [T, 2]
    ) -> torch.Tensor:                            # [B, T, D_g]
        """
        Frozen accessor for Stage 2: encode_graph() under no_grad.

        Returns a detached z_graph [B, T, D_g] so downstream losses cannot
        backprop into Stage 1.  The caller is responsible for having put the
        policy in eval() mode (Stage 2 train.py does this after loading the
        checkpoint).
        """
        return self.encode_graph(
            bev_images=bev_images,
            high_uav_poses=high_uav_poses,
            low_uav_poses_window=low_uav_poses_window,
            instruction=instruction,
            device=device,
            goal_offsets=goal_offsets,
        )

    # ── Selective checkpointing ────────────────────────────────────────────────

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        State dict containing ONLY the trainable submodules.

        Excludes BEVEncoder (GroundingDINO + SAM2) and SmolVLM2Encoder — saves
        ~10-20 GB per checkpoint compared to policy.state_dict().
        """
        sd: Dict[str, torch.Tensor] = {}
        for prefix, module in self._trainable_modules():
            for k, v in module.state_dict().items():
                sd[f"{prefix}.{k}"] = v
        return sd

    def load_trainable_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        """Load a checkpoint produced by trainable_state_dict()."""
        for prefix, module in self._trainable_modules():
            prefix_dot = f"{prefix}."
            sub_sd = {
                k[len(prefix_dot):]: v
                for k, v in state_dict.items()
                if k.startswith(prefix_dot)
            }
            if sub_sd:
                module.load_state_dict(sub_sd, strict=strict)

    def _trainable_modules(self):
        # place_node_builder includes PoseFeatureMerger (pose projectors +
        # cross-attention) alongside the PerceiverIO pooling.
        yield "place_node_builder", self.place_node_builder
        yield "obs_vertex_builder", self.obs_vertex_builder
        yield "graph_encoder",      self.graph_encoder
        yield "flow_net",           self.flow_net
