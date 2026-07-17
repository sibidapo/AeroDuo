from __future__ import annotations

import math

import airsim
from collections import deque
from typing import List, Optional, TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image

from high_uav.config import AeroduoConfig
from high_uav.aeroduo_policy import AeroDuoPolicy
from high_uav.bev_segmentation import load_models
from low_uav.config.lowuavconfig import LowUAVConfig
from low_uav.lowuav_policy import LowUAVPolicy

if TYPE_CHECKING:
    from config.evalconfig import EvalConfig


class DualUAVPilot:
    def __init__(self,
                 cfg: EvalConfig,
                 airsim_client: airsim.MultirotorClient,
                 simple_actions: bool = False,
                 activate_drone2: bool = True) -> None:  ##TODO
        self.cfg = cfg
        self.device = cfg.low_uav.device
        self.airsim_client = airsim_client
        self.window_T = cfg.high_uav.window_T

        self._high_pose_mean = np.array(cfg.high_pose_mean, dtype=np.float64)
        self._high_pose_std  = np.array(cfg.high_pose_std,  dtype=np.float64)
        self._low_pose_mean  = np.array(cfg.low_pose_mean,  dtype=np.float64)
        self._low_pose_std   = np.array(cfg.low_pose_std,   dtype=np.float64)

        self._bev_window:       deque = deque(maxlen=self.window_T)   # PIL RGB images
        self._high_pose_window: deque = deque(maxlen=self.window_T)   # np [4] each
        self._low_pose_window:  deque = deque(maxlen=self.window_T)   # np [4] each

        # Episode-start positions for computing relative poses (set on first push).
        self._high_pose_origin: np.ndarray | None = None
        self._low_pose_origin:  np.ndarray | None = None

        self.policy = self.init_policy()

    def update_airsim_client(self, airsim_client: airsim.MultirotorClient):
        self.airsim_client = airsim_client

    def push(self, bev_np, high_pose, low_pose):
        bev  = self._bgr_to_pil(bev_np)
        high = high_pose.astype(np.float32)
        low  = low_pose.astype(np.float32)

        if len(self._bev_window) == 0:
            self._high_pose_origin = high[:3].copy()
            self._low_pose_origin  = low[:3].copy()
            for _ in range(self.window_T):
                self._bev_window.append(bev)
                self._high_pose_window.append(high)
                self._low_pose_window.append(low)
        else:
            self._bev_window.append(bev)
            self._high_pose_window.append(high)
            self._low_pose_window.append(low)

    def _normalize_pose(
        self,
        pose:   np.ndarray,   # [x, y, z, heading]
        origin: np.ndarray,   # [x0, y0, z0]
        mean:   np.ndarray,
        std:    np.ndarray,
    ) -> np.ndarray:
        rel = pose[:3] - origin
        return np.array([
            (rel[0] - mean[0]) / std[0],
            (rel[1] - mean[1]) / std[1],
            (rel[2] - mean[2]) / std[2],
            pose[3],
        ], dtype=np.float32)

    def reset(self) -> None:
        self._bev_window.clear()
        self._high_pose_window.clear()
        self._low_pose_window.clear()
        self._high_pose_origin = None
        self._low_pose_origin  = None

    def _load_stage1(
            self,
            ckpt_path: str,
            device: str = "cuda",
    ) -> AeroDuoPolicy:
        cfg = self.cfg.high_uav

        sam2_predictor, grounding_model, resolved_device = load_models(device)
        cfg_device = resolved_device if device == "auto" else device

        policy = AeroDuoPolicy(cfg, sam2_predictor, grounding_model)

        sd = torch.load(ckpt_path, map_location="cpu")
        policy.load_trainable_state_dict(sd, strict=True)

        policy.to(cfg_device)
        policy.eval()
        return policy
    
    def _load_stage2(
            self,
            ckpt_path: str,
            stage1_policy: Optional[AeroDuoPolicy],
            device: str = "cuda",
    ) -> LowUAVPolicy:
        cfg = self.cfg.low_uav

        policy = LowUAVPolicy(cfg, high_uav_policy=stage1_policy)

        sd = torch.load(ckpt_path, map_location="cpu")
        policy.load_trainable_state_dict(sd, strict=True)

        policy.action_head.to(device=device, dtype=torch.bfloat16)
        policy.eval()
        return policy

    def init_policy(self):
        stage1_ckpt = self.cfg.stage1_ckpt
        stage2_ckpt = self.cfg.stage2_ckpt

        if self.cfg.low_uav.use_zgraph:
            print("Loading Stage 1 policy …")
            stage1 = self._load_stage1(stage1_ckpt, device=self.device)
        else:
            print("use_zgraph=False: standalone low-UAV eval, skipping Stage 1.")
            stage1 = None

        print("Loading Stage 2 policy …")
        stage2 = self._load_stage2(stage2_ckpt, stage1, device=self.device)

        return stage2

    def get_action(self) -> None:
        return

    def navigate(self) -> None:
        return

    def get_current_poses(self):
        kin1 = self.airsim_client.getMultirotorState(vehicle_name="Drone_1").kinematics_estimated
        p1, o1 = kin1.position, kin1.orientation
        low_pose = self._state_to_pose({
            "position":    [p1.x_val, p1.y_val, p1.z_val],
            "orientation": [o1.w_val, o1.x_val, o1.y_val, o1.z_val],
        })

        kin2 = self.airsim_client.getMultirotorState(vehicle_name="Drone_2").kinematics_estimated
        p2, o2 = kin2.position, kin2.orientation
        high_pose = self._state_to_pose({
            "position":    [p2.x_val, p2.y_val, p2.z_val],
            "orientation": [o2.w_val, o2.x_val, o2.y_val, o2.z_val],
        })
        return high_pose, low_pose

    def _state_to_pose(self, state: dict) -> np.ndarray:
        pos = state["position"][:3]
        yaw = self._quat_to_yaw(state["orientation"])
        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def _quat_to_yaw(self, orientation) -> float:
        w, x, y, z = orientation[0], orientation[1], orientation[2], orientation[3]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _bgr_to_pil(self, bgr_np: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(bgr_np, cv2.COLOR_BGR2RGB))

    def _normalize_pose(
        self,
        pose:   np.ndarray,   # [x, y, z, heading]
        origin: np.ndarray,   # [x0, y0, z0]
        mean:   np.ndarray,
        std:    np.ndarray,
    ) -> np.ndarray:
        rel = pose[:3] - origin
        return np.array([
            (rel[0] - mean[0]) / std[0],
            (rel[1] - mean[1]) / std[1],
            (rel[2] - mean[2]) / std[2],
            pose[3],
        ], dtype=np.float32)

    def _padded_window(self, window: deque) -> list:
        if len(window) == 0:
            raise RuntimeError("DualUAVPilot: push() must be called at least once before predict()")
        return list(window)

    @torch.no_grad()
    def predict(
        self,
        instruction: str,
        front_bgr_np: np.ndarray,
        low_pose_current_xyzh: np.ndarray,
        goal_position: np.ndarray | list | None = None,  # world-frame target [x, y, ...]
    ) :

        bev_images = self._padded_window(self._bev_window)

        # Per-frame goal offsets (Δnorth, Δeast) m for the fuzzy prompt cue:
        # target minus the raw (world-frame) high-UAV position at each window step.
        # Same frame as training (+x = north, +y = east); None → prompt falls
        # back to the static directional prior.
        goal_offsets = None
        if goal_position is not None:
            goal_xy = np.asarray(goal_position, dtype=np.float32)[:2]
            goal_offsets = [torch.tensor(
                np.stack([goal_xy - p[:2] for p in self._padded_window(self._high_pose_window)]),
                dtype=torch.float32,
            )]  # [B=1] × [T, 2]

        high_poses = np.stack([
            self._normalize_pose(p, self._high_pose_origin, self._high_pose_mean, self._high_pose_std)
            for p in self._padded_window(self._high_pose_window)
        ])  # [T, 4]

        low_poses = np.stack([
            self._normalize_pose(p, self._low_pose_origin, self._low_pose_mean, self._low_pose_std)
            for p in self._padded_window(self._low_pose_window)
        ])  # [T, 4]

        low_pose_current_norm = self._normalize_pose(
            low_pose_current_xyzh, self._low_pose_origin, self._low_pose_mean, self._low_pose_std
        )

        front_pil = self._bgr_to_pil(front_bgr_np)

        high_uav_poses       = torch.tensor(high_poses, dtype=torch.float32)
        low_uav_poses_window = torch.tensor(low_poses, dtype=torch.float32)
        low_uav_pose_current = torch.tensor(low_pose_current_norm, dtype=torch.float32)

        actions_tensor = self.policy.get_action(
            bev_images = [bev_images],          # [B=1][T]
            low_uav_front_image = [front_pil],           # [B=1]
            high_uav_poses = high_uav_poses.unsqueeze(0),        # [1, T, 4]
            low_uav_poses_window = low_uav_poses_window.unsqueeze(0),  # [1, T, 4]
            low_uav_pose_current = low_uav_pose_current.unsqueeze(0),  # [1, 4]
            instruction = [instruction],
            device = torch.device(self.device),
            goal_offsets = goal_offsets,        # [B=1] × [T, 2] or None
        )

        actions = actions_tensor[0].cpu().float().numpy()   # [H, 5]
        xyz = actions[:, :3] * self._low_pose_std + self._low_pose_mean + self._low_pose_origin  # [H, 3]
        heading = np.arctan2(actions[:, 3], actions[:, 4])

        waypoints = np.concatenate([xyz, heading[:, None]], axis=1).tolist()  # [H, 4] = [x, y, z, h]
        return waypoints