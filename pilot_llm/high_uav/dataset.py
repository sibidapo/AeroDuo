"""
dataset.py — AeroduoDataset for Stage 1 training.

One training sample = a sliding window of T consecutive BEV frames (high-UAV
context) aligned with H subsequent low-UAV trajectory steps (flow-matching
supervision target).

Data layout (Hal-13k):
    <dataset_root>/<town>/<episode_id>/
        bevcamera/                         <- BEV PNG frames (high UAV), sorted by name
        high_uav_traj.json                 <- {normalized_state: [[x,y,z,heading], ...]}
        low_uav_traj.json                  <- {normalized_state: [[x,y,z,heading], ...]}
        object_description_with_help.json  <- [instruction_string]

Sampling strategy:
    N_high = len(bevcamera frames) = len(high_uav_traj.normalized_state)
    N_low  = len(frontcamera frames) = len(low_uav_traj.normalized_state)
    n_overlap = min(N_high, N_low)  <- temporal range where both UAVs have data

    A sample anchored at t_end is valid when:
        t_end in [T - 1,  n_overlap - H - 1]

    This gives a T-frame BEV history [t_end-T+1 .. t_end] and H future
    low-UAV steps [t_end+1 .. t_end+H].

    All (episode_idx, t_end) pairs are enumerated at __init__ time so the
    DataLoader can shuffle and report a meaningful __len__.

Important constraints:
    - num_workers must be 0 (SmolVLM2 / SAM2 / GroundingDINO are not picklable).
    - collate_fn asserts batch_size == 1.  Each training step is one window.
    - Images are loaded lazily inside __getitem__ (PIL, RGB) — the training
      loop feeds them one-at-a-time to SmolVLM2Encoder and SAM2.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# ── Episode metadata (loaded once at dataset construction) ────────────────────

@dataclass
class _EpisodeMeta:
    episode_path: Path
    instruction: str
    high_poses: np.ndarray       # [N_high, 4] — normalized_state (x,y,z,heading)
    low_poses: np.ndarray        # [N_low,  4] — normalized_state
    bev_frame_paths: List[Path]  # sorted; len == N_high; positional index == traj index
    n_overlap: int               # min(N_high, N_low)


# ── Dataset ───────────────────────────────────────────────────────────────────

class AeroduoDataset(Dataset):
    """
    PyTorch Dataset over the Hal-13k episode collection.

    Parameters
    ----------
    dataset_root : str or Path
        Root directory containing one subdirectory per Carla town.
    window_T : int
        Number of consecutive BEV frames that form the high-UAV observation
        window (T=5 in Stage 1).
    action_horizon : int
        Number of future low-UAV trajectory steps used as the flow-matching
        supervision target (H=8 in Stage 1).  Must satisfy H >= 1.
    towns : list[str] or None
        If given, restrict to these town directory names (e.g. ["Carla_Town01"]).
        None means all towns.
    min_episode_frames : int or None
        Skip episodes whose n_overlap < this value.  Defaults to
        window_T + action_horizon + 1 so every episode yields ≥ 1 window.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        window_T: int = 5,
        action_horizon: int = 8,
        towns: Optional[List[str]] = None,
        min_episode_frames: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.window_T = window_T
        self.action_horizon = action_horizon

        # Minimum overlap needed for at least one valid window:
        #   t_end_min = T - 1,  t_end_max = n_overlap - H - 1
        #   → n_overlap >= T + H
        _min_frames = window_T + action_horizon
        self.min_episode_frames = min_episode_frames if min_episode_frames is not None else _min_frames

        dataset_root = Path(dataset_root)
        if not dataset_root.exists():
            raise FileNotFoundError(f"dataset_root not found: {dataset_root}")

        self._episodes: List[_EpisodeMeta] = []
        # Flat sample index → (episode_idx, t_end)
        self._samples: List[Tuple[int, int]] = []

        self._scan(dataset_root, towns)

        logger.info(
            "AeroduoDataset: %d episodes → %d windows  "
            "(T=%d, H=%d, min_frames=%d)",
            len(self._episodes), len(self._samples),
            window_T, action_horizon, self.min_episode_frames,
        )

    # ── Scan ─────────────────────────────────────────────────────────────────

    def _scan(self, dataset_root: Path, towns: Optional[List[str]]) -> None:
        town_dirs = sorted(
            d for d in dataset_root.iterdir()
            if d.is_dir() and d.name.startswith("Carla_")
            and (towns is None or d.name in towns)
        )
        if not town_dirs:
            logger.warning("No Carla_* town directories found under %s", dataset_root)

        skipped = 0
        for town_dir in town_dirs:
            for ep_dir in sorted(d for d in town_dir.iterdir() if d.is_dir()):
                meta = self._load_episode(ep_dir)
                if meta is None:
                    skipped += 1
                    continue
                if meta.n_overlap < self.min_episode_frames:
                    skipped += 1
                    continue

                ep_idx = len(self._episodes)
                self._episodes.append(meta)

                # t_end ∈ [T-1, n_overlap - H - 1]  (inclusive)
                t_end_min = self.window_T - 1
                t_end_max = meta.n_overlap - self.action_horizon - 1
                for t_end in range(t_end_min, t_end_max + 1):
                    self._samples.append((ep_idx, t_end))

        if skipped:
            logger.info("Skipped %d episodes (missing data or too short).", skipped)

    @staticmethod
    def _load_episode(ep_dir: Path) -> Optional[_EpisodeMeta]:
        """Load one episode's metadata; returns None on any error."""
        try:
            # ── Instruction ───────────────────────────────────────────────────
            desc_path = ep_dir / "object_description_with_help.json"
            with desc_path.open("r", encoding="utf-8") as f:
                desc = json.load(f)
            instruction: str = desc[0] if isinstance(desc, list) else str(desc)
            if not instruction.strip():
                return None

            # ── High UAV trajectory ───────────────────────────────────────────
            high_traj_path = ep_dir / "high_uav_traj.json"
            with high_traj_path.open("r", encoding="utf-8") as f:
                high_data = json.load(f)
            high_poses = np.array(high_data["normalized_state"], dtype=np.float32)  # [N_high, 4]

            # ── Low UAV trajectory ────────────────────────────────────────────
            low_traj_path = ep_dir / "low_uav_traj.json"
            with low_traj_path.open("r", encoding="utf-8") as f:
                low_data = json.load(f)
            low_poses = np.array(low_data["normalized_state"], dtype=np.float32)   # [N_low, 4]

            # ── BEV frame paths (positionally indexed — don't parse filenames) ─
            bev_dir = ep_dir / "bevcamera"
            bev_frame_paths = sorted(
                p for p in bev_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            )

            # Sanity: trajectory length must match bevcamera frame count
            if len(bev_frame_paths) != len(high_poses):
                logger.warning(
                    "Frame/traj mismatch in %s: %d BEV files vs %d traj entries — skipping",
                    ep_dir, len(bev_frame_paths), len(high_poses),
                )
                return None

            n_overlap = min(len(high_poses), len(low_poses))

            return _EpisodeMeta(
                episode_path=ep_dir,
                instruction=instruction,
                high_poses=high_poses,
                low_poses=low_poses,
                bev_frame_paths=bev_frame_paths,
                n_overlap=n_overlap,
            )

        except Exception as exc:
            logger.warning("Failed to load episode %s: %s", ep_dir, exc)
            return None

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_idx, t_end = self._samples[idx]
        ep = self._episodes[ep_idx]

        T = self.window_T
        H = self.action_horizon
        window_start = t_end - T + 1

        # ── BEV images: T PIL images in temporal order ────────────────────────
        bev_images: List[Image.Image] = [
            Image.open(ep.bev_frame_paths[t]).convert("RGB")
            for t in range(window_start, t_end + 1)
        ]

        # ── High UAV poses for the BEV window: [T, 4] ─────────────────────────
        high_uav_poses = ep.high_poses[window_start : t_end + 1].copy()  # [T, 4]

        # ── Low UAV poses for the BEV window: [T, 4] ──────────────────────────
        # SmolVLM2Encoder.forward takes BOTH high_state and low_state as tokens
        # at every timestep, so each of the T position vertices is built from the
        # low-UAV pose that was concurrent with that BEV frame.
        low_uav_poses_window = ep.low_poses[window_start : t_end + 1].copy()  # [T, 4]

        # ── Low UAV: current pose + H future steps ─────────────────────────────
        # low_uav_pose_current == low_uav_poses_window[-1], kept as a named key
        # so the training loop can pass it directly to FlowMatchingNetwork without
        # slicing every time.
        # Target trajectory: H steps immediately AFTER t_end for flow-matching
        # supervision.
        low_uav_pose_current = low_uav_poses_window[-1].copy()                    # [4]
        low_uav_traj_target  = ep.low_poses[t_end + 1 : t_end + 1 + H].copy()   # [H, 4]

        return {
            "bev_images":            bev_images,          # List[PIL.Image], length T
            "high_uav_poses":        high_uav_poses,       # np.ndarray [T, 4]
            "low_uav_poses_window":  low_uav_poses_window, # np.ndarray [T, 4]  ← per-timestep low state for VLM
            "low_uav_pose_current":  low_uav_pose_current, # np.ndarray [4]     ← low_uav_poses_window[-1], for flow matching
            "low_uav_traj_target":   low_uav_traj_target,  # np.ndarray [H, 4]  ← flow-matching supervision target
            "instruction":           ep.instruction,        # str
            "episode_path":          str(ep.episode_path),  # str
            "window_start":          window_start,          # int
            "t_end":                 t_end,                 # int
        }


# ── collate_fn ────────────────────────────────────────────────────────────────

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate one or more AeroduoDataset samples into training-ready tensors.

    Supports any batch_size B >= 1.  num_workers must remain 0 (SAM2 /
    SmolVLM2 are not picklable across worker processes).

    Returns
    -------
    dict with keys:
        bev_images            : List[List[PIL.Image]]  shape [B][T]
        high_uav_poses        : Tensor [B, T, 4]
        low_uav_poses_window  : Tensor [B, T, 4]
        low_uav_pose_current  : Tensor [B, 4]
        low_uav_traj_target   : Tensor [B, H, 4]
        instruction           : List[str]  length B
        episode_path          : List[str]  length B
        window_start          : List[int]  length B
        t_end                 : List[int]  length B
    """
    return {
        # Images stay as nested Python lists — SmolVLM2Encoder handles PIL directly
        "bev_images": [s["bev_images"] for s in batch],          # [B][T]

        # Pose tensors stacked along a leading batch dim
        "high_uav_poses": torch.stack([
            torch.from_numpy(np.asarray(s["high_uav_poses"], dtype=np.float32))
            for s in batch
        ]),                                                        # [B, T, 4]
        "low_uav_poses_window": torch.stack([
            torch.from_numpy(np.asarray(s["low_uav_poses_window"], dtype=np.float32))
            for s in batch
        ]),                                                        # [B, T, 4]
        "low_uav_pose_current": torch.stack([
            torch.from_numpy(np.asarray(s["low_uav_pose_current"], dtype=np.float32))
            for s in batch
        ]),                                                        # [B, 4]
        "low_uav_traj_target": torch.stack([
            torch.from_numpy(np.asarray(s["low_uav_traj_target"], dtype=np.float32))
            for s in batch
        ]),                                                        # [B, H, 4]

        # Metadata — Python lists
        "instruction":  [s["instruction"]  for s in batch],
        "episode_path": [s["episode_path"] for s in batch],
        "window_start": [s["window_start"] for s in batch],
        "t_end":        [s["t_end"]        for s in batch],
    }
