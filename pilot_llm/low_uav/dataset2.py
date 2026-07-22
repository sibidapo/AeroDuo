"""
dataset2.py — AeroduoDataset for Stage 2 training.

One training sample = a sliding window of T consecutive BEV frames (high-UAV
context) aligned with H subsequent low-UAV trajectory steps (flow-matching
supervision target), plus the low UAV's front-camera frame at t_end.

Episode selection
------------------
Episodes are the unique ``traj_folder_path`` values in ``train_data_new.json``
(same manifest ``data_preprocessing/generate_trajectories.py`` and
``compute_action_stats.py`` use) — NOT a directory scan. This matters because
the manifest's episodes are scattered across multiple filesystem roots (e.g.
``/storage/project/r-cj124-0/...``, ``/storage/project/r-lgan31-0/...``,
``/storage/scratch1/...``), so no single "dataset root" directory scan could
ever reach all of them, and town names aren't all ``Carla_*`` (e.g.
``TropicalIsland``, ``NYCEnvironmentMegapa``) so a ``Carla_*``-prefix filter
would silently drop entire towns.

Data layout (Hal-13k), per episode directory named by ``traj_folder_path``:
        bevcamera/                         <- BEV PNG frames (high UAV), sorted by name
        frontcamera/                       <- front-camera PNG frames (low UAV), sorted by name; len == N_low
        high_uav_traj.json                 <- {raw_state: [[x,y,z,heading], ...] absolute,
                                               rel_state: [[dx,dy,dz,heading], ...] — xyz
                                               relative to this UAV's own first frame in
                                               the episode; heading absolute (circular,
                                               left un-subtracted), wrapped to [-pi, pi]}
        low_uav_traj.json                  <- same schema, low UAV
        object_description_with_help.json  <- [instruction_string]

    Both trajectory files are produced by
    ``data_preprocessing/generate_trajectories.py``. They do NOT carry a
    pre-baked "normalized_state" — z-normalization of the xyz components is
    applied here, at load time, using ONE global mean/std pooled across
    every episode's rel_state (``LowUAVConfig.high_pose_mean/std`` and
    ``low_pose_mean/std``), not a per-episode statistic. This matches the
    normalization ``eval/dualuavpilot.py`` already applies at inference
    time from raw AirSim poses. Heading is never z-normalized — it's
    circular and is sin/cos-encoded inside the model instead.

    The flow-matching supervision target (``low_uav_traj_target``) is a
    DIFFERENT normalization: its xyz is the low UAV's displacement relative
    to its CURRENT pose (t_end), not the episode start, min-max normalized
    to [-1, 1] with GLOBAL per-horizon stats
    (``LowUAVConfig.action_min_max`` — see
    ``data_preprocessing/compute_action_stats.py``). Heading stays absolute
    and unchanged. ``eval/dualuavpilot.py`` inverts this exact
    transformation (min-max un-normalize, then add to the current raw pose)
    when decoding predicted actions back to world coordinates.

Sampling strategy:
    N_high = len(bevcamera frames) = len(high_uav_traj.rel_state)
    N_low  = len(frontcamera frames) = len(low_uav_traj.rel_state)
    n_overlap = min(N_high, N_low)  <- temporal range where both UAVs have data

    A sample anchored at t_end is valid when:
        t_end in [0,  n_overlap - H - 1]

    This gives H future low-UAV steps [t_end+1 .. t_end+H], and a T-frame
    BEV/pose history [t_end-T+1 .. t_end] — PADDED at episode start by
    repeating frame 0 for any index < 0, e.g. for t_end=0 the whole window
    is T copies of frame 0; for t_end=1 it's [0, 0, ..., 0, 1]; and so on
    until t_end >= T-1, after which it's a normal contiguous window. This
    mirrors the padding ``eval/dualuavpilot.py.push()`` applies to its
    observation deques on the very first pushed frame (repeats it T times),
    so training sees the same near-episode-start window shape eval produces
    instead of only ever training on full, unpadded T-frame histories.

    All (episode_idx, t_end) pairs are enumerated at __init__ time so the
    DataLoader can shuffle and report a meaningful __len__.

Important constraints:
    - num_workers must be 0 (SmolVLM2 / SAM2 / GroundingDINO are not picklable).
    - collate_fn supports any batch_size B >= 1.  Per-sample VLM sequences are
      ragged (instruction + fuzzy goal cue tokenize to different lengths), so
      LowUAVPolicy._encode_vlm_batch right-pads them and passes a validity mask
      down to the action head's cross-attention.
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

try:
    from .config.lowuavconfig import LowUAVConfig
except ImportError:
    from config.lowuavconfig import LowUAVConfig  # direct script execution

logger = logging.getLogger(__name__)

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# ── Episode metadata (loaded once at dataset construction) ────────────────────

@dataclass
class _EpisodeMeta:
    episode_path: Path
    instruction: str
    high_poses: np.ndarray        # [N_high, 4] — z-normalized xyz + absolute heading
    low_poses: np.ndarray         # [N_low,  4] — z-normalized xyz + absolute heading
    bev_frame_paths: List[Path]   # sorted; len == N_high; positional index == traj index
    front_frame_paths: List[Path] # sorted; len == N_low;  positional index == low traj index
    n_overlap: int                # min(N_high, N_low)
    goal_offsets: Optional[np.ndarray] = None  # [N_high, 2] — (Δnorth, Δeast) m,
                                               # target minus high-UAV position per frame
    low_goal_offsets: Optional[np.ndarray] = None  # [N_low, 2] — (Δnorth, Δeast) m,
                                                   # target minus low-UAV position per frame


# ── Dataset ───────────────────────────────────────────────────────────────────

class AeroduoDataset(Dataset):
    """
    PyTorch Dataset over the Hal-13k episode collection.

    Parameters
    ----------
    train_data_path : str or Path
        Manifest listing episodes (same file generate_trajectories.py /
        compute_action_stats.py consume). Episodes are the unique
        traj_folder_path values it contains — NOT a directory scan (see
        module docstring for why). train.py supplies this (its own
        --train_data CLI arg, defaulting to
        ``<aeroduo>/data/train_data_new.json``).
    window_T : int
        Number of consecutive BEV frames that form the high-UAV observation
        window (T=5 in Stage 1).
    action_horizon : int
        Number of future low-UAV trajectory steps used as the flow-matching
        supervision target (H=8 in Stage 1).  Must satisfy H >= 1.
    min_episode_frames : int or None
        Skip episodes whose n_overlap < this value.  Defaults to
        action_horizon + 1 so every episode yields ≥ 1 window (the T-frame
        history is padded at episode start, so it no longer gates this).
    cfg : LowUAVConfig or None
        Source of the pose z-normalization stats (high_pose_mean/std,
        low_pose_mean/std — see module docstring). None (default)
        constructs a plain LowUAVConfig(), the single source of truth
        also used by eval/dualuavpilot.py.
    """

    def __init__(
        self,
        train_data_path: str | Path,
        window_T: int = 5,
        action_horizon: int = 8,
        min_episode_frames: Optional[int] = None,
        cfg: Optional[LowUAVConfig] = None,
    ) -> None:
        super().__init__()
        self.window_T = window_T
        self.action_horizon = action_horizon

        # Minimum overlap needed for at least one valid window:
        #   t_end_min = 0 (T-frame history is padded, see _padded_window_indices),
        #   t_end_max = n_overlap - H - 1
        #   → n_overlap >= H + 1
        _min_frames = action_horizon + 1
        self.min_episode_frames = min_episode_frames if min_episode_frames is not None else _min_frames

        cfg = cfg if cfg is not None else LowUAVConfig()
        self._high_pose_mean = np.asarray(cfg.high_pose_mean, dtype=np.float32)
        self._high_pose_std = np.asarray(cfg.high_pose_std, dtype=np.float32)
        self._low_pose_mean = np.asarray(cfg.low_pose_mean, dtype=np.float32)
        self._low_pose_std = np.asarray(cfg.low_pose_std, dtype=np.float32)

        # GLOBAL min/max for the current-pose-relative low-UAV action, keyed
        # by horizon (see module docstring + _normalize_action). Must have an
        # entry for this instance's action_horizon.
        self._action_min_max: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
            int(h): (
                np.asarray(stats["min"], dtype=np.float32),
                np.asarray(stats["max"], dtype=np.float32),
            )
            for h, stats in cfg.action_min_max.items()
        }
        if action_horizon not in self._action_min_max:
            raise KeyError(
                f"cfg.action_min_max has no entry for action_horizon={action_horizon} "
                f"(available: {sorted(self._action_min_max)})"
            )

        train_data_path = Path(train_data_path)
        if not train_data_path.exists():
            raise FileNotFoundError(f"train_data_path not found: {train_data_path}")

        self._episodes: List[_EpisodeMeta] = []
        # Flat sample index → (episode_idx, t_end)
        self._samples: List[Tuple[int, int]] = []

        self._scan(train_data_path)

        logger.info(
            "AeroduoDataset: %d episodes → %d windows  "
            "(T=%d, H=%d, min_frames=%d)",
            len(self._episodes), len(self._samples),
            window_T, action_horizon, self.min_episode_frames,
        )

    # ── Scan ─────────────────────────────────────────────────────────────────

    def _scan(self, train_data_path: Path) -> None:
        with train_data_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        ep_dirs = sorted({entry["traj_folder_path"] for entry in manifest})

        if not ep_dirs:
            logger.warning("No episodes found in %s", train_data_path)

        skipped = 0
        for ep in ep_dirs:
            ep_dir = Path(ep)
            meta = self._load_episode(ep_dir)
            if meta is None:
                skipped += 1
                continue
            if meta.n_overlap < self.min_episode_frames:
                skipped += 1
                continue

            ep_idx = len(self._episodes)
            self._episodes.append(meta)

            # t_end ∈ [0, n_overlap - H - 1]  (inclusive); t_end < T-1
            # still yields a valid sample — its T-frame history is
            # padded (see _padded_window_indices).
            t_end_min = 0
            t_end_max = meta.n_overlap - self.action_horizon - 1
            for t_end in range(t_end_min, t_end_max + 1):
                self._samples.append((ep_idx, t_end))

        if skipped:
            logger.info("Skipped %d episodes (missing data or too short).", skipped)

    @staticmethod
    def _normalize_state(rel_state: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """z-normalize the xyz columns of a [.., 4] rel_state array against a
        GLOBAL (pooled, whole-dataset) mean/std; heading (column 3) passes
        through unchanged — it's circular and gets sin/cos-encoded inside
        the model instead of z-normalized."""
        out = rel_state.copy()
        out[..., :3] = (rel_state[..., :3] - mean) / std
        return out

    @staticmethod
    def _padded_window_indices(t_end: int, T: int) -> List[int]:
        """T frame indices ending at t_end, clamped to >= 0 — repeats frame 0
        to fill the window when t_end < T - 1 (episode start). Matches the
        padding eval/dualuavpilot.py.push() applies to its observation
        deques on the very first pushed frame (T copies of frame 0), so
        training sees the same window shape eval produces near episode
        start instead of only ever seeing full, unpadded histories."""
        return [max(0, t_end - T + 1 + k) for k in range(T)]

    @staticmethod
    def _normalize_action(action_xyz: np.ndarray, min_vec: np.ndarray, max_vec: np.ndarray) -> np.ndarray:
        """Min-max normalize a current-pose-relative action's xyz to [-1, 1]
        using GLOBAL (pooled, whole-dataset) per-horizon min/max
        (LowUAVConfig.action_min_max — see data_preprocessing/
        compute_action_stats.py). This is a distinct normalization from
        _normalize_state's z-norm: the action is relative to the CURRENT
        pose (t_end), not the episode start."""
        return 2.0 * (action_xyz - min_vec) / (max_vec - min_vec) - 1.0

    def _load_episode(self, ep_dir: Path) -> Optional[_EpisodeMeta]:
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
            # rel_state is unnormalized xyz relative to the episode start (+
            # absolute heading); z-normalization uses the GLOBAL pooled stats
            # from LowUAVConfig, not a per-episode statistic (see module
            # docstring).
            high_traj_path = ep_dir / "high_uav_traj.json"
            with high_traj_path.open("r", encoding="utf-8") as f:
                high_data = json.load(f)
            rel_state = np.array(high_data["rel_state"], dtype=np.float32)  # [N_high, 4]
            high_poses = self._normalize_state(rel_state, self._high_pose_mean, self._high_pose_std)

            # ── Per-frame goal offsets (Δnorth, Δeast) in metres ──────────────
            # rel_state is in the same frame as mark.json (+x = north, +y =
            # east), so goal − current = (target − start) − rel_state[t], using
            # mark["target"]["position"] (the object's ground-truth location)
            # as "target" — NOT mark["end"] (where the recorded demonstration
            # trajectory happened to stop, which sampling shows is ~2m from the
            # target on average and up to ~20m). eval.py has no notion of
            # "end" — its goal_position is the live target position
            # (object_position), the same quantity as mark["target"]["position"]
            # — so anchoring here to "end" instead would train the fuzzy
            # direction cue on a systematically different point than eval ever
            # provides.
            goal_offsets: Optional[np.ndarray] = None
            goal_xy: Optional[np.ndarray] = None
            mark_path = ep_dir / "mark.json"
            if mark_path.exists():
                with mark_path.open("r", encoding="utf-8") as f:
                    mark = json.load(f)
                goal_xy = (
                    np.array(mark["target"]["position"][:2], dtype=np.float32)
                    - np.array(mark["start"][:2], dtype=np.float32)
                )
                goal_offsets = goal_xy[None, :] - rel_state[:, :2]  # [N_high, 2]

            # ── Low UAV trajectory ────────────────────────────────────────────
            low_traj_path = ep_dir / "low_uav_traj.json"
            with low_traj_path.open("r", encoding="utf-8") as f:
                low_data = json.load(f)
            low_rel_state = np.array(low_data["rel_state"], dtype=np.float32)   # [N_low, 4]
            low_poses = self._normalize_state(low_rel_state, self._low_pose_mean, self._low_pose_std)

            # Same mark-based construction in the low-UAV frame: both UAVs share
            # the mark.json start, so goal − current = (target − start) − rel_state[t].
            low_goal_offsets: Optional[np.ndarray] = None
            if goal_xy is not None:
                low_goal_offsets = goal_xy[None, :] - low_rel_state[:, :2]  # [N_low, 2]

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

            # ── Front-camera frame paths (low UAV, positionally indexed) ─────
            front_dir = ep_dir / "frontcamera"
            front_frame_paths = sorted(
                p for p in front_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            )

            if len(front_frame_paths) != len(low_poses):
                logger.warning(
                    "Frame/traj mismatch in %s: %d front-cam files vs %d low traj entries — skipping",
                    ep_dir, len(front_frame_paths), len(low_poses),
                )
                return None

            n_overlap = min(len(high_poses), len(low_poses))

            return _EpisodeMeta(
                episode_path=ep_dir,
                instruction=instruction,
                high_poses=high_poses,
                low_poses=low_poses,
                bev_frame_paths=bev_frame_paths,
                front_frame_paths=front_frame_paths,
                n_overlap=n_overlap,
                goal_offsets=goal_offsets,
                low_goal_offsets=low_goal_offsets,
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

        # T frame indices ending at t_end; near episode start (t_end < T-1)
        # this repeats frame 0 to fill the window instead of a contiguous
        # slice — see _padded_window_indices.
        window_indices = self._padded_window_indices(t_end, T)

        # ── BEV images: T PIL images in temporal order ────────────────────────
        bev_images: List[Image.Image] = [
            Image.open(ep.bev_frame_paths[t]).convert("RGB")
            for t in window_indices
        ]

        # ── High UAV poses for the BEV window: [T, 4] ─────────────────────────
        high_uav_poses = ep.high_poses[window_indices].copy()  # [T, 4]

        # ── Low UAV poses for the BEV window: [T, 4] ──────────────────────────
        # PositionVertexBuilder cross-attends each timestep's VLM hidden states
        # with BOTH the high and low UAV poses, so each of the T position
        # vertices is built from the low-UAV pose concurrent with that BEV frame.
        low_uav_poses_window = ep.low_poses[window_indices].copy()  # [T, 4]

        # ── Low UAV: current pose + H future steps ─────────────────────────────
        # low_uav_pose_current == low_uav_poses_window[-1], kept as a named key
        # so the training loop can pass it directly to FlowMatchingNetwork without
        # slicing every time.
        low_uav_pose_current = low_uav_poses_window[-1].copy()                    # [4]

        # Action target: xyz is the CURRENT-pose-relative displacement (relative
        # to t_end, NOT the episode start), min-max normalized with GLOBAL
        # per-horizon stats (cfg.action_min_max) — a different normalization
        # from the z-normalized low_poses above. Recovered from low_poses by
        # undoing the /std (mean cancels exactly under differencing, so no
        # separate raw array is needed): diff(z-normed) * std == diff(raw).
        # Heading stays absolute/unchanged, sin/cos-encoded inside the model.
        raw_action_xyz = (
            (ep.low_poses[t_end + 1 : t_end + 1 + H, :3] - ep.low_poses[t_end, :3])
            * self._low_pose_std
        )                                                                          # [H, 3]
        action_min, action_max = self._action_min_max[H]
        norm_action_xyz = self._normalize_action(raw_action_xyz, action_min, action_max)  # [H, 3]
        heading_target = ep.low_poses[t_end + 1 : t_end + 1 + H, 3:4]              # [H, 1]
        low_uav_traj_target = np.concatenate(
            [norm_action_xyz, heading_target], axis=-1
        ).astype(np.float32)                                                       # [H, 4]

        # ── Low UAV front-camera frame at t_end ───────────────────────────────
        # Positionally aligned with low_uav_pose_current: the low UAV's forward-
        # facing view at the current timestep, used as visual conditioning in
        # Stage 2.
        low_uav_front_image = Image.open(ep.front_frame_paths[t_end]).convert("RGB")

        # ── Goal offsets for the BEV window: [T, 2] or None ────────────────────
        goal_offsets = (
            ep.goal_offsets[window_indices].copy()
            if ep.goal_offsets is not None else None
        )

        # ── Low-UAV goal offset at t_end: [2] or None ──────────────────────────
        # Aligned with low_uav_front_image / low_uav_pose_current; used for the
        # egocentric fuzzy direction cue in the low-UAV VLM prompt.
        low_goal_offset = (
            ep.low_goal_offsets[t_end].copy()
            if ep.low_goal_offsets is not None else None
        )

        return {
            "bev_images":            bev_images,           # List[PIL.Image], length T
            "low_uav_front_image":   low_uav_front_image,  # PIL.Image — front cam at t_end
            "high_uav_poses":        high_uav_poses,        # np.ndarray [T, 4]
            "low_uav_poses_window":  low_uav_poses_window,  # np.ndarray [T, 4]  ← per-timestep low state for VLM
            "low_uav_pose_current":  low_uav_pose_current,  # np.ndarray [4]     ← low_uav_poses_window[-1], for flow matching
            "low_uav_traj_target":   low_uav_traj_target,   # np.ndarray [H, 4]  ← flow-matching target: min-max normalized current-pose-relative xyz + absolute heading
            "goal_offsets":          goal_offsets,          # np.ndarray [T, 2] or None ← (Δnorth, Δeast) m per frame
            "low_goal_offset":       low_goal_offset,       # np.ndarray [2] or None    ← (Δnorth, Δeast) m at t_end, low-UAV frame
            "instruction":           ep.instruction,         # str
            "episode_path":          str(ep.episode_path),   # str
            "window_start":          window_start,           # int
            "t_end":                 t_end,                  # int
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
        low_uav_front_image   : List[PIL.Image]        length B — front cam at t_end per sample
        high_uav_poses        : Tensor [B, T, 4]
        low_uav_poses_window  : Tensor [B, T, 4]
        low_uav_pose_current  : Tensor [B, 4]
        low_uav_traj_target   : Tensor [B, H, 4]  — min-max normalized current-pose-relative xyz + absolute heading
        goal_offsets          : List[Tensor [T, 2] or None]  length B
        low_goal_offset       : List[Tensor [2] or None]     length B
        instruction           : List[str]  length B
        episode_path          : List[str]  length B
        window_start          : List[int]  length B
        t_end                 : List[int]  length B
    """
    return {
        # Images stay as Python lists — SmolVLM2Encoder handles PIL directly
        "bev_images":           [s["bev_images"]          for s in batch],  # [B][T]
        "low_uav_front_image":  [s["low_uav_front_image"] for s in batch],  # [B]

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

        # Per-episode goal offsets — kept as a list so an episode without
        # mark.json (None) degrades to the prompt's directional-prior fallback
        "goal_offsets": [
            torch.from_numpy(np.asarray(s["goal_offsets"], dtype=np.float32))
            if s["goal_offsets"] is not None else None
            for s in batch
        ],                                                         # [B] × ([T, 2] or None)
        "low_goal_offset": [
            torch.from_numpy(np.asarray(s["low_goal_offset"], dtype=np.float32))
            if s["low_goal_offset"] is not None else None
            for s in batch
        ],                                                         # [B] × ([2] or None)

        # Metadata — Python lists
        "instruction":  [s["instruction"]  for s in batch],
        "episode_path": [s["episode_path"] for s in batch],
        "window_start": [s["window_start"] for s in batch],
        "t_end":        [s["t_end"]        for s in batch],
    }
