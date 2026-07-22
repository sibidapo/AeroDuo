#!/usr/bin/env python3
"""generate_trajectories.py — Build per-episode ``high_uav_traj.json`` /
``low_uav_traj.json`` files consumed by ``high_uav/dataset.py`` and
``low_uav/dataset2.py``.

Episode selection
------------------
Episodes are the unique ``traj_folder_path`` values in ``data/train_data_new.json``
(the output of ``generate_train_data.py`` + ``filter_discrepant_episodes.py``).

Source-of-truth verification (see ``data_preprocessing.md`` step 9)
---------------------------------------------------------------------
Two candidate sources were checked against the *actual* camera frame counts
for every episode in ``train_data_new.json`` (8901 episodes):

  High UAV (must align with ``bevcamera/``):
    - ``drone2_traj.json``: position-only [x, y, z] triples, one per raw sim
      frame. Present in only ~51% of episodes.
    - ``log/<frame>.json``: ``sensors.state.position`` is *byte-identical* to
      ``drone2_traj.json[frame]`` in every episode where both exist (verified
      over 400/400 sampled episodes, 0 mismatches), AND ``log/`` also carries
      ``sensors.state.orientation`` (quaternion), which ``drone2_traj.json``
      lacks entirely. ``log/`` is present for 100% of episodes.
    -> Use ``log/<frame>.json`` exclusively: same positions as drone2_traj.json
       when both exist, always available, and the only source with heading.
       ``bevcamera/000NNN.png`` filenames give the exact raw frame index to
       read from ``log/000NNN.json`` (frame_interval=5 in the manifest, with
       an irregular final step — never assume a fixed stride, parse the
       filename).

  Low UAV (must align with ``frontcamera/``):
    - ``gt_waypoints.json``: [x, y, z] only (no heading); length matches
      ``frontcamera/`` frame count in just 17/8901 episodes.
    - ``action.json["pos"]``: [x, y, z, heading] rows; length matches
      ``frontcamera/`` frame count in 8899/8901 episodes with an action.json
      (the other 2 lack action.json outright) and the *i*-th row lines up
      with the *i*-th (sorted) frontcamera frame by construction (list
      position, not raw frame number — frontcamera filenames run past the
      length of ``action.json["pos"]``, e.g. 34 pos rows vs frame numbers up
      to 000165, so pairing must be positional, not by parsed index).
    -> Use ``action.json["pos"]``, paired positionally with sorted
       ``frontcamera/`` files.

Heading
-------
``log/*.json`` orientation is ``[x, y, z, w]`` (AirSim ``Quaternionr.__iter__``
order — confirmed against ``airsim/types.py`` and matches the x,y,z,w
convention already used in ``eval/vlnce_src/env_uav.py:to_eularian_angles``,
*not* the w,x,y,z order in ``eval/dualuavpilot.py:_quat_to_yaw``, which is fed
manually-reordered live AirSim values in a different call site). Yaw is
recovered with the standard formula and is, by construction of ``atan2``,
already in (-pi, pi].

``action.json["pos"][:, 3]`` was empirically checked over 400 sampled
episodes (~20k rows) and its range is exactly [-pi, pi] as well — i.e. it is
already an absolute, wrapped world-yaw in the same convention.

So an explicit ``wrap_to_pi`` is *not* strictly needed for either source (the
data is already circular/wrapped) but is applied defensively — it's a no-op
on well-formed input and guards against any future source that isn't
pre-wrapped.

Output schema (per episode, per UAV)
-------------------------------------
    {
        "raw_state": [[x, y, z, heading], ...],   # absolute world coords
        "rel_state": [[dx, dy, dz, heading], ...] # xyz relative to this
                                                    # UAV's own first frame
                                                    # in the episode; heading
                                                    # left absolute (world
                                                    # yaw, wrapped) — heading
                                                    # is circular so it is
                                                    # NOT start-subtracted
    }

No "normalized_state" key is written. z-normalization (using ONE global
mean/std pooled over every episode's relative [dx, dy, dz], not a per-episode
statistic) is applied at load time by the Dataset classes
(``high_uav/dataset.py``, ``low_uav/dataset2.py``), using the stats this
script prints/reports and that were copied into
``high_uav/config.py`` (``AeroduoConfig.high_pose_mean/std``,
``low_pose_mean/std``) and ``low_uav/config/lowuavconfig.py``
(``LowUAVConfig.high_pose_mean/std``, ``low_pose_mean/std``). This is the
same convention ``eval/dualuavpilot.py`` already uses at inference time, so
generating the stats this way removes the previous train/eval normalization
mismatch (see project memory ``aeroduo-stage2-action-normalization``).

Usage
-----
    python generate_trajectories.py --train-data data/train_data_new.json
    python generate_trajectories.py --train-data data/train_data_new.json --dry-run --limit 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
AERODUO_DIR = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_DATA = AERODUO_DIR / "data" / "train_data_new.json"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


# ── Math helpers ────────────────────────────────────────────────────────────

def wrap_to_pi(angle: float) -> float:
    """Wrap an angle (radians) to (-pi, pi]. Defensive; atan2-derived and the
    observed action.json headings are already in this range."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_xyzw_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw from a quaternion given in [x, y, z, w] order (AirSim
    ``Quaternionr.__iter__`` / list(orientation) order, as stored by
    ``AirVLNSimulatorClientTool.py`` into log/*.json)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# ── Per-episode extraction ──────────────────────────────────────────────────

class EpisodeSkipped(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def _sorted_images(dir_path: Path) -> List[Path]:
    return sorted(
        p for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


def build_high_uav_state(ep_dir: Path) -> np.ndarray:
    """[N_high, 4] = [x, y, z, heading], one row per bevcamera frame, sourced
    from log/<frame>.json (position + orientation)."""
    bev_dir = ep_dir / "bevcamera"
    log_dir = ep_dir / "log"
    if not bev_dir.is_dir():
        raise EpisodeSkipped("missing_bevcamera_dir")
    if not log_dir.is_dir():
        raise EpisodeSkipped("missing_log_dir")

    bev_files = _sorted_images(bev_dir)
    if not bev_files:
        raise EpisodeSkipped("empty_bevcamera_dir")

    rows: List[List[float]] = []
    for bev_path in bev_files:
        try:
            frame_idx = int(bev_path.stem)
        except ValueError:
            raise EpisodeSkipped("unparseable_bev_filename", bev_path.name)

        log_path = log_dir / f"{frame_idx:06d}.json"
        if not log_path.is_file():
            raise EpisodeSkipped("bev_frame_missing_log", log_path.name)

        try:
            with log_path.open("r", encoding="utf-8") as f:
                log_data = json.load(f)
            state = log_data["sensors"]["state"]
            x, y, z = state["position"]
            qx, qy, qz, qw = state["orientation"]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise EpisodeSkipped("invalid_log_state", f"{log_path.name}: {exc}")

        heading = wrap_to_pi(quat_xyzw_to_yaw(qx, qy, qz, qw))
        rows.append([float(x), float(y), float(z), float(heading)])

    return np.asarray(rows, dtype=np.float64)


def build_low_uav_state(ep_dir: Path) -> np.ndarray:
    """[N_low, 4] = [x, y, z, heading], one row per frontcamera frame, paired
    positionally (not by parsed frame number) with action.json["pos"]."""
    front_dir = ep_dir / "frontcamera"
    action_path = ep_dir / "action.json"
    if not front_dir.is_dir():
        raise EpisodeSkipped("missing_frontcamera_dir")
    if not action_path.is_file():
        raise EpisodeSkipped("missing_action_json")

    front_files = _sorted_images(front_dir)
    if not front_files:
        raise EpisodeSkipped("empty_frontcamera_dir")

    try:
        with action_path.open("r", encoding="utf-8") as f:
            action_data = json.load(f)
        pos = action_data["pos"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise EpisodeSkipped("invalid_action_json", str(exc))

    if len(pos) != len(front_files):
        raise EpisodeSkipped(
            "action_pos_frontcamera_length_mismatch",
            f"{len(pos)} pos rows vs {len(front_files)} frontcamera files",
        )

    rows: List[List[float]] = []
    for row in pos:
        if len(row) != 4:
            raise EpisodeSkipped("invalid_action_pos_row", str(row))
        x, y, z, heading = row
        rows.append([float(x), float(y), float(z), float(wrap_to_pi(heading))])

    return np.asarray(rows, dtype=np.float64)


def to_rel_state(raw_state: np.ndarray) -> np.ndarray:
    """[N, 4] raw -> [N, 4] rel: xyz relative to this array's own first row;
    heading left absolute (circular — start-subtracting it would require its
    own wrap and gains nothing, since the model sin/cos-encodes heading)."""
    origin_xyz = raw_state[0, :3]
    rel = raw_state.copy()
    rel[:, :3] = raw_state[:, :3] - origin_xyz[None, :]
    return rel


# ── Driver ───────────────────────────────────────────────────────────────────

def process_dataset(
    train_data_path: Path,
    dry_run: bool,
    limit: Optional[int],
    progress_every: int,
) -> Dict[str, Any]:
    with train_data_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    ep_dirs = sorted({entry["traj_folder_path"] for entry in manifest})
    if limit is not None:
        ep_dirs = ep_dirs[:limit]

    skip_reasons: Dict[str, int] = {}
    written = 0

    # Pool of relative [dx, dy, dz] rows across the WHOLE dataset, kept
    # separately per UAV, for the single global normalization stat.
    high_rel_pool: List[np.ndarray] = []
    low_rel_pool: List[np.ndarray] = []

    # Per-episode results are buffered until the global stats are known,
    # then written in a second pass so the printed stats and the files on
    # disk are always in sync with each other in one run.
    episode_results: List[Tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for i, ep in enumerate(ep_dirs):
        ep_dir = Path(ep)
        try:
            high_raw = build_high_uav_state(ep_dir)
            low_raw = build_low_uav_state(ep_dir)
        except EpisodeSkipped as exc:
            skip_reasons[exc.reason] = skip_reasons.get(exc.reason, 0) + 1
            continue

        high_rel = to_rel_state(high_raw)
        low_rel = to_rel_state(low_raw)

        high_rel_pool.append(high_rel[:, :3])
        low_rel_pool.append(low_rel[:, :3])
        episode_results.append((ep_dir, high_raw, high_rel, low_raw, low_rel))

        if progress_every and (i + 1) % progress_every == 0:
            print(f"...scanned {i + 1}/{len(ep_dirs)} episodes "
                  f"({len(episode_results)} usable so far)", file=sys.stderr, flush=True)

    if not episode_results:
        raise RuntimeError("No usable episodes found — check skip_reasons")

    high_pool = np.concatenate(high_rel_pool, axis=0)  # [sum(N_high), 3]
    low_pool = np.concatenate(low_rel_pool, axis=0)    # [sum(N_low), 3]

    high_pose_mean = high_pool.mean(axis=0)
    high_pose_std = high_pool.std(axis=0)
    low_pose_mean = low_pool.mean(axis=0)
    low_pose_std = low_pool.std(axis=0)

    if not dry_run:
        for ep_dir, high_raw, high_rel, low_raw, low_rel in episode_results:
            high_out = {
                "raw_state": high_raw.tolist(),
                "rel_state": high_rel.tolist(),
            }
            low_out = {
                "raw_state": low_raw.tolist(),
                "rel_state": low_rel.tolist(),
            }
            with (ep_dir / "high_uav_traj.json").open("w", encoding="utf-8") as f:
                json.dump(high_out, f)
            with (ep_dir / "low_uav_traj.json").open("w", encoding="utf-8") as f:
                json.dump(low_out, f)
            written += 1

    return {
        "episodes_considered": len(ep_dirs),
        "episodes_written" if not dry_run else "episodes_would_write": (
            written if not dry_run else len(episode_results)
        ),
        "episodes_skipped": sum(skip_reasons.values()),
        "skip_reasons": skip_reasons,
        "high_uav_rows_pooled": int(high_pool.shape[0]),
        "low_uav_rows_pooled": int(low_pool.shape[0]),
        "high_pose_mean": high_pose_mean.tolist(),
        "high_pose_std": high_pose_std.tolist(),
        "low_pose_mean": low_pose_mean.tolist(),
        "low_pose_std": low_pose_std.tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA,
                         help=f"manifest listing episodes (default: {DEFAULT_TRAIN_DATA})")
    parser.add_argument("--dry-run", action="store_true",
                         help="compute stats and report skip counts without writing any files")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N unique episodes (for quick iteration)")
    parser.add_argument("--progress-every", type=int, default=500,
                         help="print progress every N episodes scanned; 0 disables")
    parser.add_argument("--stats-out", type=Path, default=None,
                         help="optionally write the summary (incl. mean/std) as JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = process_dataset(
        train_data_path=args.train_data,
        dry_run=args.dry_run,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))
    if args.stats_out is not None:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        with args.stats_out.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
