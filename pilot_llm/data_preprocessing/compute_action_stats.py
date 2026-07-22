#!/usr/bin/env python3
"""compute_action_stats.py — Compute GLOBAL min/max normalization stats for
the low-UAV "action": the current-pose-relative xyz displacement used as the
flow-matching supervision target in both ``high_uav/dataset.py`` and
``low_uav/dataset2.py`` (see their ``__getitem__``).

Action definition
------------------
For an episode's low-UAV ``rel_state`` xyz (from ``low_uav_traj.json``,
already relative to the episode's own first frame — that offset cancels out
under differencing, so it's safe to use directly) and a horizon H, the action
anchored at frame t is:

    action_t = [rel_xyz[t+1] - rel_xyz[t], ..., rel_xyz[t+H] - rel_xyz[t]]   # [H, 3]

i.e. every one of the H future steps expressed relative to the CURRENT pose
at t, not relative to the episode start (that's ``rel_state`` itself, already
z-normalized elsewhere for the *pose* — this is a separate, min-max
normalized quantity for the *action*).

This is computed for every valid anchor t (0 <= t, t + H <= N_low - 1) in
every episode, for each of the requested horizons (2, 4, 8 by default), and
the resulting [dx, dy, dz] rows are pooled dataset-wide (not per-episode) to
get one global min/max per horizon — same "pool everything, one global stat"
convention as the pose z-normalization stats in generate_trajectories.py.

Unlike the pose mean/std pass, min/max is streaming-friendly (associative),
so this script runs a single pass and never materializes the full row pool.

Usage
-----
    python compute_action_stats.py --train-data data/train_data_new.json
    python compute_action_stats.py --train-data data/train_data_new.json --horizons 2 4 8 --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
AERODUO_DIR = SCRIPT_DIR.parents[1]
DEFAULT_TRAIN_DATA = AERODUO_DIR / "data" / "train_data_new.json"

DEFAULT_HORIZONS = (2, 4, 8)


def process_dataset(
    train_data_path: Path,
    horizons: List[int],
    limit: Optional[int],
    progress_every: int,
) -> Dict[str, object]:
    with train_data_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    ep_dirs = sorted({entry["traj_folder_path"] for entry in manifest})
    if limit is not None:
        ep_dirs = ep_dirs[:limit]

    running_min = {h: np.full(3, np.inf, dtype=np.float64) for h in horizons}
    running_max = {h: np.full(3, -np.inf, dtype=np.float64) for h in horizons}
    row_counts = {h: 0 for h in horizons}
    skipped = 0

    for i, ep in enumerate(ep_dirs):
        ep_dir = Path(ep)
        traj_path = ep_dir / "low_uav_traj.json"
        if not traj_path.is_file():
            skipped += 1
            continue
        try:
            with traj_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            rel_xyz = np.asarray(data["rel_state"], dtype=np.float64)[:, :3]  # [N_low, 3]
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            skipped += 1
            continue

        n_low = rel_xyz.shape[0]
        for h in horizons:
            n_anchors = n_low - h
            if n_anchors <= 0:
                continue
            # chunks[t, k, :] = rel_xyz[t + 1 + k] - rel_xyz[t]  for k in [0, h)
            anchors = rel_xyz[:n_anchors]                                    # [n_anchors, 3]
            futures = np.stack(
                [rel_xyz[k + 1 : k + 1 + n_anchors] for k in range(h)],
                axis=1,
            )                                                                 # [n_anchors, h, 3]
            chunks = futures - anchors[:, None, :]                            # [n_anchors, h, 3]
            flat = chunks.reshape(-1, 3)
            running_min[h] = np.minimum(running_min[h], flat.min(axis=0))
            running_max[h] = np.maximum(running_max[h], flat.max(axis=0))
            row_counts[h] += flat.shape[0]

        if progress_every and (i + 1) % progress_every == 0:
            print(f"...scanned {i + 1}/{len(ep_dirs)} episodes", file=sys.stderr, flush=True)

    return {
        "episodes_considered": len(ep_dirs),
        "episodes_skipped": skipped,
        "horizons": horizons,
        "action_min_max": {
            str(h): {
                "min": running_min[h].tolist(),
                "max": running_max[h].tolist(),
                "rows_pooled": row_counts[h],
            }
            for h in horizons
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA,
                         help=f"manifest listing episodes (default: {DEFAULT_TRAIN_DATA})")
    parser.add_argument("--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS),
                         help=f"action horizons to compute stats for (default: {DEFAULT_HORIZONS})")
    parser.add_argument("--limit", type=int, default=None,
                         help="only process the first N unique episodes (for quick iteration)")
    parser.add_argument("--progress-every", type=int, default=1000,
                         help="print progress every N episodes scanned; 0 disables")
    parser.add_argument("--stats-out", type=Path, default=None,
                         help="optionally write the summary as JSON")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = process_dataset(
        train_data_path=args.train_data,
        horizons=args.horizons,
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
