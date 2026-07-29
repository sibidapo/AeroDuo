"""Plot a 2D top-down view of a low-UAV eval rollout: start, end, target, and drone1 trajectory.

Usage:
    python plot_rollout_trajectory.py <rollout_dir> [--output PATH]

<rollout_dir> is an eval output folder such as
aeroduo/output_testrun/oracle_01abdaad-59e8-45ff-907d-f8148bdf7b5d
(the drone1 positions come from its log/*.json files, and the end point is
the drone's last logged waypoint; start/target come from the episode's
mark.json in aeroduo/data/).
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "aeroduo" / "data"

UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE
)


def find_episode_uuid(rollout_dir: Path) -> str:
    match = UUID_RE.search(rollout_dir.name)
    if not match:
        raise ValueError(f"Could not find an episode UUID in '{rollout_dir.name}'")
    return match.group(0)


def find_mark_json(uuid: str) -> Path:
    matches = list(DATA_ROOT.glob(f"*/*/{uuid}/mark.json"))
    if not matches:
        raise FileNotFoundError(f"No mark.json found for episode '{uuid}' under {DATA_ROOT}")
    return matches[0]


def load_drone1_trajectory(rollout_dir: Path) -> list[list[float]]:
    log_dir = rollout_dir / "log"
    files = sorted(log_dir.glob("*.json"), key=lambda p: int(p.stem))
    positions = []
    for f in files:
        with open(f) as fh:
            positions.append(json.load(fh))
    return positions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path, help="Path to a rollout output folder")
    parser.add_argument("--output", type=Path, default=None, help="Output image path (default: <rollout_dir>/trajectory_2d.png)")
    args = parser.parse_args()

    rollout_dir = args.rollout_dir.resolve()
    uuid = find_episode_uuid(rollout_dir)
    mark_path = find_mark_json(uuid)

    with open(mark_path) as f:
        mark = json.load(f)
    start = mark["start"]
    target = mark["target"]["position"]

    trajectory = load_drone1_trajectory(rollout_dir)
    if not trajectory:
        raise ValueError(f"No drone1 trajectory points found in {rollout_dir / 'log'}")

    end = trajectory[-1]

    traj_x = [p[0] for p in trajectory]
    traj_y = [p[1] for p in trajectory]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(traj_x, traj_y, "-o", color="tab:blue", markersize=3, linewidth=1.5, label="Drone1 (low UAV) trajectory")
    ax.scatter(*start[:2], color="tab:green", s=140, marker="^", zorder=5, label="Start")
    ax.scatter(*end[:2], color="tab:orange", s=140, marker="s", zorder=5, label="End")
    ax.scatter(*target[:2], color="tab:red", s=140, marker="*", zorder=5, label="Target")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Rollout trajectory: {uuid}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path = args.output or (rollout_dir / "trajectory_2d.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
