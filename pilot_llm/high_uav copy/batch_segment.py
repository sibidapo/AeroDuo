"""
batch_segment.py — Run the BEV segmentation pipeline over an entire dataset.

Usage:
    python batch_segment.py <dataset_dir> [--output-dir <dir>] [--device cuda]
                            [--max-episodes N] [--frames-per-episode N]
                            [--report-path report.json]

The dataset directory is expected to have the structure used by Hal-13k:
    <dataset_dir>/
        <town_name>/
            <episode_id>/
                bevcamera/          # BEV PNG frames
                object_description_with_help.json

One output directory is created per episode under <output_dir>/<town>/<episode>/.
A summary JSON report is written to <output_dir>/batch_report.json.
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Optional

# ─── Path setup ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_GSAM2_DIR = _REPO_ROOT / "Grounded-SAM-2"
if str(_GSAM2_DIR) not in sys.path:
    sys.path.insert(0, str(_GSAM2_DIR))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from bev_segmentation import (
    load_models,
    segment_bev_image,
    save_segmentation,
    visualise_segmentation,
    save_visualisation,
)
from noun_extractor import parse_instruction
import numpy as np


# ─── Dataset discovery ───────────────────────────────────────────────────────

def discover_episodes(dataset_dir: Path) -> list[dict]:
    """
    Walk <dataset_dir> and find all (episode_dir, instruction_json) pairs
    that contain at least one bevcamera frame.

    Returns a list of dicts:
        {"episode_dir": Path, "instruction": Path, "bev_frames": list[Path]}
    """
    episodes = []
    # Support both flat (dataset_dir/<episode>/) and nested (<town>/<episode>/) layouts
    for candidate in sorted(dataset_dir.rglob("object_description_with_help.json")):
        episode_dir = candidate.parent
        bev_dir = episode_dir / "bevcamera"
        if not bev_dir.is_dir():
            continue
        frames = sorted(bev_dir.glob("*.png")) + sorted(bev_dir.glob("*.jpg"))
        if frames:
            episodes.append({
                "episode_dir": episode_dir,
                "instruction":  candidate,
                "bev_frames":  frames,
            })
    return episodes


# ─── Per-episode processing ──────────────────────────────────────────────────

def process_episode(
    episode: dict,
    output_base: Path,
    sam2_pred,
    gdino_model,
    device: str,
    frames_per_episode: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run segmentation on all (or the first N) BEV frames in an episode.

    Returns an episode-level summary dict.
    """
    episode_dir = episode["episode_dir"]
    instruction  = episode["instruction"]
    frames       = episode["bev_frames"]

    if frames_per_episode is not None:
        frames = frames[:frames_per_episode]

    # Parse instruction once per episode
    try:
        goal, contextual, desc = parse_instruction(instruction)
    except Exception as e:
        return {
            "episode": str(episode_dir),
            "error":   f"Instruction parse failed: {e}",
            "frames_processed": 0,
        }

    # Output directory: mirrors the dataset structure under output_base
    try:
        rel = episode_dir.relative_to(episode_dir.parents[1])  # <town>/<episode>
    except ValueError:
        rel = Path(episode_dir.name)
    out_dir = output_base / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_results = []
    goal_detected_count = 0
    contextual_detected_total = 0
    failed_images = []

    for frame_path in frames:
        try:
            result = segment_bev_image(
                image_path=frame_path,
                goal_object=goal,
                contextual_nouns=contextual,
                sam2_predictor=sam2_pred,
                grounding_model=gdino_model,
                device=device,
            )
            result["instruction_path"] = str(instruction)
            result["description_text"] = desc
            result["contextual_categories"] = contextual

            # Build masks dict before saving strips _masks_array
            masks_raw = result.get("_masks_array", np.zeros((0,), dtype=bool))
            masks_dict = {
                d["mask_key"]: masks_raw[i]
                for i, d in enumerate(result["detections"])
            }

            save_segmentation(result, out_dir)

            vis = visualise_segmentation(result, masks_dict)
            save_visualisation(vis, out_dir, frame_path.stem)

            contextual_count = sum(1 for d in result["detections"] if not d["is_goal"])
            contextual_detected_total += contextual_count
            if result["goal_detected"]:
                goal_detected_count += 1
            if result["num_detections"] == 0:
                failed_images.append({
                    "frame": frame_path.name,
                    "reason": "no_detections",
                })

            frame_results.append({
                "frame":                   frame_path.name,
                "goal_detected":           result["goal_detected"],
                "num_detections":          result["num_detections"],
                "num_contextual_detections": contextual_count,
                "detections":  [{"category": d["category"], "confidence": d["confidence"],
                                  "is_goal": d["is_goal"]}
                                 for d in result["detections"]],
            })

            if verbose:
                status = "GOAL" if result["goal_detected"] else "    "
                n = result["num_detections"]
                print(f"  [{status}] {frame_path.name}  detections={n}")

        except Exception as e:
            failed_images.append({
                "frame": frame_path.name,
                "reason": "error",
                "error": str(e),
            })
            if verbose:
                print(f"  [ERROR] {frame_path.name}: {e}")

    n_frames = len(frames)
    return {
        "episode":              str(episode_dir),
        "goal_object":          goal,
        "contextual_nouns":     contextual,
        "frames_total":         n_frames,
        "frames_processed":     len(frame_results),
        "frames_failed":        len(failed_images),
        "goal_detected_count":  goal_detected_count,
        "goal_detection_rate":  goal_detected_count / max(len(frame_results), 1),
        "avg_detections_per_frame": (
            sum(r["num_detections"] for r in frame_results) / max(len(frame_results), 1)
        ),
        "avg_contextual_detections_per_frame": (
            contextual_detected_total / max(len(frame_results), 1)
        ),
        "frame_results":        frame_results,
        "failed_images":        failed_images,
        "output_dir":           str(out_dir),
    }


# ─── Batch runner ────────────────────────────────────────────────────────────

def run_batch(
    dataset_dir: Path,
    output_dir: Path,
    device: str = "auto",
    max_episodes: Optional[int] = None,
    frames_per_episode: Optional[int] = None,
    report_path: Optional[Path] = None,
    verbose: bool = True,
) -> dict:
    """
    Process all episodes and produce a summary report.

    Returns the full report dict and also writes it to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = discover_episodes(dataset_dir)
    if not episodes:
        print(f"No episodes found under {dataset_dir}", file=sys.stderr)
        return {}

    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    print(f"Found {len(episodes)} episode(s) in {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()

    # Load models once
    sam2_pred, gdino_model, dev = load_models(device)

    episode_summaries = []
    total_goal_detected = 0
    total_frames = 0
    total_detections = 0
    total_contextual_detections = 0
    complete_failures = []
    failed_images = []

    for ep_idx, episode in enumerate(episodes):
        ep_name = episode["episode_dir"].name
        n_frames = len(episode["bev_frames"])
        if frames_per_episode is not None:
            n_frames = min(n_frames, frames_per_episode)
        print(f"[{ep_idx+1:3d}/{len(episodes)}] {ep_name}  ({n_frames} frame(s))")

        summary = process_episode(
            episode=episode,
            output_base=output_dir,
            sam2_pred=sam2_pred,
            gdino_model=gdino_model,
            device=dev,
            frames_per_episode=frames_per_episode,
            verbose=verbose,
        )
        episode_summaries.append(summary)

        if "error" in summary:
            complete_failures.append(ep_name)
            print(f"  ERROR: {summary['error']}")
            continue

        total_goal_detected  += summary["goal_detected_count"]
        total_frames         += summary["frames_processed"]
        total_detections     += sum(
            r["num_detections"] for r in summary.get("frame_results", [])
        )
        total_contextual_detections += sum(
            r["num_contextual_detections"] for r in summary.get("frame_results", [])
        )
        failed_images.extend(
            {
                "episode": summary["episode"],
                **item,
            }
            for item in summary.get("failed_images", [])
        )
        if summary["frames_processed"] == 0:
            complete_failures.append(ep_name)

    # ── Dataset-level statistics ──────────────────────────────────────────────
    report = {
        "dataset_dir":          str(dataset_dir),
        "output_dir":           str(output_dir),
        "num_episodes":         len(episodes),
        "total_frames_processed": total_frames,
        "goal_detection_rate_overall": (
            total_goal_detected / max(total_frames, 1)
        ),
        "avg_detections_per_frame_overall": (
            total_detections / max(total_frames, 1)
        ),
        "avg_contextual_detections_per_frame_overall": (
            total_contextual_detections / max(total_frames, 1)
        ),
        "failed_images": failed_images,
        "complete_failure_episodes": complete_failures,
        "episode_summaries":    episode_summaries,
    }

    # Print high-level summary
    print()
    print("=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)
    print(f"Episodes processed        : {len(episodes)}")
    print(f"Frames processed          : {total_frames}")
    print(f"Overall goal detection    : {report['goal_detection_rate_overall']:.1%}")
    print(f"Avg detections / frame    : {report['avg_detections_per_frame_overall']:.2f}")
    print(f"Avg contextual / frame    : {report['avg_contextual_detections_per_frame_overall']:.2f}")
    print(f"Failed images             : {len(report['failed_images'])}")
    if complete_failures:
        print(f"Complete failures         : {len(complete_failures)}")
        for ep in complete_failures:
            print(f"  - {ep}")
    print("=" * 60)

    # Save report
    if report_path is None:
        report_path = output_dir / "batch_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")

    return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Batch BEV segmentation over a Hal-13k dataset.")
    p.add_argument("dataset_dir",
                   help="Root of the dataset (e.g., aeroduo/data/Hal-13k)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write segmentation outputs (default: <dataset_dir>/seg_outputs)")
    p.add_argument("--device", default="auto",
                   help="'cuda', 'cpu', or 'auto' (default: auto)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Process at most N episodes (for quick testing)")
    p.add_argument("--frames-per-episode", type=int, default=None,
                   help="Process at most N frames per episode")
    p.add_argument("--report-path", default=None,
                   help="Override path for the JSON report file")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-frame output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir  = Path(args.output_dir) if args.output_dir else dataset_dir / "seg_outputs"
    report_path = Path(args.report_path) if args.report_path else None

    run_batch(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        device=args.device,
        max_episodes=args.max_episodes,
        frames_per_episode=args.frames_per_episode,
        report_path=report_path,
        verbose=not args.quiet,
    )
