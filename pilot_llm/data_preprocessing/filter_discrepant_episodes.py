#!/usr/bin/env python3
"""Filter discrepant episodes from an AeroDuo training manifest.

For every unique ``traj_folder_path`` in the input manifest, the discrepancy is
defined as::

    frontcamera_image_count - bevcamera_image_count

An episode is flagged when that value exceeds ``--max-discrepancy``. All
manifest entries belonging to flagged episodes are removed from the training
manifest. Raw dataset directories and their contents are never deleted or
modified.

The default input is ``aeroduo/data/train_data_new.json`` and, unless a separate
``--output-manifest`` is supplied, that file is rewritten atomically in place.
Run with ``--dry-run`` to audit without changing the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
AERODUO_DIR = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_MANIFEST = AERODUO_DIR / "data" / "train_data_new.json"
DEFAULT_MAX_DISCREPANCY = 70
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


class FilterError(RuntimeError):
    """Raised when filtering cannot be completed safely."""


@dataclass(frozen=True)
class EpisodeInspection:
    trajectory_path: str
    bev_count: int
    front_count: int
    discrepancy: int
    front_only_stems: int
    bev_only_stems: int
    flag_reason: Optional[str] = None
    warning: Optional[str] = None

    @property
    def flagged(self) -> bool:
        return self.flag_reason is not None


def derived_report_path(input_manifest: Path) -> Path:
    return input_manifest.with_name(
        f"{input_manifest.stem}_discrepancy_report.json"
    )


def normalize_trajectory_path(raw_path: str, input_manifest: Path) -> str:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        # Existing AeroDuo manifests use paths such as data/HaL-13k/... and
        # expect them to be relative to the aeroduo project directory.
        project_dir = (
            input_manifest.parent.parent
            if input_manifest.parent.name == "data"
            else AERODUO_DIR
        )
        path = project_dir / path
    return os.path.abspath(os.path.normpath(str(path)))


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as error:
        raise FilterError(f"Input manifest does not exist: {path}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FilterError(f"Cannot read input manifest {path}: {error}") from error

    if not isinstance(data, list):
        raise FilterError("Input manifest must be a JSON array")
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise FilterError(f"Manifest entry {index} is not a JSON object")
        trajectory_path = item.get("traj_folder_path")
        if not isinstance(trajectory_path, str) or not trajectory_path.strip():
            raise FilterError(
                f"Manifest entry {index} has no valid traj_folder_path"
            )
    return data


def image_stems(camera_dir: Path) -> Optional[Tuple[int, Set[str]]]:
    if not camera_dir.is_dir():
        return None
    count = 0
    stems: Set[str] = set()
    try:
        for entry in os.scandir(camera_dir):
            if entry.is_file() and Path(entry.name).suffix.lower() in IMAGE_SUFFIXES:
                count += 1
                stems.add(Path(entry.name).stem)
    except OSError as error:
        raise FilterError(f"Cannot inspect camera directory {camera_dir}: {error}") from error
    return count, stems


def inspect_episode(
    trajectory_path: str, max_discrepancy: int
) -> EpisodeInspection:
    episode_dir = Path(trajectory_path)
    if not episode_dir.is_dir():
        return EpisodeInspection(
            trajectory_path=trajectory_path,
            bev_count=0,
            front_count=0,
            discrepancy=0,
            front_only_stems=0,
            bev_only_stems=0,
            flag_reason="missing_trajectory_directory",
        )

    bev_result = image_stems(episode_dir / "bevcamera")
    front_result = image_stems(episode_dir / "frontcamera")
    if bev_result is None:
        return EpisodeInspection(
            trajectory_path=trajectory_path,
            bev_count=0,
            front_count=front_result[0] if front_result else 0,
            discrepancy=front_result[0] if front_result else 0,
            front_only_stems=len(front_result[1]) if front_result else 0,
            bev_only_stems=0,
            flag_reason="missing_bevcamera_directory",
            warning=("missing_frontcamera_directory" if front_result is None else None),
        )

    bev_count, bev_stems = bev_result
    if front_result is None:
        return EpisodeInspection(
            trajectory_path=trajectory_path,
            bev_count=bev_count,
            front_count=0,
            discrepancy=-bev_count,
            front_only_stems=0,
            bev_only_stems=len(bev_stems),
            warning="missing_frontcamera_directory",
        )

    front_count, front_stems = front_result
    discrepancy = front_count - bev_count
    reason = (
        "discrepancy_exceeds_threshold"
        if discrepancy > max_discrepancy
        else None
    )
    return EpisodeInspection(
        trajectory_path=trajectory_path,
        bev_count=bev_count,
        front_count=front_count,
        discrepancy=discrepancy,
        front_only_stems=len(front_stems.difference(bev_stems)),
        bev_only_stems=len(bev_stems.difference(front_stems)),
        flag_reason=reason,
    )


def inspect_all_episodes(
    trajectory_paths: Sequence[str],
    max_discrepancy: int,
    workers: int,
    progress_every: int,
) -> List[EpisodeInspection]:
    inspections: Dict[str, EpisodeInspection] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(inspect_episode, path, max_discrepancy): path
            for path in trajectory_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                inspections[path] = future.result()
            except FilterError:
                raise
            except Exception as error:
                raise FilterError(f"Failed to inspect {path}: {error}") from error
            if progress_every and completed % progress_every == 0:
                print(
                    f"Inspected {completed}/{len(trajectory_paths)} trajectories...",
                    file=sys.stderr,
                    flush=True,
                )
    return [inspections[path] for path in trajectory_paths]


def atomic_write_json(path: Path, value: Any, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FilterError(
            f"Output already exists: {path}. Pass --overwrite to replace it."
        )
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def filter_manifest(
    input_manifest: Path,
    output_manifest: Optional[Path],
    report_path: Optional[Path],
    max_discrepancy: int = DEFAULT_MAX_DISCREPANCY,
    workers: int = 8,
    progress_every: int = 500,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if max_discrepancy < 0:
        raise FilterError("--max-discrepancy must be non-negative")
    if workers <= 0:
        raise FilterError("--workers must be positive")

    input_manifest = input_manifest.expanduser().resolve()
    output_manifest = (
        output_manifest.expanduser().resolve() if output_manifest else None
    )
    report_path = report_path.expanduser().resolve() if report_path else None
    if report_path == input_manifest:
        raise FilterError("Report must not replace the input manifest")
    if output_manifest is not None and report_path == output_manifest:
        raise FilterError("Output manifest and report paths must be different")

    # Check separate outputs before doing any work or replacing the input. This
    # prevents an existing report from causing a late failure after an in-place
    # manifest update has already succeeded.
    for candidate in (output_manifest, report_path):
        if (
            candidate is not None
            and candidate != input_manifest
            and candidate.exists()
            and not overwrite
        ):
            raise FilterError(
                f"Output already exists: {candidate}. Pass --overwrite to replace it."
            )

    data = load_manifest(input_manifest)
    trajectory_paths = sorted(
        {
            normalize_trajectory_path(item["traj_folder_path"], input_manifest)
            for item in data
        }
    )
    inspections = inspect_all_episodes(
        trajectory_paths,
        max_discrepancy=max_discrepancy,
        workers=workers,
        progress_every=progress_every,
    )
    flagged_paths = {
        inspection.trajectory_path
        for inspection in inspections
        if inspection.flagged
    }

    filtered_data = [
        item
        for item in data
        if normalize_trajectory_path(item["traj_folder_path"], input_manifest)
        not in flagged_paths
    ]
    reasons = Counter(
        inspection.flag_reason for inspection in inspections if inspection.flag_reason
    )
    warnings = Counter(
        inspection.warning for inspection in inspections if inspection.warning
    )
    flagged = sorted(
        (inspection for inspection in inspections if inspection.flagged),
        key=lambda item: (item.discrepancy, item.trajectory_path),
        reverse=True,
    )

    summary: Dict[str, Any] = {
        "input_manifest": str(input_manifest),
        "output_manifest": str(output_manifest) if output_manifest else None,
        "in_place": output_manifest == input_manifest,
        "max_discrepancy": max_discrepancy,
        "input_records": len(data),
        "output_records": len(filtered_data),
        "removed_records": len(data) - len(filtered_data),
        "input_trajectories": len(trajectory_paths),
        "output_trajectories": len(trajectory_paths) - len(flagged_paths),
        "flagged_trajectories": len(flagged_paths),
        "flag_reasons": dict(sorted(reasons.items())),
        "warnings": dict(sorted(warnings.items())),
        "flagged": [asdict(inspection) for inspection in flagged],
    }

    if output_manifest is not None:
        # Replacing the input manifest is the normal mode. atomic_write_json()
        # writes a temporary file in the same directory and then uses
        # os.replace(), so the original remains intact if serialization fails.
        atomic_write_json(
            output_manifest,
            filtered_data,
            overwrite=(overwrite or output_manifest == input_manifest),
        )
    if report_path is not None:
        atomic_write_json(report_path, summary, overwrite=overwrite)
    return summary


def print_summary(summary: Dict[str, Any], show_limit: int) -> None:
    if summary["output_manifest"] is None:
        mode = "DRY RUN"
    elif summary["in_place"]:
        mode = "IN-PLACE MANIFEST UPDATE"
    else:
        mode = "WRITE SEPARATE MANIFEST"
    print(f"Input manifest:       {summary['input_manifest']}")
    print(f"Output manifest:      {summary['output_manifest'] or '(not written)'}")
    print(f"Mode:                 {mode}")
    print(f"Max discrepancy:      {summary['max_discrepancy']}")
    print(f"Input trajectories:   {summary['input_trajectories']}")
    print(f"Flagged trajectories: {summary['flagged_trajectories']}")
    print(f"Output trajectories:  {summary['output_trajectories']}")
    print(f"Input records:        {summary['input_records']}")
    print(f"Removed records:      {summary['removed_records']}")
    print(f"Output records:       {summary['output_records']}")
    if summary["warnings"]:
        print(f"Warnings:             {summary['warnings']}")

    flagged = summary["flagged"]
    if not flagged or show_limit == 0:
        return
    print()
    print(
        f"{'Difference':>10}  {'BEV':>5}  {'Front':>5}  "
        f"{'Front-only':>10}  Reason / trajectory"
    )
    print("-" * 110)
    for item in flagged[:show_limit]:
        print(
            f"{item['discrepancy']:>10}  {item['bev_count']:>5}  "
            f"{item['front_count']:>5}  {item['front_only_stems']:>10}  "
            f"{item['flag_reason']}: {item['trajectory_path']}"
        )
    if len(flagged) > show_limit:
        print(f"... {len(flagged) - show_limit} additional flagged trajectories in report")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Remove entries for camera-count-discrepant trajectories from an "
            "AeroDuo training manifest. By default the manifest is updated in "
            "place; raw episode folders are never deleted or modified."
        )
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=DEFAULT_INPUT_MANIFEST,
        help=f"input manifest (default: {DEFAULT_INPUT_MANIFEST})",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        help=(
            "write a separate filtered manifest instead of updating the input "
            "manifest in place"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="audit report (default: <input stem>_discrepancy_report.json)",
    )
    parser.add_argument(
        "--max-discrepancy",
        type=int,
        default=DEFAULT_MAX_DISCREPANCY,
        help="maximum allowed frontcamera - bevcamera count (default: %(default)s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="audit only; do not write the filtered manifest or default report",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing separate output manifest or report",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parallel directory scanners (default: %(default)s)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="progress interval in trajectories; 0 disables it (default: %(default)s)",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=100,
        help="maximum flagged trajectories printed; 0 hides details (default: %(default)s)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    input_manifest = args.input_manifest.expanduser().resolve()
    if args.dry_run:
        output_manifest = None
        report_path = args.report
    else:
        output_manifest = args.output_manifest or input_manifest
        report_path = args.report or derived_report_path(input_manifest)

    try:
        summary = filter_manifest(
            input_manifest=input_manifest,
            output_manifest=output_manifest,
            report_path=report_path,
            max_discrepancy=args.max_discrepancy,
            workers=args.workers,
            progress_every=args.progress_every,
            overwrite=args.overwrite,
        )
    except (FilterError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print_summary(summary, show_limit=max(args.show_limit, 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
