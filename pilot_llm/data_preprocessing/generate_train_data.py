#!/usr/bin/env python3
"""Generate an AeroDuo training manifest from one or more HaL-13K roots.

The manifest format matches ``data/train_data.json``.  Test maps/trajectories
are derived from a test manifest and are always checked again before the output
file is committed.
"""

import argparse
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
AERODUO_DIR = SCRIPT_DIR.parents[1]
WORKSPACE_DIR = AERODUO_DIR.parent
DEFAULT_ROOTS = (
    Path("/storage/project/r-cj124-0/sibidapo3/Hal-13k"),
    Path("/storage/project/r-lgan31-0/sibidapo3/Hal-13k"),
    Path("/storage/home/hcoda1/3/sibidapo3/scratch/8750/Hal-13k"),
)
DEFAULT_TEST_MANIFEST = (
    WORKSPACE_DIR / "aeroduo_mock" / "data" / "test_unseen_new.json"
)
DEFAULT_OUTPUT = AERODUO_DIR / "data" / "train_data_new.json"
SUPPLEMENT_SUFFIX = "_supp"

TrajectoryKey = Tuple[str, str]


class ManifestError(RuntimeError):
    """Raised when the manifest cannot be generated safely."""


class TrajectoryError(RuntimeError):
    """Raised when trajectory-level metadata is unusable."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


class Audit:
    """Accumulate generation statistics and a few examples of each problem."""

    def __init__(self, example_limit: int = 5) -> None:
        self.counts: Counter = Counter()
        self.skip_reasons: Counter = Counter()
        self.map_trajectories: Counter = Counter()
        self.map_records: Counter = Counter()
        self.high_position_sources: Counter = Counter()
        self.examples: DefaultDict[str, List[str]] = defaultdict(list)
        self.example_limit = example_limit
        self.maps_seen: Set[str] = set()
        self.test_maps: Set[str] = set()
        self.test_keys: Set[TrajectoryKey] = set()
        self.test_keys_found: Set[TrajectoryKey] = set()
        self.included_keys: Set[TrajectoryKey] = set()
        self.included_base_keys: Set[TrajectoryKey] = set()
        self.final_test_overlap: Set[TrajectoryKey] = set()

    def skip(self, reason: str, path: Path, detail: str = "") -> None:
        self.skip_reasons[reason] += 1
        if len(self.examples[reason]) < self.example_limit:
            message = str(path)
            if detail:
                message = f"{message}: {detail}"
            self.examples[reason].append(message)

    def as_dict(
        self,
        roots: Sequence[Path],
        test_manifest: Path,
        output: Optional[Path],
        test_exclusion_scope: str,
        include_supp: bool,
        frame_interval: int,
        log_workers: int,
    ) -> Dict[str, Any]:
        return {
            "roots": [str(path) for path in roots],
            "test_manifest": str(test_manifest),
            "output": str(output) if output is not None else None,
            "test_exclusion_scope": test_exclusion_scope,
            "include_supp": include_supp,
            "frame_interval": frame_interval,
            "log_workers": log_workers,
            "maps_seen": sorted(self.maps_seen),
            "test_maps": sorted(self.test_maps),
            "test_manifest_trajectories": len(self.test_keys),
            "test_trajectories_found_in_roots": len(self.test_keys_found),
            "final_test_overlap": len(self.final_test_overlap),
            "counts": dict(sorted(self.counts.items())),
            "skip_reasons": dict(sorted(self.skip_reasons.items())),
            "high_position_sources": dict(
                sorted(self.high_position_sources.items())
            ),
            "map_trajectories": dict(sorted(self.map_trajectories.items())),
            "map_records": dict(sorted(self.map_records.items())),
            "examples": dict(sorted(self.examples.items())),
        }


def normalize_trajectory_id(trajectory_id: str) -> str:
    """Map a supplementary trajectory to the base trajectory for leakage checks."""
    if trajectory_id.endswith(SUPPLEMENT_SUFFIX):
        return trajectory_id[: -len(SUPPLEMENT_SUFFIX)]
    return trajectory_id


def canonical_key(raw_path: str) -> TrajectoryKey:
    """Extract ``(map_name, trajectory_id)`` without relying on a path prefix."""
    parts = [part for part in raw_path.replace("\\", "/").rstrip("/").split("/") if part]
    if len(parts) < 2:
        raise ManifestError(f"Cannot derive map/trajectory from path: {raw_path!r}")
    return parts[-2], normalize_trajectory_id(parts[-1])


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise TrajectoryError("invalid_json", str(error)) from error


def load_test_keys(path: Path) -> Tuple[Set[TrajectoryKey], Set[str]]:
    try:
        data = load_json(path)
    except FileNotFoundError as error:
        raise ManifestError(f"Test manifest does not exist: {path}") from error
    except TrajectoryError as error:
        raise ManifestError(f"Cannot read test manifest {path}: {error}") from error

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ManifestError(f"Test manifest must be a JSON list of paths: {path}")

    keys = {canonical_key(item) for item in data}
    if len(keys) != len(data):
        raise ManifestError(
            f"Test manifest contains duplicate canonical trajectories: {path}"
        )
    return keys, {map_name for map_name, _ in keys}


def is_xyz(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 3:
        return False
    for coordinate in value:
        if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
            return False
        if not math.isfinite(float(coordinate)):
            return False
    return True


def as_xyz(value: Any) -> List[Any]:
    """Return a detached JSON-compatible XYZ list while preserving numeric values."""
    if not is_xyz(value):
        raise ValueError(f"expected a finite numeric [x, y, z], got {value!r}")
    return list(value)


def require_json(path: Path, reason: str) -> Any:
    try:
        return load_json(path)
    except FileNotFoundError as error:
        raise TrajectoryError(reason, f"missing {path.name}") from error
    except TrajectoryError as error:
        raise TrajectoryError(reason, error.detail) from error


def log_position(log_path: Path) -> List[Any]:
    log_data = require_json(log_path, "invalid_log")
    try:
        position = log_data["sensors"]["state"]["position"]
    except (KeyError, TypeError) as error:
        raise TrajectoryError(
            "invalid_log_position", "missing sensors.state.position"
        ) from error
    try:
        return as_xyz(position)
    except ValueError as error:
        raise TrajectoryError("invalid_log_position", str(error)) from error


def load_drone2_trajectory(path: Path, audit: Audit) -> Optional[List[Any]]:
    if not path.exists():
        return None
    try:
        value = load_json(path)
    except TrajectoryError as error:
        audit.skip("invalid_drone2_traj_fallback_to_log", path, error.detail)
        return None
    if not isinstance(value, list):
        audit.skip(
            "invalid_drone2_traj_fallback_to_log", path, "expected a JSON list"
        )
        return None
    return value


def records_for_trajectory(
    trajectory_dir: Path,
    frame_interval: int,
    audit: Audit,
    log_executor: Optional[ThreadPoolExecutor] = None,
) -> List[Dict[str, Any]]:
    gt_waypoints = require_json(
        trajectory_dir / "gt_waypoints.json", "missing_or_invalid_gt_waypoints"
    )
    if (
        not isinstance(gt_waypoints, list)
        or not gt_waypoints
        or not all(is_xyz(waypoint) for waypoint in gt_waypoints)
    ):
        raise TrajectoryError(
            "invalid_gt_waypoints", "expected a non-empty list of XYZ positions"
        )

    mark = require_json(trajectory_dir / "mark.json", "missing_or_invalid_mark")
    try:
        target_position = as_xyz(mark["target"]["position"])
    except (KeyError, TypeError, ValueError) as error:
        raise TrajectoryError("invalid_target_position", str(error)) from error

    description = require_json(
        trajectory_dir / "object_description_with_help.json",
        "missing_or_invalid_description",
    )
    if (
        not isinstance(description, list)
        or not description
        or not isinstance(description[0], str)
        or not description[0].strip()
    ):
        raise TrajectoryError(
            "invalid_description", "expected a non-empty list whose first item is text"
        )

    bev_dir = trajectory_dir / "bevcamera"
    depth_dir = trajectory_dir / "bevcamera_depth"
    log_dir = trajectory_dir / "log"
    for directory, reason in (
        (bev_dir, "missing_bevcamera_directory"),
        (depth_dir, "missing_bevcamera_depth_directory"),
        (log_dir, "missing_log_directory"),
    ):
        if not directory.is_dir():
            raise TrajectoryError(reason, f"missing directory {directory.name}")

    drone2_trajectory = load_drone2_trajectory(
        trajectory_dir / "drone2_traj.json", audit
    )

    candidates: List[Tuple[int, Path, Path]] = []
    for frame in range(0, len(gt_waypoints), frame_interval):
        frame_name = f"{frame:06d}"
        bev_path = bev_dir / f"{frame_name}.png"
        depth_path = depth_dir / f"{frame_name}.png"
        log_path = log_dir / f"{frame_name}.json"
        if not bev_path.is_file():
            continue
        if not depth_path.is_file():
            audit.skip(
                "frame_missing_bevcamera_depth",
                trajectory_dir,
                f"frame {frame}",
            )
            continue
        if not log_path.is_file():
            audit.skip("frame_missing_log", trajectory_dir, f"frame {frame}")
            continue
        candidates.append((frame, bev_path, log_path))

    fallback_positions: Dict[int, List[Any]] = {}
    if drone2_trajectory is None and log_executor is not None:
        futures = {
            log_executor.submit(log_position, log_path): (frame, log_path)
            for frame, _, log_path in candidates
        }
        for future in as_completed(futures):
            frame, log_path = futures[future]
            try:
                fallback_positions[frame] = future.result()
            except TrajectoryError as error:
                audit.skip(error.reason, log_path, error.detail)

    records: List[Dict[str, Any]] = []
    for frame, bev_path, log_path in candidates:
        high_position = None
        high_source = ""
        if drone2_trajectory is not None and frame < len(drone2_trajectory):
            try:
                high_position = as_xyz(drone2_trajectory[frame])
                high_source = "drone2_traj"
            except ValueError:
                audit.skip(
                    "invalid_drone2_position_fallback_to_log",
                    trajectory_dir / "drone2_traj.json",
                    f"frame {frame}",
                )

        if high_position is None:
            if frame in fallback_positions:
                high_position = fallback_positions[frame]
                high_source = "log"
            elif drone2_trajectory is None and log_executor is not None:
                # The concurrent load failed and was already added to the audit.
                continue
            else:
                try:
                    high_position = log_position(log_path)
                    high_source = "log"
                except TrajectoryError as error:
                    audit.skip(error.reason, log_path, error.detail)
                    continue

        records.append(
            {
                "image_path": str(bev_path.resolve()),
                "traj_folder_path": str(trajectory_dir.resolve()),
                "int_time": frame,
                "high_uav_pos_now": high_position,
                "end_pos": target_position,
            }
        )
        audit.high_position_sources[high_source] += 1

    if not records:
        raise TrajectoryError(
            "no_usable_frames",
            "no interval-aligned BEV frame is valid within the gt_waypoints horizon",
        )
    return records


class AtomicJsonArrayWriter:
    """Stream a JSON array to a temporary file and atomically commit it."""

    def __init__(self, output: Path, overwrite: bool) -> None:
        self.output = output
        self.overwrite = overwrite
        self.temp_path: Optional[Path] = None
        self.handle = None
        self.first = True

    def __enter__(self) -> "AtomicJsonArrayWriter":
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists() and not self.overwrite:
            raise ManifestError(
                f"Output already exists: {self.output}. Pass --overwrite to replace it."
            )
        descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self.output.name}.", suffix=".tmp", dir=str(self.output.parent)
        )
        self.temp_path = Path(temp_name)
        self.handle = os.fdopen(descriptor, "w", encoding="utf-8")
        self.handle.write("[")
        return self

    def write(self, record: Dict[str, Any]) -> None:
        if self.handle is None:
            raise RuntimeError("writer is not open")
        serialized = json.dumps(record, ensure_ascii=False, indent=4)
        indented = "\n".join(f"    {line}" for line in serialized.splitlines())
        self.handle.write("\n" if self.first else ",\n")
        self.handle.write(indented)
        self.first = False

    def commit(self) -> None:
        if self.handle is None or self.temp_path is None:
            raise RuntimeError("writer is not open")
        self.handle.write("]\n" if self.first else "\n]\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.handle = None
        os.replace(self.temp_path, self.output)
        self.temp_path = None

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.temp_path is not None:
            try:
                self.temp_path.unlink()
            except FileNotFoundError:
                pass


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def iter_map_directories(roots: Sequence[Path]) -> Iterable[Tuple[Path, Path]]:
    for root in roots:
        if not root.is_dir():
            raise ManifestError(f"Dataset root does not exist or is not a directory: {root}")
        for map_dir in sorted(root.iterdir(), key=lambda path: path.name):
            if map_dir.is_dir() and map_dir.name != "maps":
                yield root, map_dir


def generate_manifest(
    roots: Sequence[Path],
    test_manifest: Path,
    output: Optional[Path],
    report_path: Optional[Path] = None,
    test_exclusion_scope: str = "map",
    include_supp: bool = False,
    frame_interval: int = 5,
    overwrite: bool = False,
    progress_every: int = 500,
    log_workers: int = 8,
) -> Dict[str, Any]:
    if frame_interval <= 0:
        raise ManifestError("--frame-interval must be positive")
    if test_exclusion_scope not in {"map", "trajectory"}:
        raise ManifestError("test exclusion scope must be 'map' or 'trajectory'")
    if log_workers <= 0:
        raise ManifestError("--log-workers must be positive")

    roots = [path.expanduser().resolve() for path in roots]
    test_manifest = test_manifest.expanduser().resolve()
    output = output.expanduser().resolve() if output is not None else None
    report_path = report_path.expanduser().resolve() if report_path else None
    if output == test_manifest:
        raise ManifestError("Output path must not replace the test manifest")
    if report_path == test_manifest:
        raise ManifestError("Report path must not replace the test manifest")
    if output is not None and report_path == output:
        raise ManifestError("Output and report paths must be different")

    test_keys, test_maps = load_test_keys(test_manifest)
    audit = Audit()
    audit.test_keys = test_keys
    audit.test_maps = test_maps

    seen_keys: Dict[TrajectoryKey, Path] = {}
    processed = 0
    writer_context = (
        AtomicJsonArrayWriter(output, overwrite) if output is not None else None
    )
    log_executor = (
        ThreadPoolExecutor(max_workers=log_workers) if log_workers > 1 else None
    )

    try:
        writer = writer_context.__enter__() if writer_context is not None else None
        for _, map_dir in iter_map_directories(roots):
            map_name = map_dir.name
            audit.maps_seen.add(map_name)
            audit.counts["map_directories_scanned"] += 1
            exclude_whole_map = (
                test_exclusion_scope == "map" and map_name in test_maps
            )

            for trajectory_dir in sorted(map_dir.iterdir(), key=lambda path: path.name):
                if not trajectory_dir.is_dir():
                    continue
                audit.counts["trajectory_directories_seen"] += 1
                trajectory_id = trajectory_dir.name
                base_key = (map_name, normalize_trajectory_id(trajectory_id))
                if base_key in test_keys:
                    audit.test_keys_found.add(base_key)

                if exclude_whole_map:
                    audit.skip("excluded_test_map", trajectory_dir)
                    continue
                if base_key in test_keys:
                    audit.skip("excluded_test_trajectory", trajectory_dir)
                    continue
                if trajectory_id.endswith(SUPPLEMENT_SUFFIX) and not include_supp:
                    audit.skip("excluded_supplement", trajectory_dir)
                    continue

                exact_key = (map_name, trajectory_id)
                if exact_key in seen_keys:
                    audit.skip(
                        "duplicate_trajectory_directory",
                        trajectory_dir,
                        f"already using {seen_keys[exact_key]}",
                    )
                    continue
                seen_keys[exact_key] = trajectory_dir
                processed += 1
                if progress_every and processed % progress_every == 0:
                    print(
                        f"Processed {processed} candidate trajectories; "
                        f"generated {audit.counts['records_generated']} records...",
                        file=sys.stderr,
                        flush=True,
                    )

                try:
                    records = records_for_trajectory(
                        trajectory_dir, frame_interval, audit, log_executor
                    )
                except TrajectoryError as error:
                    audit.skip(error.reason, trajectory_dir, error.detail)
                    continue

                audit.included_keys.add(exact_key)
                audit.included_base_keys.add(base_key)
                audit.map_trajectories[map_name] += 1
                audit.map_records[map_name] += len(records)
                audit.counts["trajectories_included"] += 1
                audit.counts["records_generated"] += len(records)
                if writer is not None:
                    for record in records:
                        writer.write(record)

        audit.final_test_overlap = audit.included_base_keys.intersection(test_keys)
        if audit.final_test_overlap:
            examples = sorted(audit.final_test_overlap)[:5]
            raise ManifestError(
                f"Safety check failed: output overlaps test set: {examples}"
            )
        if test_exclusion_scope == "map":
            included_test_maps = {
                map_name for map_name, _ in audit.included_keys if map_name in test_maps
            }
            if included_test_maps:
                raise ManifestError(
                    f"Safety check failed: output includes test maps: "
                    f"{sorted(included_test_maps)}"
                )

        audit.counts["candidate_trajectories_processed"] = processed
        audit.counts["test_trajectories_not_found_in_roots"] = len(
            test_keys.difference(audit.test_keys_found)
        )
        if writer_context is not None:
            writer_context.commit()
    except BaseException:
        if writer_context is not None:
            writer_context.__exit__(*sys.exc_info())
        raise
    else:
        if writer_context is not None:
            writer_context.__exit__(None, None, None)
    finally:
        if log_executor is not None:
            log_executor.shutdown()

    summary = audit.as_dict(
        roots=roots,
        test_manifest=test_manifest,
        output=output,
        test_exclusion_scope=test_exclusion_scope,
        include_supp=include_supp,
        frame_interval=frame_interval,
        log_workers=log_workers,
    )
    if report_path is not None:
        atomic_write_json(report_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an AeroDuo train_data JSON from split HaL-13K roots while "
            "enforcing train/test isolation."
        )
    )
    parser.add_argument(
        "--root",
        action="append",
        type=Path,
        help="HaL-13K root; repeat for multiple roots (defaults to the three known roots)",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=DEFAULT_TEST_MANIFEST,
        help=f"test trajectory list (default: {DEFAULT_TEST_MANIFEST})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output manifest (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="optionally write the audit summary as JSON",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="perform the complete scan and audit without writing the manifest",
    )
    parser.add_argument(
        "--test-exclusion-scope",
        choices=("map", "trajectory"),
        default="map",
        help=(
            "exclude every map named by the test manifest (default), or only its "
            "listed trajectories"
        ),
    )
    parser.add_argument(
        "--include-supp",
        action="store_true",
        help="include *_supp trajectories (excluded by default)",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=5,
        help="only include BEV frames divisible by this value (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow replacement of an existing output file",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="print progress after this many candidate trajectories; 0 disables it",
    )
    parser.add_argument(
        "--log-workers",
        type=int,
        default=8,
        help="parallel readers used when drone2_traj.json is absent (default: 8)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    roots = args.root if args.root else list(DEFAULT_ROOTS)
    output = None if args.dry_run else args.output
    try:
        summary = generate_manifest(
            roots=roots,
            test_manifest=args.test_manifest,
            output=output,
            report_path=args.report,
            test_exclusion_scope=args.test_exclusion_scope,
            include_supp=args.include_supp,
            frame_interval=args.frame_interval,
            overwrite=args.overwrite,
            progress_every=args.progress_every,
            log_workers=args.log_workers,
        )
    except (ManifestError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
