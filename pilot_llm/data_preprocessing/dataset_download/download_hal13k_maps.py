#!/usr/bin/env python3
"""Download complete, per-scene HaL-13k map archives from ModelScope.

The maps are split ZIP archives.  A scene consists of zero or more numbered
parts (``Scene.z01``, ``Scene.z02``, ...) and one terminal ``Scene.zip`` file.
This script reads the live ModelScope manifest, validates each split set, and
finishes every file for one scene before proceeding to the next scene.

Examples:

    ./download_hal13k_maps.py --list
    ./download_hal13k_maps.py Carla_Town01
    ./download_hal13k_maps.py Carla_Town01 Carla_Town02
    ./download_hal13k_maps.py --all

Interrupted transfers are left as ``*.part`` files and resumed by the next
run.  Completed files are checked against the size and SHA-256 digest published
in the ModelScope manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ID = "Reynard/HaL-13k"
REVISION = "master"
REMOTE_DIR = "maps"
DEFAULT_ENDPOINT = "https://www.modelscope.cn"
DEFAULT_ARCHIVE_DIR = Path("aeroduo/data/Hal-13k/maps")
ARCHIVE_RE = re.compile(r"^(?P<scene>.+)\.(?P<suffix>zip|z(?P<part>\d+))$", re.IGNORECASE)


@dataclass(frozen=True)
class RemoteFile:
    path: str
    name: str
    size: int
    sha256: str
    part_number: int | None

    @property
    def is_terminal_zip(self) -> bool:
        return self.part_number is None


@dataclass(frozen=True)
class SceneArchive:
    name: str
    files: tuple[RemoteFile, ...]

    @property
    def total_size(self) -> int:
        return sum(item.size for item in self.files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download complete HaL-13k split ZIP sets, finishing one scene "
            "before starting the next."
        )
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        metavar="SCENE",
        help="Scene names to download, in the order they should be processed.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Download every scene.")
    mode.add_argument("--list", action="store_true", help="List available scenes and exit.")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Destination for archive parts (default: {DEFAULT_ARCHIVE_DIR}).",
    )
    parser.add_argument("--repo-id", default=REPO_ID, help=argparse.SUPPRESS)
    parser.add_argument("--revision", default=REVISION, help="Repository revision (default: master).")
    parser.add_argument(
        "--endpoint",
        default=DEFAULT_ENDPOINT,
        help=f"ModelScope endpoint (default: {DEFAULT_ENDPOINT}).",
    )
    parser.add_argument(
        "--curl-retries",
        type=int,
        default=8,
        help="Retries performed by curl for each archive part (default: 8).",
    )
    parser.add_argument(
        "--no-checksum",
        action="store_true",
        help="Check only file sizes instead of also verifying SHA-256 digests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the files that would be downloaded without downloading them.",
    )
    args = parser.parse_args()

    if args.curl_retries < 0:
        parser.error("--curl-retries must be non-negative")
    if args.all and args.scenes:
        parser.error("pass either explicit SCENE names or --all, not both")
    if args.list and args.scenes:
        parser.error("--list does not accept SCENE names")
    if not args.list and not args.all and not args.scenes:
        parser.error("provide at least one SCENE, or use --all/--list")
    return args


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def request_json(
    endpoint: str,
    path: str,
    params: dict[str, Any] | None = None,
    attempts: int = 4,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params or {})
    url = f"{endpoint.rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"

    headers = {"User-Agent": "hal13k-scene-downloader/1.0"}
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            if payload.get("Code") != 200:
                raise RuntimeError(
                    f"ModelScope API error {payload.get('Code')}: {payload.get('Message')}"
                )
            return payload
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt < attempts:
                delay = min(2 ** (attempt - 1), 8)
                print(f"[retry] Manifest request failed ({exc}); retrying in {delay}s...", file=sys.stderr)
                time.sleep(delay)

    raise RuntimeError(f"could not query ModelScope: {last_error}")


def fetch_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    owner, separator, dataset_name = args.repo_id.partition("/")
    if not separator or not owner or not dataset_name or "/" in dataset_name:
        raise RuntimeError(f"invalid ModelScope dataset ID: {args.repo_id!r}")

    detail = request_json(
        args.endpoint,
        f"/api/v1/datasets/{urllib.parse.quote(owner)}/{urllib.parse.quote(dataset_name)}",
    )
    dataset_id = detail.get("Data", {}).get("Id")
    if dataset_id is None:
        raise RuntimeError("ModelScope dataset response did not contain a dataset ID")

    files: list[dict[str, Any]] = []
    page_number = 1
    page_size = 500
    while True:
        payload = request_json(
            args.endpoint,
            f"/api/v1/datasets/{dataset_id}/repo/tree",
            {
                "Revision": args.revision,
                "Root": REMOTE_DIR,
                "Recursive": "True",
                "PageNumber": page_number,
                "PageSize": page_size,
            },
        )
        data = payload.get("Data") or {}
        page = data.get("Files") or []
        files.extend(page)
        total_count = data.get("TotalCount", payload.get("TotalCount"))
        if not page or (total_count is not None and len(files) >= int(total_count)):
            break
        if len(page) < page_size:
            break
        page_number += 1
    return files


def build_scene_archives(manifest: list[dict[str, Any]]) -> dict[str, SceneArchive]:
    grouped: dict[str, list[RemoteFile]] = {}
    for entry in manifest:
        if str(entry.get("Type", "")).lower() != "blob":
            continue
        remote_path = str(entry.get("Path", ""))
        pure_path = PurePosixPath(remote_path)
        if pure_path.parent.as_posix() != REMOTE_DIR:
            continue
        match = ARCHIVE_RE.fullmatch(pure_path.name)
        if not match:
            continue

        scene = match.group("scene")
        part_text = match.group("part")
        item = RemoteFile(
            path=remote_path,
            name=pure_path.name,
            size=int(entry.get("Size", 0)),
            sha256=str(entry.get("Sha256", "")).lower(),
            part_number=int(part_text) if part_text is not None else None,
        )
        grouped.setdefault(scene, []).append(item)

    archives: dict[str, SceneArchive] = {}
    for scene, items in grouped.items():
        terminals = [item for item in items if item.is_terminal_zip]
        if len(terminals) != 1:
            raise RuntimeError(
                f"remote scene {scene!r} has {len(terminals)} terminal .zip files; expected one"
            )

        part_numbers = sorted(
            item.part_number for item in items if item.part_number is not None
        )
        if len(part_numbers) != len(set(part_numbers)):
            raise RuntimeError(f"remote scene {scene!r} contains duplicate numbered parts")
        if part_numbers and part_numbers != list(range(1, part_numbers[-1] + 1)):
            raise RuntimeError(
                f"remote scene {scene!r} has a non-contiguous split set: {part_numbers}"
            )
        if any(item.size <= 0 for item in items):
            raise RuntimeError(f"remote scene {scene!r} contains a file with an invalid size")

        ordered = tuple(
            sorted(
                items,
                key=lambda item: (
                    item.is_terminal_zip,
                    item.part_number if item.part_number is not None else 0,
                ),
            )
        )
        archives[scene] = SceneArchive(name=scene, files=ordered)

    if not archives:
        raise RuntimeError(f"no split ZIP archives were found under {REMOTE_DIR!r}")
    return archives


def select_scenes(
    archives: dict[str, SceneArchive], requested: list[str], select_all: bool
) -> list[SceneArchive]:
    if select_all:
        return [archives[name] for name in sorted(archives, key=str.casefold)]

    by_casefold = {name.casefold(): name for name in archives}
    selected: list[SceneArchive] = []
    seen: set[str] = set()
    for requested_name in requested:
        canonical = by_casefold.get(requested_name.casefold())
        if canonical is None:
            available = ", ".join(sorted(archives, key=str.casefold))
            raise RuntimeError(
                f"unknown scene {requested_name!r}. Available scenes: {available}"
            )
        if canonical not in seen:
            selected.append(archives[canonical])
            seen.add(canonical)
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_is_complete(path: Path, remote: RemoteFile, verify_checksum: bool) -> bool:
    if not path.is_file() or path.stat().st_size != remote.size:
        return False
    if verify_checksum and remote.sha256:
        print(f"[verify] {path.name}")
        return sha256_file(path) == remote.sha256
    return True


def download_url(args: argparse.Namespace, remote_path: str) -> str:
    owner_and_name = "/".join(
        urllib.parse.quote(component, safe="") for component in args.repo_id.split("/")
    )
    query = urllib.parse.urlencode(
        {"Revision": args.revision, "FilePath": remote_path}
    )
    return f"{args.endpoint.rstrip('/')}/api/v1/datasets/{owner_and_name}/repo?{query}"


def prepare_partial(target: Path, partial: Path, remote: RemoteFile) -> None:
    if target.exists():
        target_size = target.stat().st_size
        if target_size < remote.size and not partial.exists():
            target.replace(partial)
            print(f"[resume] Moved incomplete {target.name} to {partial.name}")
        else:
            target.unlink()
            print(f"[replace] Removed incomplete or invalid {target.name}")

    if partial.exists() and partial.stat().st_size > remote.size:
        partial.unlink()
        print(f"[replace] Removed oversized partial file {partial.name}")


def download_file(
    args: argparse.Namespace,
    curl: str,
    archive_dir: Path,
    remote: RemoteFile,
) -> str:
    target = archive_dir / remote.name
    partial = archive_dir / f"{remote.name}.part"
    verify_checksum = not args.no_checksum

    if file_is_complete(target, remote, verify_checksum):
        print(f"[skip] {remote.name} is complete")
        return "skipped"
    prepare_partial(target, partial, remote)

    if file_is_complete(partial, remote, verify_checksum):
        partial.replace(target)
        print(f"[done] {remote.name} (recovered completed partial)")
        return "downloaded"
    if partial.exists() and partial.stat().st_size == remote.size:
        partial.unlink()
        print(f"[replace] Removed checksum-mismatched partial file {partial.name}")

    url = download_url(args, remote.path)
    checksum_attempts = 2 if verify_checksum and remote.sha256 else 1
    for checksum_attempt in range(1, checksum_attempts + 1):
        already_have = partial.stat().st_size if partial.exists() else 0
        print(
            f"[download] {remote.name} ({human_size(remote.size)}, "
            f"resuming at {human_size(already_have)})"
        )
        command = [
            curl,
            "--fail",
            "--location",
            "--continue-at",
            "-",
            "--retry",
            str(args.curl_retries),
            "--retry-delay",
            "5",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--output",
            str(partial),
            url,
        ]
        subprocess.run(command, check=True)

        actual_size = partial.stat().st_size
        if actual_size != remote.size:
            raise RuntimeError(
                f"size mismatch for {remote.name}: expected {remote.size}, got {actual_size}; "
                f"partial download kept at {partial}"
            )

        if verify_checksum and remote.sha256:
            print(f"[verify] {remote.name}")
            actual_digest = sha256_file(partial)
            if actual_digest != remote.sha256:
                if checksum_attempt < checksum_attempts:
                    print(
                        f"[retry] SHA-256 mismatch for {remote.name}; downloading it again",
                        file=sys.stderr,
                    )
                    partial.unlink()
                    continue
                raise RuntimeError(
                    f"SHA-256 mismatch for {remote.name}: expected {remote.sha256}, "
                    f"got {actual_digest}"
                )

        partial.replace(target)
        print(f"[done] {remote.name}")
        return "downloaded"

    raise AssertionError("unreachable")


def list_archives(archives: dict[str, SceneArchive]) -> None:
    print(f"Available scenes in {REPO_ID}/{REMOTE_DIR}:")
    for scene in sorted(archives, key=str.casefold):
        archive = archives[scene]
        numbered_count = sum(not item.is_terminal_zip for item in archive.files)
        print(
            f"  {scene:<26} {len(archive.files):>3} files "
            f"({numbered_count} numbered + .zip), {human_size(archive.total_size):>10}"
        )


def run(args: argparse.Namespace) -> None:
    print(f"[manifest] Reading {args.repo_id}@{args.revision}/{REMOTE_DIR}...")
    archives = build_scene_archives(fetch_manifest(args))
    if args.list:
        list_archives(archives)
        return

    selected = select_scenes(archives, args.scenes, args.all)
    total_size = sum(scene.total_size for scene in selected)
    print(
        f"[plan] {len(selected)} scene(s), "
        f"{sum(len(scene.files) for scene in selected)} files, {human_size(total_size)}"
    )
    for index, scene in enumerate(selected, start=1):
        print(
            f"\n=== Scene {index}/{len(selected)}: {scene.name} "
            f"({len(scene.files)} files, {human_size(scene.total_size)}) ==="
        )
        for item in scene.files:
            print(f"  {item.name:<34} {human_size(item.size):>10}")

    if args.dry_run:
        print("\n[dry-run] No files were downloaded.")
        return

    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for resumable downloads but was not found in PATH")

    archive_dir = args.archive_dir.expanduser().resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    for index, scene in enumerate(selected, start=1):
        print(f"\n=== Downloading scene {index}/{len(selected)}: {scene.name} ===")
        for remote in scene.files:
            result = download_file(args, curl, archive_dir, remote)
            downloaded += result == "downloaded"
            skipped += result == "skipped"
        print(
            f"=== Scene complete: {scene.name} "
            f"({len(scene.files)}/{len(scene.files)} files) ==="
        )

    print(
        f"\nAll selected scenes are complete. Downloaded {downloaded} file(s); "
        f"reused {skipped} existing file(s)."
    )
    print(f"Archives: {archive_dir}")


def main() -> int:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted; partial files were kept for the next run.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
