#!/usr/bin/env python3
"""Extract complete HaL-13k per-scene split ZIP archives.

The downloader stores scenes as ``Scene.z01``, ``Scene.z02``, ... and
``Scene.zip``.  This script validates the split set using the terminal ZIP
record, then exposes the parts as one seekable stream to Python's ZIP reader.
No joined archive or additional compressed-size temporary space is required.

Scenes are processed sequentially.  If validation or extraction fails, the
script stops without starting the next scene.

Examples:

    ./extract_hal13k_maps.py --list
    ./extract_hal13k_maps.py Carla_Town01
    ./extract_hal13k_maps.py Carla_Town01 Carla_Town02
    ./extract_hal13k_maps.py --all
"""

from __future__ import annotations

import argparse
import bisect
import io
import os
import re
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


DEFAULT_ARCHIVE_DIR = Path("aeroduo/data/Hal-13k/maps")
DEFAULT_OUTPUT_DIR = Path("aeroduo/data/Hal-13k")
ARCHIVE_RE = re.compile(r"^(?P<scene>.+)\.(?P<suffix>zip|z(?P<part>\d+))$", re.IGNORECASE)

EOCD_SIGNATURE = b"PK\x05\x06"
EOCD_STRUCT = struct.Struct("<4s4H2LH")
ZIP64_EOCD_SIGNATURE = b"PK\x06\x06"
ZIP64_EOCD_STRUCT = struct.Struct("<4sQ2H2L4Q")
ZIP64_LOCATOR_SIGNATURE = b"PK\x06\x07"
ZIP64_LOCATOR_STRUCT = struct.Struct("<4sLQL")
MAX_ZIP_COMMENT = 65535


@dataclass(frozen=True)
class LocalSceneArchive:
    name: str
    terminal_zip: Path | None
    numbered_parts: tuple[tuple[int, Path], ...]

    @property
    def files(self) -> tuple[Path, ...]:
        parts = tuple(path for _number, path in self.numbered_parts)
        return parts + ((self.terminal_zip,) if self.terminal_zip is not None else ())

    @property
    def total_size(self) -> int:
        return sum(path.stat().st_size for path in self.files)


@dataclass(frozen=True)
class SplitZipLayout:
    paths: tuple[Path, ...]
    starts: tuple[int, ...]
    sizes: tuple[int, ...]
    overlays: tuple[tuple[int, bytes], ...]
    zipfile_concat: int

    @property
    def total_size(self) -> int:
        return sum(self.sizes)


class SplitArchiveStream(io.RawIOBase):
    """Seekable read-only view over split ZIP files with small metadata patches."""

    def __init__(self, layout: SplitZipLayout):
        super().__init__()
        self.layout = layout
        self._position = 0
        self._handles: dict[int, BinaryIO] = {}

    @property
    def name(self) -> str:
        return str(self.layout.paths[-1])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self.layout.total_size + offset
        else:
            raise ValueError(f"invalid whence: {whence}")
        if position < 0:
            raise OSError("negative seek position")
        self._position = position
        return position

    def _handle(self, index: int) -> BinaryIO:
        handle = self._handles.get(index)
        if handle is None:
            handle = self.layout.paths[index].open("rb")
            self._handles[index] = handle
        return handle

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed split archive")
        available = max(0, self.layout.total_size - self._position)
        wanted = available if size is None or size < 0 else min(size, available)
        if wanted == 0:
            return b""

        read_start = self._position
        chunks: list[bytes] = []
        remaining = wanted
        while remaining:
            index = bisect.bisect_right(self.layout.starts, self._position) - 1
            if index < 0 or index >= len(self.layout.paths):
                break
            local_offset = self._position - self.layout.starts[index]
            segment_remaining = self.layout.sizes[index] - local_offset
            amount = min(remaining, segment_remaining)
            if amount <= 0:
                break
            handle = self._handle(index)
            handle.seek(local_offset)
            chunk = handle.read(amount)
            if not chunk:
                break
            chunks.append(chunk)
            self._position += len(chunk)
            remaining -= len(chunk)

        data = bytearray(b"".join(chunks))
        read_end = read_start + len(data)
        for overlay_start, replacement in self.layout.overlays:
            overlay_end = overlay_start + len(replacement)
            overlap_start = max(read_start, overlay_start)
            overlap_end = min(read_end, overlay_end)
            if overlap_start < overlap_end:
                destination = overlap_start - read_start
                source = overlap_start - overlay_start
                length = overlap_end - overlap_start
                data[destination : destination + length] = replacement[source : source + length]
        return bytes(data)

    def close(self) -> None:
        if not self.closed:
            for handle in self._handles.values():
                handle.close()
            self._handles.clear()
        super().close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and stream-extract complete HaL-13k split ZIP sets, "
            "finishing one scene before starting the next."
        )
    )
    parser.add_argument(
        "scenes",
        nargs="*",
        metavar="SCENE",
        help="Scene names to extract, in the order they should be processed.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Extract every local scene.")
    mode.add_argument("--list", action="store_true", help="List local scene archives and exit.")
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE_DIR,
        help=f"Directory containing archive parts (default: {DEFAULT_ARCHIVE_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Extraction destination (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files (the default skips them).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate archives and show the extraction plan without writing files.",
    )
    args = parser.parse_args()

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


def discover_archives(archive_dir: Path) -> dict[str, LocalSceneArchive]:
    grouped: dict[str, dict[str, object]] = {}
    if not archive_dir.is_dir():
        raise RuntimeError(f"archive directory does not exist: {archive_dir}")

    for path in archive_dir.iterdir():
        if not path.is_file():
            continue
        match = ARCHIVE_RE.fullmatch(path.name)
        if not match:
            continue
        scene = match.group("scene")
        group = grouped.setdefault(scene, {"terminal": None, "parts": []})
        part_text = match.group("part")
        if part_text is None:
            if group["terminal"] is not None:
                raise RuntimeError(f"scene {scene!r} has multiple terminal .zip files")
            group["terminal"] = path
        else:
            parts = group["parts"]
            assert isinstance(parts, list)
            parts.append((int(part_text), path))

    archives: dict[str, LocalSceneArchive] = {}
    for scene, group in grouped.items():
        terminal = group["terminal"]
        parts = group["parts"]
        assert terminal is None or isinstance(terminal, Path)
        assert isinstance(parts, list)
        archives[scene] = LocalSceneArchive(
            name=scene,
            terminal_zip=terminal,
            numbered_parts=tuple(sorted(parts)),
        )
    return archives


def read_terminal_eocd(terminal_zip: Path) -> tuple[int, list[int | bytes]]:
    file_size = terminal_zip.stat().st_size
    read_size = min(
        file_size,
        MAX_ZIP_COMMENT
        + EOCD_STRUCT.size
        + ZIP64_LOCATOR_STRUCT.size
        + ZIP64_EOCD_STRUCT.size,
    )
    with terminal_zip.open("rb") as handle:
        handle.seek(file_size - read_size)
        tail = handle.read(read_size)

    index = tail.rfind(EOCD_SIGNATURE)
    if index < 0 or index + EOCD_STRUCT.size > len(tail):
        raise RuntimeError(f"could not find a complete ZIP end record in {terminal_zip.name}")
    fields = list(EOCD_STRUCT.unpack_from(tail, index))
    comment_length = int(fields[7])
    if index + EOCD_STRUCT.size + comment_length != len(tail):
        raise RuntimeError(f"invalid ZIP end record in {terminal_zip.name}")
    local_offset = file_size - read_size + index
    return local_offset, fields


def expected_numbered_parts(terminal_zip: Path) -> int:
    _offset, fields = read_terminal_eocd(terminal_zip)
    disk_number = int(fields[1])
    if disk_number == 0xFFFF:
        raise RuntimeError(
            f"{terminal_zip.name} uses more than 65,535 split disks; this "
            "extractor cannot determine its part count"
        )
    return disk_number


def validate_archive(archive: LocalSceneArchive) -> int:
    if archive.terminal_zip is None:
        raise RuntimeError(f"scene {archive.name!r} is missing {archive.name}.zip")

    numbers = [number for number, _path in archive.numbered_parts]
    if len(numbers) != len(set(numbers)):
        raise RuntimeError(f"scene {archive.name!r} has duplicate numbered parts")
    if numbers and numbers != list(range(1, numbers[-1] + 1)):
        raise RuntimeError(
            f"scene {archive.name!r} has a gap in its local numbered parts: {numbers}"
        )

    expected = expected_numbered_parts(archive.terminal_zip)
    if numbers != list(range(1, expected + 1)):
        width = max(2, len(str(expected)))
        expected_names = (
            f"{archive.name}.z01 through {archive.name}.z{expected:0{width}d}"
            if expected
            else "no numbered parts"
        )
        raise RuntimeError(
            f"scene {archive.name!r} is incomplete: terminal ZIP expects "
            f"{expected_names}, but found {len(numbers)} numbered part(s)"
        )
    if any(path.stat().st_size == 0 for path in archive.files):
        raise RuntimeError(f"scene {archive.name!r} contains an empty archive part")
    return expected


def make_starts(sizes: tuple[int, ...]) -> tuple[int, ...]:
    starts: list[int] = []
    offset = 0
    for size in sizes:
        starts.append(offset)
        offset += size
    return tuple(starts)


def pack_eocd_with_single_disk(
    fields: list[int | bytes], central_directory_offset: int
) -> bytes:
    fields[1] = 0
    fields[2] = 0
    fields[3] = fields[4]
    if int(fields[6]) != 0xFFFFFFFF:
        if central_directory_offset > 0xFFFFFFFF:
            raise RuntimeError("ZIP central-directory offset exceeds its 32-bit field")
        fields[6] = central_directory_offset
    return EOCD_STRUCT.pack(*fields)


def build_layout(archive: LocalSceneArchive) -> SplitZipLayout:
    assert archive.terminal_zip is not None
    paths = archive.files
    sizes = tuple(path.stat().st_size for path in paths)
    starts = make_starts(sizes)
    terminal_index = len(paths) - 1
    eocd_local_offset, eocd_fields = read_terminal_eocd(archive.terminal_zip)
    eocd_global_offset = starts[terminal_index] + eocd_local_offset
    overlays: list[tuple[int, bytes]] = []
    zipfile_concat = 0

    locator_global_offset = eocd_global_offset - ZIP64_LOCATOR_STRUCT.size
    locator_bytes = b""
    if eocd_local_offset >= ZIP64_LOCATOR_STRUCT.size:
        with archive.terminal_zip.open("rb") as handle:
            handle.seek(eocd_local_offset - ZIP64_LOCATOR_STRUCT.size)
            locator_bytes = handle.read(ZIP64_LOCATOR_STRUCT.size)

    if locator_bytes.startswith(ZIP64_LOCATOR_SIGNATURE):
        locator_fields = list(ZIP64_LOCATOR_STRUCT.unpack(locator_bytes))
        zip64_disk = int(locator_fields[1])
        zip64_local_offset = int(locator_fields[2])
        if zip64_disk >= len(paths):
            raise RuntimeError(f"invalid ZIP64 disk number in {archive.terminal_zip.name}")
        zip64_global_offset = starts[zip64_disk] + zip64_local_offset
        expected_zip64_offset = locator_global_offset - ZIP64_EOCD_STRUCT.size
        if zip64_global_offset != expected_zip64_offset:
            raise RuntimeError(
                f"unsupported extensible ZIP64 end record in {archive.terminal_zip.name}"
            )

        with paths[zip64_disk].open("rb") as handle:
            handle.seek(zip64_local_offset)
            zip64_bytes = handle.read(ZIP64_EOCD_STRUCT.size)
        if len(zip64_bytes) != ZIP64_EOCD_STRUCT.size:
            raise RuntimeError(f"truncated ZIP64 end record in {archive.terminal_zip.name}")
        zip64_fields = list(ZIP64_EOCD_STRUCT.unpack(zip64_bytes))
        if zip64_fields[0] != ZIP64_EOCD_SIGNATURE:
            raise RuntimeError(f"invalid ZIP64 end record in {archive.terminal_zip.name}")

        cd_disk = int(zip64_fields[5])
        cd_local_offset = int(zip64_fields[9])
        if cd_disk >= len(paths):
            raise RuntimeError(f"invalid central-directory disk in {archive.terminal_zip.name}")
        absolute_cd_offset = starts[cd_disk] + cd_local_offset
        # The ZIP64 record below supplies the absolute combined offset to
        # zipfile.  Keep the legacy EOCD field volume-relative so it remains
        # representable when the complete split archive exceeds 4 GiB.
        eocd_cd_offset = cd_local_offset
        zip64_fields[4] = 0
        zip64_fields[5] = 0
        zip64_fields[6] = zip64_fields[7]
        zip64_fields[9] = absolute_cd_offset
        overlays.append((zip64_global_offset, ZIP64_EOCD_STRUCT.pack(*zip64_fields)))

        locator_fields[1] = 0
        locator_fields[2] = zip64_global_offset
        locator_fields[3] = 1
        overlays.append(
            (locator_global_offset, ZIP64_LOCATOR_STRUCT.pack(*locator_fields))
        )
    else:
        cd_disk = int(eocd_fields[2])
        cd_local_offset = int(eocd_fields[6])
        if cd_disk >= len(paths):
            raise RuntimeError(f"invalid central-directory disk in {archive.terminal_zip.name}")
        absolute_cd_offset = starts[cd_disk] + cd_local_offset

        if absolute_cd_offset <= 0xFFFFFFFF:
            # A normal one-disk EOCD can represent the absolute offset.
            eocd_cd_offset = absolute_cd_offset
        else:
            # Some very large split ZIPs do not carry a ZIP64 end record
            # because every per-volume offset still fits in 32 bits.  Keep
            # the central directory's per-volume offset and let zipfile's
            # self-extracting-archive adjustment supply the volume start.
            # We remove that common adjustment from each member below and
            # replace it with the member's actual volume start.
            eocd_cd_offset = cd_local_offset
            zipfile_concat = starts[cd_disk]
            inferred_concat = (
                eocd_global_offset - int(eocd_fields[5]) - cd_local_offset
            )
            if inferred_concat != zipfile_concat:
                raise RuntimeError(
                    f"cannot map the multi-volume central directory in "
                    f"{archive.terminal_zip.name}"
                )

    overlays.append(
        (
            eocd_global_offset,
            pack_eocd_with_single_disk(eocd_fields, eocd_cd_offset),
        )
    )
    return SplitZipLayout(
        paths=paths,
        starts=starts,
        sizes=sizes,
        overlays=tuple(sorted(overlays)),
        zipfile_concat=zipfile_concat,
    )


def select_archives(
    archives: dict[str, LocalSceneArchive], requested: list[str], select_all: bool
) -> list[LocalSceneArchive]:
    if not archives:
        raise RuntimeError("no local HaL-13k map archives were found")
    if select_all:
        return [archives[name] for name in sorted(archives, key=str.casefold)]

    by_casefold = {name.casefold(): name for name in archives}
    selected: list[LocalSceneArchive] = []
    seen: set[str] = set()
    for requested_name in requested:
        canonical = by_casefold.get(requested_name.casefold())
        if canonical is None:
            available = ", ".join(sorted(archives, key=str.casefold))
            raise RuntimeError(
                f"unknown local scene {requested_name!r}. Available scenes: {available}"
            )
        if canonical not in seen:
            selected.append(archives[canonical])
            seen.add(canonical)
    return selected


def member_target(output_dir: Path, member: zipfile.ZipInfo) -> Path:
    """Mirror zipfile's path sanitizing and reject pre-existing symlink escapes."""
    arcname = member.filename.replace("/", os.path.sep)
    if os.path.altsep:
        arcname = arcname.replace(os.path.altsep, os.path.sep)
    arcname = os.path.splitdrive(arcname)[1]
    invalid = ("", os.path.curdir, os.path.pardir)
    arcname = os.path.sep.join(part for part in arcname.split(os.path.sep) if part not in invalid)
    if not arcname and not member.is_dir():
        raise RuntimeError(f"unsafe empty path in ZIP member {member.filename!r}")

    target = output_dir / arcname
    root = output_dir.resolve()
    resolved = target.resolve(strict=False)
    try:
        common = Path(os.path.commonpath((root, resolved)))
    except ValueError as exc:
        raise RuntimeError(f"unsafe ZIP member path: {member.filename!r}") from exc
    if common != root:
        raise RuntimeError(f"ZIP member escapes output directory: {member.filename!r}")
    return target


def normalize_member_offsets(
    zip_archive: zipfile.ZipFile, layout: SplitZipLayout
) -> list[zipfile.ZipInfo]:
    """Map per-volume member offsets and rebuild Python's overlap boundaries."""
    members = zip_archive.infolist()
    for member in members:
        original_volume = member.volume
        if original_volume >= len(layout.starts):
            raise RuntimeError(
                f"ZIP member {member.filename!r} refers to missing disk "
                f"{original_volume}"
            )
        member.header_offset += (
            layout.starts[original_volume] - layout.zipfile_concat
        )
        member.volume = 0

    # Python 3.11+ records the next member boundary while initially reading
    # the central directory. Those boundaries were computed from the original
    # per-volume offsets, so rebuild them after mapping the offsets into our
    # concatenated address space. Older versions accept this otherwise-unused
    # attribute as well.
    if members and hasattr(members[0], "_end_offset"):
        end_offset = zip_archive.start_dir
        for member in sorted(
            members, key=lambda item: item.header_offset, reverse=True
        ):
            member._end_offset = end_offset
            end_offset = member.header_offset
    return members


def extract_archive(
    archive: LocalSceneArchive,
    output_dir: Path,
    overwrite: bool,
) -> tuple[int, int]:
    layout = build_layout(archive)
    extracted = 0
    skipped = 0
    with SplitArchiveStream(layout) as stream:
        with zipfile.ZipFile(stream, mode="r") as zip_archive:
            members = normalize_member_offsets(zip_archive, layout)

            print(f"[archive] {len(members)} member(s)")
            for index, member in enumerate(members, start=1):
                target = member_target(output_dir, member)
                if target.exists() and not overwrite:
                    skipped += 1
                else:
                    zip_archive.extract(member, path=output_dir)
                    extracted += 1
                if index % 1000 == 0 or index == len(members):
                    print(f"[progress] {index}/{len(members)} archive members")
    return extracted, skipped


def list_archives(archives: dict[str, LocalSceneArchive]) -> None:
    if not archives:
        print("No local HaL-13k map archives found.")
        return
    print("Local scene archives:")
    for name in sorted(archives, key=str.casefold):
        archive = archives[name]
        try:
            numbered_count = validate_archive(archive)
            status = f"complete ({numbered_count} numbered + .zip)"
        except RuntimeError as exc:
            status = f"INCOMPLETE ({exc})"
        print(f"  {name:<26} {human_size(archive.total_size):>10}  {status}")


def run(args: argparse.Namespace) -> None:
    archive_dir = args.archive_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    archives = discover_archives(archive_dir)
    if args.list:
        list_archives(archives)
        return

    selected = select_archives(archives, args.scenes, args.all)
    for archive in selected:
        validate_archive(archive)

    print(
        f"[plan] {len(selected)} scene(s), {human_size(sum(a.total_size for a in selected))}; "
        f"output: {output_dir}"
    )
    if args.dry_run:
        for archive in selected:
            print(
                f"  {archive.name}: {len(archive.files)} complete archive files, "
                f"{human_size(archive.total_size)}"
            )
        print("[dry-run] No files were extracted.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    total_extracted = 0
    total_skipped = 0
    for index, archive in enumerate(selected, start=1):
        print(
            f"\n=== Extracting scene {index}/{len(selected)}: {archive.name} "
            f"({len(archive.files)} files, {human_size(archive.total_size)}) ==="
        )
        extracted, skipped = extract_archive(archive, output_dir, args.overwrite)
        total_extracted += extracted
        total_skipped += skipped
        print(
            f"=== Scene extraction complete: {archive.name} "
            f"({extracted} extracted, {skipped} existing skipped) ==="
        )

    print(
        f"\nAll selected scenes were extracted to {output_dir}. "
        f"Extracted {total_extracted} member(s); skipped {total_skipped}."
    )


def main() -> int:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted; no later scene was started.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, struct.error, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
