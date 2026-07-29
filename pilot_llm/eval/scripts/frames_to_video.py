"""Turn a folder of numbered frames (e.g. an eval run's frontcamera/) into a video.

Usage:
    python frames_to_video.py <frames_dir> [-o out.mp4] [--fps 5]

Frames are sorted by the number in their filename (000000.png, 000001.png, ...),
matching the step-indexed naming used by save_images in eval.py.

Encodes H.264 via imageio-ffmpeg's bundled ffmpeg so the result plays in
VS Code / browsers. Falls back to OpenCV's mp4v (MPEG-4 Part 2, playable in
VLC but not Chromium-based players) if imageio-ffmpeg is unavailable.
"""

import argparse
import os
import re
import subprocess
import sys

import cv2


def numeric_key(filename: str) -> int:
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1


def list_frames(frames_dir: str) -> list:
    exts = ('.png', '.jpg', '.jpeg')
    frames = sorted(
        (f for f in os.listdir(frames_dir) if f.lower().endswith(exts)),
        key=numeric_key,
    )
    if not frames:
        sys.exit(f"error: no image files in {frames_dir}")
    return frames


def write_h264(frames_dir, frames, output, fps, width, height) -> bool:
    """Encode with imageio-ffmpeg's bundled ffmpeg (libx264). Returns False if
    imageio-ffmpeg isn't installed."""
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
    except ImportError:
        return False

    # H.264 needs even dimensions; yuv420p is what browser players require.
    even_w, even_h = width - width % 2, height - height % 2
    cmd = [
        get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-",
        "-vf", f"crop={even_w}:{even_h}:0:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        output,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    written = 0
    for name in frames:
        img = cv2.imread(os.path.join(frames_dir, name))
        if img is None:
            print(f"warning: skipping unreadable frame {name}")
            continue
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        proc.stdin.write(img.tobytes())
        written += 1
    proc.stdin.close()
    if proc.wait() != 0:
        sys.exit("error: ffmpeg encoding failed")
    print(f"wrote {written}/{len(frames)} frames to {output} "
          f"({even_w}x{even_h} @ {fps} fps, H.264)")
    return True


def write_mp4v(frames_dir, frames, output, fps, width, height) -> None:
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        sys.exit(f"error: could not open video writer for {output}")
    written = 0
    for name in frames:
        img = cv2.imread(os.path.join(frames_dir, name))
        if img is None:
            print(f"warning: skipping unreadable frame {name}")
            continue
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        writer.write(img)
        written += 1
    writer.release()
    print(f"wrote {written}/{len(frames)} frames to {output} "
          f"({width}x{height} @ {fps} fps, mp4v — use VLC, not VS Code)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an image sequence folder to a video.")
    parser.add_argument("frames_dir",
                        help="Folder containing the frames (e.g. .../frontcamera)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output video path (default: <frames_dir>.mp4)")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frames per second (default: 5)")
    args = parser.parse_args()

    frames_dir = os.path.abspath(args.frames_dir)
    if not os.path.isdir(frames_dir):
        sys.exit(f"error: not a directory: {frames_dir}")

    frames = list_frames(frames_dir)
    output = args.output or frames_dir.rstrip(os.sep) + ".mp4"

    first = cv2.imread(os.path.join(frames_dir, frames[0]))
    if first is None:
        sys.exit(f"error: could not read {frames[0]}")
    height, width = first.shape[:2]

    if not write_h264(frames_dir, frames, output, args.fps, width, height):
        print("imageio-ffmpeg not found; falling back to mp4v")
        write_mp4v(frames_dir, frames, output, args.fps, width, height)


if __name__ == "__main__":
    main()
