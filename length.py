import statistics

import subprocess
from pathlib import Path

VIDEO_DIR = Path("./training")
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv"}


def get_duration(path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def format_hms(seconds: float) -> str:
    h, rem = divmod(int(round(seconds)), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


total = 0.0
count = 0
times = []
for f in sorted(VIDEO_DIR.rglob("*")):  # use .glob("*") to skip subfolders
    # print(f)
    if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
        # print("reading file")
        try:
            d = get_duration(f)
            total += d
            times.append(d)
            count += 1
            print(f"{format_hms(d)}  {f.name}")
        except (subprocess.CalledProcessError, ValueError):
            print(f"[skipped] could not read duration: {f.name}")

print(f"\n{count} videos, total length: {format_hms(total)} ({total:.2f} seconds)")
print(
    f"median video length: {statistics.median(times)}, mean video length: {statistics.mean(times)}, num videos: {len(times)}"
)
