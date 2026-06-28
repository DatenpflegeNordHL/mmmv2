"""Optional FFmpeg helpers used by future codec preview/export workflows."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_available() -> bool:
    """Return whether an ffmpeg executable is available on PATH."""
    return shutil.which("ffmpeg") is not None


def convert_with_ffmpeg(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert an audio file with ffmpeg using deterministic non-interactive flags."""
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not available on PATH")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return Path(output_path)
