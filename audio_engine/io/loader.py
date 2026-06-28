"""Audio file loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SUPPORTED_INPUT_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif"}


def load_audio(path: str | Path) -> dict[str, Any]:
    """Load WAV/FLAC/AIFF audio as float64 with shape ``(samples, channels)``."""
    input_path = Path(path)
    if input_path.suffix.casefold() not in SUPPORTED_INPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise ValueError(f"Unsupported quality-engine input format. Supported: {supported}")
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    info = sf.info(str(input_path))
    audio, sample_rate = sf.read(str(input_path), dtype="float64", always_2d=True)
    if audio.size == 0:
        raise ValueError(f"Audio file contains no samples: {input_path}")

    return {
        "path": input_path,
        "audio": np.asarray(audio, dtype=np.float64),
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "frames": int(audio.shape[0]),
        "duration_seconds": float(audio.shape[0] / sample_rate),
        "format": info.format,
        "subtype": info.subtype,
    }
