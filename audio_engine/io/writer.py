"""Audio export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


SUBTYPE_BY_BIT_DEPTH = {
    16: "PCM_16",
    24: "PCM_24",
    32: "FLOAT",
}


def write_audio(
    path: str | Path,
    audio: np.ndarray,
    sample_rate: int,
    *,
    bit_depth: int = 24,
) -> Path:
    """Write audio to WAV/FLAC/AIFF while preserving sample rate."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subtype = SUBTYPE_BY_BIT_DEPTH.get(int(bit_depth), "PCM_24")
    write_data = np.asarray(audio, dtype=np.float64)
    if write_data.ndim == 2 and write_data.shape[1] == 1:
        write_data = write_data[:, 0]
    sf.write(str(output_path), write_data, sample_rate, subtype=subtype)
    return output_path
