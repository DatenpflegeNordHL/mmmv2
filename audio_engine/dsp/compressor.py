"""Light bus compression."""

from __future__ import annotations

import numpy as np
from scipy import signal


def light_bus_compressor(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = -16.0,
    ratio: float = 1.35,
    max_gain_reduction_db: float = 1.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply very light RMS-envelope bus compression."""
    mono = np.mean(audio, axis=1)
    frame = max(64, int(0.02 * sample_rate))
    rms = np.sqrt(signal.convolve(mono**2, np.ones(frame) / frame, mode="same") + 1e-12)
    level_db = 20.0 * np.log10(rms + 1e-12)
    over_db = np.maximum(0.0, level_db - threshold_dbfs)
    gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
    gain_reduction_db = np.minimum(gain_reduction_db, max_gain_reduction_db)
    smoothing = max(3, int(0.08 * sample_rate))
    envelope = signal.convolve(gain_reduction_db, np.ones(smoothing) / smoothing, mode="same")
    gain = 10.0 ** (-envelope / 20.0)
    processed = audio * gain[:, None]
    return processed, {"max_gain_reduction_db": float(np.max(envelope))}
