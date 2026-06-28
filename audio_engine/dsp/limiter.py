"""Final peak limiter."""

from __future__ import annotations

import numpy as np

from audio_engine.analysis.loudness import linear_from_db


def peak_limiter(audio: np.ndarray, ceiling_dbtp: float = -1.5) -> tuple[np.ndarray, dict[str, float]]:
    """Apply deterministic peak ceiling scaling."""
    ceiling = linear_from_db(ceiling_dbtp)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= ceiling:
        return audio.copy(), {"gain_reduction_db": 0.0, "ceiling_dbtp": float(ceiling_dbtp)}
    gain = ceiling / max(peak, 1e-12)
    return audio * gain, {
        "gain_reduction_db": float(-20.0 * np.log10(gain)),
        "ceiling_dbtp": float(ceiling_dbtp),
    }
