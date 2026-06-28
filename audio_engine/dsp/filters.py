"""Filter helpers."""

from __future__ import annotations

import numpy as np
from scipy import signal


def highpass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 25.0) -> np.ndarray:
    """Apply a conservative zero-phase highpass filter."""
    if sample_rate / 2.0 <= cutoff_hz or audio.shape[0] < 32:
        return audio.copy()
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)
