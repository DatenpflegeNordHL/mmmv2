"""Subtle saturation helpers."""

from __future__ import annotations

import numpy as np


def subtle_tanh_saturation(audio: np.ndarray, drive: float = 1.015, mix: float = 0.12) -> np.ndarray:
    """Blend a tiny amount of tanh saturation while preserving level."""
    if drive <= 1.0 or mix <= 0.0:
        return audio.copy()
    saturated = np.tanh(audio * drive) / np.tanh(drive)
    return (1.0 - mix) * audio + mix * saturated
