"""Gain staging helpers."""

from __future__ import annotations

import numpy as np


def apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply static gain in dB."""
    return audio * (10.0 ** (float(gain_db) / 20.0))
