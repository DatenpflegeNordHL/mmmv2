"""Clipper helpers."""

from __future__ import annotations

import numpy as np


def soft_clip(audio: np.ndarray, ceiling_linear: float) -> np.ndarray:
    """Soft clip only near the ceiling."""
    ceiling = max(float(ceiling_linear), 1e-6)
    return ceiling * np.tanh(audio / ceiling)
