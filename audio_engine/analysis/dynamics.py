"""Dynamics and sample integrity metrics."""

from __future__ import annotations

import numpy as np

from .loudness import dbfs


def dynamics_metrics(audio: np.ndarray) -> dict[str, float | int | list[float]]:
    """Calculate crest factor, clipping count, and DC offset."""
    mono = np.mean(audio, axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
    clipping_count = int(np.sum(np.abs(audio) >= 0.999))
    dc_offset = [float(np.mean(audio[:, idx])) for idx in range(audio.shape[1])]

    return {
        "crest_factor": float(dbfs(peak) - dbfs(rms)) if rms > 0 else 0.0,
        "dc_offset": dc_offset,
        "clipping_sample_count": clipping_count,
    }
