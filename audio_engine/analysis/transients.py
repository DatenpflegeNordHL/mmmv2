"""Lightweight transient analysis placeholders for release reports."""

from __future__ import annotations

import numpy as np


def transient_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    """Estimate transient density from RMS envelope derivative."""
    mono = np.mean(audio, axis=1)
    frame = max(64, int(0.02 * sample_rate))
    hop = max(16, frame // 2)
    if mono.size < frame:
        return {"transient_density_per_second": 0.0, "transient_count": 0}
    rms = []
    for start in range(0, mono.size - frame + 1, hop):
        segment = mono[start : start + frame]
        rms.append(float(np.sqrt(np.mean(segment**2))))
    envelope = np.asarray(rms)
    novelty = np.diff(envelope, prepend=envelope[0])
    threshold = float(np.percentile(novelty, 90))
    count = int(np.sum(novelty > max(threshold, 1e-6)))
    duration = mono.size / sample_rate
    return {
        "transient_density_per_second": float(count / max(duration, 1e-9)),
        "transient_count": count,
    }
