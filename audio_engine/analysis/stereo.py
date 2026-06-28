"""Stereo and mono-compatibility metrics."""

from __future__ import annotations

import numpy as np


def stereo_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | None]:
    """Calculate stereo correlation, mid/side ratio, and low-end width."""
    if audio.shape[1] < 2:
        return {
            "stereo_correlation": None,
            "mid_side_ratio": 0.0,
            "low_end_width": 0.0,
        }

    left = audio[:, 0]
    right = audio[:, 1]
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        correlation = 1.0
    else:
        correlation = float(np.corrcoef(left, right)[0, 1])

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = float(np.sqrt(np.mean(mid**2))) + 1e-12
    side_rms = float(np.sqrt(np.mean(side**2)))

    return {
        "stereo_correlation": correlation,
        "mid_side_ratio": float(side_rms / mid_rms),
        "low_end_width": _low_end_width(mid, side, sample_rate),
    }


def _low_end_width(mid: np.ndarray, side: np.ndarray, sample_rate: int) -> float:
    if mid.size == 0:
        return 0.0
    freqs = np.fft.rfftfreq(mid.size, d=1.0 / sample_rate)
    mask = freqs <= 120.0
    mid_power = np.abs(np.fft.rfft(mid)) ** 2
    side_power = np.abs(np.fft.rfft(side)) ** 2
    return float(np.sum(side_power[mask]) / (np.sum(mid_power[mask]) + 1e-20))
