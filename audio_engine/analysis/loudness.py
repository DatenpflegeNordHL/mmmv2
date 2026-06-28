"""Loudness and peak metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal


EPSILON = 1e-12


def dbfs(value: float) -> float:
    """Convert linear full-scale amplitude to dBFS."""
    return float(20.0 * np.log10(max(float(value), EPSILON)))


def linear_from_db(db_value: float) -> float:
    """Convert dB value to a linear amplitude scalar."""
    return float(10.0 ** (float(db_value) / 20.0))


def loudness_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Calculate robust loudness metrics with optional pyloudnorm support."""
    mono = np.mean(audio, axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0

    method = "rms_approximation"
    integrated_lufs = dbfs(rms) - 0.691
    loudness_range_lra = None

    try:
        import pyloudnorm as pyln  # type: ignore

        meter = pyln.Meter(sample_rate)
        integrated_lufs = float(meter.integrated_loudness(audio))
        method = "pyloudnorm_bs1770"
    except Exception:
        pass

    short_term_curve = short_term_loudness_curve(mono, sample_rate)
    if short_term_curve:
        values = np.array([point["lufs"] for point in short_term_curve], dtype=float)
        loudness_range_lra = float(np.percentile(values, 95) - np.percentile(values, 10))

    return {
        "sample_peak_dbfs": dbfs(peak),
        "estimated_true_peak_dbtp": estimate_true_peak_dbtp(audio),
        "integrated_lufs": integrated_lufs,
        "loudness_method": method,
        "loudness_range_lra": loudness_range_lra,
        "short_term_loudness_curve": short_term_curve,
        "peak_to_loudness_ratio": float(dbfs(peak) - integrated_lufs),
    }


def estimate_true_peak_dbtp(audio: np.ndarray, oversample: int = 4) -> float:
    """Estimate true peak by polyphase oversampling each channel."""
    if audio.size == 0:
        return float("-inf")
    peaks = []
    for channel_idx in range(audio.shape[1]):
        channel = audio[:, channel_idx]
        if channel.size < 8:
            peaks.append(float(np.max(np.abs(channel))))
            continue
        upsampled = signal.resample_poly(channel, oversample, 1)
        peaks.append(float(np.max(np.abs(upsampled))))
    return dbfs(max(peaks) if peaks else 0.0)


def short_term_loudness_curve(mono_audio: np.ndarray, sample_rate: int) -> list[dict[str, float]]:
    """Return a lightweight 3-second short-term loudness curve."""
    if mono_audio.size == 0 or sample_rate <= 0:
        return []
    window = max(1, int(3.0 * sample_rate))
    hop = max(1, int(1.0 * sample_rate))
    if mono_audio.size < window:
        rms = float(np.sqrt(np.mean(mono_audio**2)))
        return [{"time_seconds": 0.0, "lufs": dbfs(rms) - 0.691}]

    curve = []
    for start in range(0, mono_audio.size - window + 1, hop):
        frame = mono_audio[start : start + window]
        rms = float(np.sqrt(np.mean(frame**2)))
        curve.append({"time_seconds": float(start / sample_rate), "lufs": dbfs(rms) - 0.691})
    return curve
