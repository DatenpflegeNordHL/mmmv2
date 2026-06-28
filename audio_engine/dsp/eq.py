"""Static EQ helpers."""

from __future__ import annotations

import numpy as np

from audio_engine.guardrails.limits import clamp_db


def smooth_band_gain(
    freqs: np.ndarray,
    low_hz: float,
    high_hz: float,
    gain_db: float,
    transition_hz: float = 160.0,
) -> np.ndarray:
    """Create a smooth FFT-domain gain mask with raised-cosine transitions."""
    target = 10.0 ** (float(gain_db) / 20.0)
    gain = np.ones_like(freqs, dtype=np.float64)
    inside = (freqs >= low_hz) & (freqs <= high_hz)
    gain[inside] = target

    lower = (freqs >= max(0.0, low_hz - transition_hz)) & (freqs < low_hz)
    if np.any(lower):
        t = (freqs[lower] - max(0.0, low_hz - transition_hz)) / max(transition_hz, 1e-9)
        gain[lower] = 1.0 + (target - 1.0) * _half_cosine(t)

    upper = (freqs > high_hz) & (freqs <= high_hz + transition_hz)
    if np.any(upper):
        t = (freqs[upper] - high_hz) / max(transition_hz, 1e-9)
        gain[upper] = target + (1.0 - target) * _half_cosine(t)
    return gain


def fft_band_eq(
    audio: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    gain_db: float,
    *,
    max_gain_db: float = 2.0,
) -> np.ndarray:
    """Apply a gentle linear-phase band EQ."""
    if audio.shape[0] < 32 or sample_rate / 2.0 <= low_hz:
        return audio.copy()
    gain_db = clamp_db(gain_db, max_gain_db)
    high_hz = min(high_hz, sample_rate / 2.0)
    spectrum = np.fft.rfft(audio, axis=0)
    freqs = np.fft.rfftfreq(audio.shape[0], d=1.0 / sample_rate)
    gain = smooth_band_gain(freqs, low_hz, high_hz, gain_db)[:, None]
    return np.fft.irfft(spectrum * gain, n=audio.shape[0], axis=0).astype(np.float64)


def _half_cosine(t: np.ndarray) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
