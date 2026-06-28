"""Mid/side helpers."""

from __future__ import annotations

import numpy as np


def mono_bass(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 100.0) -> np.ndarray:
    """Reduce low-frequency side energy below cutoff without changing duration."""
    if audio.shape[1] < 2 or audio.shape[0] < 32 or sample_rate / 2.0 <= cutoff_hz:
        return audio.copy()

    left = audio[:, 0]
    right = audio[:, 1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    side_spectrum = np.fft.rfft(side)
    freqs = np.fft.rfftfreq(side.size, d=1.0 / sample_rate)
    gain = np.ones_like(freqs)
    gain[freqs <= cutoff_hz] = 0.0
    transition = (freqs > cutoff_hz) & (freqs < cutoff_hz + 60.0)
    if np.any(transition):
        t = (freqs[transition] - cutoff_hz) / 60.0
        gain[transition] = 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0.0, 1.0))
    filtered_side = np.fft.irfft(side_spectrum * gain, n=side.size)
    processed = audio.copy()
    processed[:, 0] = mid + filtered_side
    processed[:, 1] = mid - filtered_side
    if audio.shape[1] > 2:
        processed[:, 2:] = audio[:, 2:]
    return processed
