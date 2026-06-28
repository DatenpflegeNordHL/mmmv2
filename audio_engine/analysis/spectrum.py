"""Spectral balance analysis."""

from __future__ import annotations

import numpy as np


BANDS = {
    "sub_20_60_hz": (20.0, 60.0),
    "bass_60_120_hz": (60.0, 120.0),
    "low_mid_120_350_hz": (120.0, 350.0),
    "mid_350_2000_hz": (350.0, 2000.0),
    "presence_2000_5000_hz": (2000.0, 5000.0),
    "harsh_5000_9000_hz": (5000.0, 9000.0),
    "air_9000_16000_hz": (9000.0, 16000.0),
}


def spectrum_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | dict[str, float]]:
    """Calculate centroid, rolloff, and normalized band energy."""
    mono = np.mean(audio, axis=1)
    if mono.size == 0:
        return {"spectral_centroid": 0.0, "spectral_rolloff": 0.0, "band_energy": {}}

    windowed = mono * np.hanning(mono.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sample_rate)
    total = float(np.sum(spectrum)) + 1e-20
    centroid = float(np.sum(freqs * spectrum) / total)
    cumulative = np.cumsum(spectrum)
    rolloff_idx = int(np.searchsorted(cumulative, 0.85 * cumulative[-1]))
    rolloff = float(freqs[min(rolloff_idx, freqs.size - 1)])

    band_energy: dict[str, float] = {}
    for name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < min(high, sample_rate / 2.0))
        band_energy[name] = float(np.sum(spectrum[mask]) / total) if np.any(mask) else 0.0

    return {
        "spectral_centroid": centroid,
        "spectral_rolloff": rolloff,
        "band_energy": band_energy,
    }
