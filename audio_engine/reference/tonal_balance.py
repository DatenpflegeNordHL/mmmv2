"""Tonal balance helper placeholders for future reference QC expansion."""

from __future__ import annotations


def band_delta(target_band_energy: dict[str, float], reference_band_energy: dict[str, float]) -> dict[str, float]:
    """Return target-minus-reference band energy deltas."""
    keys = set(target_band_energy) | set(reference_band_energy)
    return {key: float(target_band_energy.get(key, 0.0) - reference_band_energy.get(key, 0.0)) for key in keys}
