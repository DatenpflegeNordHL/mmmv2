"""Lightweight dynamic EQ decision helpers."""

from __future__ import annotations


def suggested_cuts_from_metrics(metrics: dict) -> list[dict[str, float | str]]:
    """Return conservative EQ cuts based on analysis metrics."""
    band_energy = metrics.get("band_energy", {})
    cuts: list[dict[str, float | str]] = []
    if band_energy.get("low_mid_120_350_hz", 0.0) > 0.34:
        cuts.append({"name": "low_mid_cleanup", "low_hz": 180.0, "high_hz": 350.0, "gain_db": -1.2})
    if band_energy.get("harsh_5000_9000_hz", 0.0) > 0.18:
        cuts.append({"name": "gentle_deharsh", "low_hz": 5500.0, "high_hz": 8500.0, "gain_db": -1.0})
    return cuts
