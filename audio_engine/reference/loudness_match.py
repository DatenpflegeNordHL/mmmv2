"""Loudness matching helper for reference QC."""

from __future__ import annotations


def gain_to_match_lufs(target_lufs: float, reference_lufs: float, max_gain_db: float = 3.0) -> float:
    """Return a conservative gain suggestion to approach reference loudness."""
    return max(-max_gain_db, min(max_gain_db, float(reference_lufs - target_lufs)))
