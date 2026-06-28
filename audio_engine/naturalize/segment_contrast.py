"""Segment contrast analysis helpers."""

from __future__ import annotations


def segment_gain_for_type(segment_type: str) -> float:
    """Return tiny pre-limiter gain movement for a segment type."""
    if segment_type == "quiet":
        return -0.25
    if segment_type == "dense":
        return 0.15
    return 0.0
