"""Before/after comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.analysis.readiness import analyze_quality


COMPARE_KEYS = (
    "integrated_lufs",
    "estimated_true_peak_dbtp",
    "sample_peak_dbfs",
    "crest_factor",
    "peak_to_loudness_ratio",
    "stereo_correlation",
    "mid_side_ratio",
    "low_end_width",
    "spectral_centroid",
    "spectral_rolloff",
    "clipping_sample_count",
)


def compare_masters(reference_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    """Analyze and compare an original file against a rendered master."""
    before = analyze_quality(reference_path)
    after = analyze_quality(candidate_path)
    return {
        "action": "compare_master",
        "reference": str(Path(reference_path)),
        "candidate": str(Path(candidate_path)),
        "before": before,
        "after": after,
        "comparison": compare_metric_dicts(before, after),
    }


def compare_metric_dicts(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return simple numeric deltas for key mastering metrics."""
    deltas = {}
    for key in COMPARE_KEYS:
        before_value = before.get(key)
        after_value = after.get(key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            deltas[f"{key}_before"] = before_value
            deltas[f"{key}_after"] = after_value
            deltas[f"{key}_delta"] = float(after_value - before_value)
    return deltas
