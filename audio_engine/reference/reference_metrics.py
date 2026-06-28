"""Reference-track comparison metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.analysis.readiness import analyze_quality


def reference_qc(target_path: str | Path, reference_path: str | Path) -> dict[str, Any]:
    """Compare target and reference without copying the reference curve."""
    target = analyze_quality(target_path)
    reference = analyze_quality(reference_path)
    return {
        "action": "reference_qc",
        "target": target,
        "reference": reference,
        "suggestions": _suggestions(target, reference),
    }


def _suggestions(target: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    suggestions = []
    if target["integrated_lufs"] < reference["integrated_lufs"] - 3.0:
        suggestions.append("Target is materially quieter than reference; consider moderate gain staging.")
    if target["band_energy"]["low_mid_120_350_hz"] > reference["band_energy"]["low_mid_120_350_hz"] + 0.08:
        suggestions.append("Target has more low-mid energy than reference; consider a small low-mid cut.")
    if target["low_end_width"] > reference["low_end_width"] + 0.08:
        suggestions.append("Target has wider low-end than reference; check mono bass.")
    return suggestions
