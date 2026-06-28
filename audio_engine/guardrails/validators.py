"""Validation helpers for conservative processing results."""

from __future__ import annotations

from typing import Any

import numpy as np

from .limits import GuardrailLimits


def validate_render(
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    limits: GuardrailLimits,
) -> dict[str, Any]:
    """Validate that a rendered master stays inside conservative limits."""
    warnings: list[str] = []
    passed = True

    if after_metrics["sample_rate"] != before_metrics["sample_rate"] and limits.preserve_sample_rate:
        passed = False
        warnings.append("Sample rate changed despite preserve_sample_rate guardrail.")

    if after_metrics["estimated_true_peak_dbtp"] > limits.limiter_ceiling_dbtp + 0.2:
        passed = False
        warnings.append("Estimated true peak exceeds limiter ceiling guardrail.")

    before_width = float(before_metrics.get("mid_side_ratio", 0.0))
    after_width = float(after_metrics.get("mid_side_ratio", 0.0))
    if before_width > 1e-9:
        width_change_percent = abs(after_width - before_width) / before_width * 100.0
        low_end_was_risky = float(before_metrics.get("low_end_width", 0.0)) > 0.18
        low_end_improved = float(after_metrics.get("low_end_width", 0.0)) <= float(
            before_metrics.get("low_end_width", 0.0)
        )
        width_increased = after_width > before_width
        if (
            width_change_percent > limits.max_width_change_percent
            and (width_increased or not (low_end_was_risky and low_end_improved))
        ):
            passed = False
            warnings.append("Stereo width changed beyond guardrail.")

    return {"passed": passed, "warnings": warnings}


def ensure_finite_audio(audio: np.ndarray) -> np.ndarray:
    """Replace non-finite samples and prevent accidental hard clipping."""
    cleaned = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
    if peak > 1.0:
        cleaned = cleaned / peak
    return cleaned
