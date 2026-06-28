"""Naturalize render entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from audio_engine.analysis.readiness import analyze_quality
from audio_engine.dsp.pipeline import render_safe_master
from audio_engine.guardrails.limits import DEFAULT_LIMITS, GuardrailLimits
from audio_engine.io.loader import load_audio
from audio_engine.io.writer import write_audio
from audio_engine.naturalize.energy_automation import apply_energy_automation, build_energy_automation
from audio_engine.reports.before_after import compare_metric_dicts


def render_naturalized_master(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limits: GuardrailLimits = DEFAULT_LIMITS,
) -> dict[str, Any]:
    """Apply subtle segment automation before the safe-master chain."""
    loaded = load_audio(input_path)
    automation = build_energy_automation(loaded["audio"], loaded["sample_rate"])
    moved_audio = apply_energy_automation(loaded["audio"], loaded["sample_rate"], automation)
    temp_path = Path(output_path).with_suffix(".naturalize-stage.wav")
    write_audio(temp_path, moved_audio, loaded["sample_rate"], bit_depth=limits.export_default_bit_depth)
    report = render_safe_master(temp_path, output_path, limits=limits)
    try:
        temp_path.unlink()
    except OSError:
        pass
    before = analyze_quality(input_path)
    after = report["after"]
    report.update(
        {
            "action": "naturalize",
            "input": str(Path(input_path)),
            "automation": automation,
            "before": before,
            "comparison": compare_metric_dicts(before, after),
        }
    )
    return report
