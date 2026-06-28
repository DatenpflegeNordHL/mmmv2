"""Safe master rendering pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audio_engine.analysis.readiness import analyze_quality
from audio_engine.dsp.compressor import light_bus_compressor
from audio_engine.dsp.dynamic_eq import suggested_cuts_from_metrics
from audio_engine.dsp.eq import fft_band_eq
from audio_engine.dsp.filters import highpass
from audio_engine.dsp.gain import apply_gain_db
from audio_engine.dsp.limiter import peak_limiter
from audio_engine.dsp.midside import mono_bass
from audio_engine.dsp.saturation import subtle_tanh_saturation
from audio_engine.guardrails.limits import DEFAULT_LIMITS, GuardrailLimits
from audio_engine.guardrails.validators import ensure_finite_audio, validate_render
from audio_engine.io.loader import load_audio
from audio_engine.io.writer import write_audio
from audio_engine.reports.before_after import compare_metric_dicts


def render_safe_master(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limits: GuardrailLimits = DEFAULT_LIMITS,
    mono_bass_hz: float = 100.0,
) -> dict[str, Any]:
    """Render a conservative safe master and return an explainable report."""
    loaded = load_audio(input_path)
    before = analyze_quality(input_path)
    audio = loaded["audio"].copy()
    steps: list[dict[str, Any]] = []

    target_lufs = -14.0
    current_lufs = float(before["integrated_lufs"])
    input_gain_db = float(np.clip(target_lufs - current_lufs, -3.0, 3.0))
    if abs(input_gain_db) > 0.05:
        audio = apply_gain_db(audio, input_gain_db)
    steps.append({"name": "input_gain_staging", "gain_db": input_gain_db})

    audio = highpass(audio, loaded["sample_rate"], cutoff_hz=25.0)
    steps.append({"name": "highpass", "cutoff_hz": 25.0})

    for cut in suggested_cuts_from_metrics(before):
        audio = fft_band_eq(
            audio,
            loaded["sample_rate"],
            float(cut["low_hz"]),
            float(cut["high_hz"]),
            float(cut["gain_db"]),
            max_gain_db=limits.max_eq_gain_db,
        )
        steps.append(cut)

    if float(before["low_end_width"]) > 0.18:
        audio = mono_bass(audio, loaded["sample_rate"], cutoff_hz=mono_bass_hz)
        steps.append({"name": "mono_bass", "cutoff_hz": mono_bass_hz})
    else:
        steps.append({"name": "mono_bass", "applied": False, "reason": "low_end_stable"})

    audio, compression_stats = light_bus_compressor(
        audio,
        loaded["sample_rate"],
        max_gain_reduction_db=limits.max_compressor_gain_reduction_db,
    )
    steps.append({"name": "light_bus_compressor", **compression_stats})

    audio = subtle_tanh_saturation(audio)
    steps.append({"name": "subtle_tanh_saturation", "drive": 1.015, "mix": 0.12})

    audio, limiter_stats = peak_limiter(audio, ceiling_dbtp=limits.limiter_ceiling_dbtp)
    steps.append({"name": "final_peak_limiter", **limiter_stats})
    audio = ensure_finite_audio(audio)

    output = write_audio(
        output_path,
        audio,
        loaded["sample_rate"],
        bit_depth=limits.export_default_bit_depth,
    )
    after = analyze_quality(output)
    guardrails = validate_render(before, after, limits)
    comparison = compare_metric_dicts(before, after)
    return {
        "action": "safe_master",
        "input": str(Path(input_path)),
        "output": str(output),
        "guardrails": {"limits": limits.to_dict(), **guardrails},
        "processing_steps": steps,
        "before": before,
        "after": after,
        "comparison": comparison,
    }
