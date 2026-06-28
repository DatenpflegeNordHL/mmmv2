"""Release-readiness analysis and transparent scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from audio_engine.analysis.dynamics import dynamics_metrics
from audio_engine.analysis.loudness import loudness_metrics
from audio_engine.analysis.segments import energy_segments
from audio_engine.analysis.spectrum import spectrum_metrics
from audio_engine.analysis.stereo import stereo_metrics
from audio_engine.analysis.transients import transient_metrics
from audio_engine.io.loader import load_audio


def analyze_quality(path: str | Path) -> dict[str, Any]:
    """Analyze a local WAV/FLAC/AIFF file for release-readiness."""
    loaded = load_audio(path)
    audio = loaded["audio"]
    sample_rate = loaded["sample_rate"]
    metrics: dict[str, Any] = {
        "input_path": str(loaded["path"]),
        "duration_seconds": loaded["duration_seconds"],
        "sample_rate": sample_rate,
        "channels": loaded["channels"],
        "format": loaded["format"],
        "subtype": loaded["subtype"],
    }
    metrics.update(loudness_metrics(audio, sample_rate))
    metrics.update(dynamics_metrics(audio))
    metrics.update(stereo_metrics(audio, sample_rate))
    metrics.update(spectrum_metrics(audio, sample_rate))
    metrics.update(transient_metrics(audio, sample_rate))
    metrics["segments"] = energy_segments(audio, sample_rate)
    metrics["release_readiness"] = score_release_readiness(metrics)
    return metrics


def score_release_readiness(metrics: dict[str, Any]) -> dict[str, Any]:
    """Build an explainable 0-100 release-readiness score."""
    checks = [
        _score_loudness(metrics),
        _score_true_peak(metrics),
        _score_clipping(metrics),
        _score_stereo(metrics),
        _score_low_end(metrics),
        _score_harshness(metrics),
        _score_dynamic_contrast(metrics),
        _score_export(metrics),
    ]
    total_weight = sum(check["weight"] for check in checks)
    score = sum(check["score"] * check["weight"] for check in checks) / total_weight
    recommendations = [
        recommendation
        for check in checks
        for recommendation in check.get("recommendations", [])
    ]
    return {
        "score": float(round(score, 1)),
        "grade": _grade(score),
        "checks": checks,
        "recommendations": recommendations,
    }


def _check(name: str, score: float, weight: float, value: Any, recommendations: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "score": float(max(0.0, min(100.0, score))),
        "weight": float(weight),
        "value": value,
        "recommendations": recommendations,
    }


def _score_loudness(metrics: dict[str, Any]) -> dict[str, Any]:
    lufs = float(metrics["integrated_lufs"])
    distance = abs(lufs - (-14.0))
    score = 100.0 - min(60.0, distance * 8.0)
    recs = []
    if lufs > -8.0:
        recs.append("Track is very loud; consider lower input gain and lighter limiting.")
    elif lufs < -20.0:
        recs.append("Track is quiet for streaming; consider conservative gain staging.")
    return _check("loudness_suitability", score, 1.2, lufs, recs)


def _score_true_peak(metrics: dict[str, Any]) -> dict[str, Any]:
    true_peak = float(metrics["estimated_true_peak_dbtp"])
    score = 100.0 if true_peak <= -1.0 else max(20.0, 100.0 - (true_peak + 1.0) * 35.0)
    recs = ["Lower limiter ceiling to protect codec conversion."] if true_peak > -1.0 else []
    return _check("true_peak_safety", score, 1.2, true_peak, recs)


def _score_clipping(metrics: dict[str, Any]) -> dict[str, Any]:
    count = int(metrics["clipping_sample_count"])
    score = 100.0 if count == 0 else max(0.0, 100.0 - np.log10(count + 1) * 35.0)
    recs = ["Clipping samples detected; reduce input gain or use repair_light settings."] if count else []
    return _check("clipping_safety", score, 1.1, count, recs)


def _score_stereo(metrics: dict[str, Any]) -> dict[str, Any]:
    corr = metrics["stereo_correlation"]
    if corr is None:
        return _check("stereo_mono_compatibility", 85.0, 0.9, None, ["Mono input detected."])
    corr_float = float(corr)
    score = 100.0 if corr_float >= 0.1 else max(30.0, 100.0 + corr_float * 160.0)
    recs = ["Stereo correlation is low; check mono compatibility."] if corr_float < 0.1 else []
    return _check("stereo_mono_compatibility", score, 0.9, corr_float, recs)


def _score_low_end(metrics: dict[str, Any]) -> dict[str, Any]:
    width = float(metrics["low_end_width"])
    score = 100.0 if width <= 0.20 else max(35.0, 100.0 - (width - 0.20) * 160.0)
    recs = ["Low-end side energy is high; mono bass below 80-120 Hz is recommended."] if width > 0.20 else []
    return _check("low_end_stability", score, 1.0, width, recs)


def _score_harshness(metrics: dict[str, Any]) -> dict[str, Any]:
    harsh = float(metrics["band_energy"].get("harsh_5000_9000_hz", 0.0))
    score = 100.0 if harsh <= 0.18 else max(40.0, 100.0 - (harsh - 0.18) * 220.0)
    recs = ["Harshness band energy is elevated; apply gentle dynamic deharshing."] if harsh > 0.18 else []
    return _check("harshness_risk", score, 0.9, harsh, recs)


def _score_dynamic_contrast(metrics: dict[str, Any]) -> dict[str, Any]:
    plr = float(metrics["peak_to_loudness_ratio"])
    score = 100.0 if 8.0 <= plr <= 18.0 else max(45.0, 100.0 - min(abs(plr - 10.0), 10.0) * 4.5)
    recs = ["Dynamic contrast is low; use lighter compression/limiting."] if plr < 7.0 else []
    return _check("dynamic_contrast", score, 0.8, plr, recs)


def _score_export(metrics: dict[str, Any]) -> dict[str, Any]:
    channels = int(metrics["channels"])
    sample_rate = int(metrics["sample_rate"])
    score = 100.0
    recs = []
    if channels not in (1, 2):
        score -= 20.0
        recs.append("Unexpected channel count for stereo release workflow.")
    if sample_rate < 44100:
        score -= 15.0
        recs.append("Sample rate is below 44.1 kHz.")
    return _check("export_readiness", score, 0.7, {"channels": channels, "sample_rate": sample_rate}, recs)


def _grade(score: float) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "needs_review"
    return "repair_recommended"
