"""
Durable forensic report helpers for sanitizer outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_forensic_report(
    input_file: Path,
    output_file: Path,
    stats: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    metadata_clean: Optional[bool] = None,
    signal_delta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_hash = sha256_file(input_file)
    output_hash = sha256_file(output_file)
    profile = stats.get("processing_profile", {})

    return _json_safe(
        {
            "schema_version": 1,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input": {
                "path": str(input_file),
                "sha256": input_hash,
                "size": input_file.stat().st_size,
            },
            "output": {
                "path": str(output_file),
                "sha256": output_hash,
                "size": output_file.stat().st_size,
                "hash_changed": input_hash != output_hash,
            },
            "processing": {
                "engine": stats.get("processing_engine")
                or profile.get("engine")
                or "unknown",
                "methods_used": stats.get("methods_used")
                or profile.get("methods_used")
                or [],
                "passes_run": stats.get("passes_run")
                or profile.get("passes_run")
                or 1,
                "paranoid_mode": profile.get("paranoid_mode"),
                "stats": stats,
            },
            "verification": {
                "metadata_clean": metadata_clean,
                "signal_delta": signal_delta or _signal_delta_from_stats(stats),
            },
            "detector_findings": analysis,
        }
    )


def write_forensic_report(
    input_file: Path,
    output_file: Path,
    stats: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    metadata_clean: Optional[bool] = None,
    signal_delta: Optional[Dict[str, Any]] = None,
) -> Path:
    report = build_forensic_report(
        input_file=input_file,
        output_file=output_file,
        stats=stats,
        analysis=analysis,
        metadata_clean=metadata_clean,
        signal_delta=signal_delta,
    )
    report_path = output_file.with_suffix(output_file.suffix + ".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def signal_delta_metrics(original: Any, processed: Any) -> Dict[str, Any]:
    """Compute lightweight signal-difference metrics for report evidence."""
    import numpy as np

    original_arr = np.asarray(original, dtype=np.float32)
    processed_arr = np.asarray(processed, dtype=np.float32)
    if original_arr.ndim == 1:
        original_arr = original_arr[:, None]
    if processed_arr.ndim == 1:
        processed_arr = processed_arr[:, None]

    frames = min(original_arr.shape[0], processed_arr.shape[0])
    channels = min(original_arr.shape[1], processed_arr.shape[1])
    if frames == 0 or channels == 0:
        return {
            "signal_changed": False,
            "signal_delta_rms": 0.0,
            "signal_delta_ratio": 0.0,
            "source_rms": 0.0,
        }

    original_view = original_arr[:frames, :channels]
    processed_view = processed_arr[:frames, :channels]
    delta = processed_view - original_view
    source_rms = float(np.sqrt(np.mean(original_view * original_view)))
    delta_rms = float(np.sqrt(np.mean(delta * delta)))
    delta_ratio = delta_rms / max(source_rms, 1e-12)
    return {
        "signal_changed": bool(delta_rms > 0),
        "signal_delta_rms": delta_rms,
        "signal_delta_ratio": delta_ratio,
        "source_rms": source_rms,
        "signal_max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
    }


def _signal_delta_from_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    keys = (
        "signal_changed",
        "signal_delta_required",
        "signal_delta_rms",
        "signal_delta_ratio",
        "signal_delta_db",
        "signal_max_abs_delta",
        "source_rms",
    )
    return {key: stats[key] for key in keys if key in stats}


def _json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a runtime dependency
        np = None

    if np is not None:
        if isinstance(value, np.generic):
            return _json_safe(value.item())
        if isinstance(value, np.ndarray):
            return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value
