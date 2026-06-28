"""Coarse energy segmentation for naturalize automation."""

from __future__ import annotations

import numpy as np


def energy_segments(audio: np.ndarray, sample_rate: int) -> list[dict[str, float | str]]:
    """Classify coarse windows as quiet, balanced, or dense."""
    mono = np.mean(audio, axis=1)
    window = max(1, int(8.0 * sample_rate))
    hop = max(1, int(4.0 * sample_rate))
    if mono.size < window:
        rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
        return [{"start": 0.0, "end": float(mono.size / sample_rate), "type": "balanced", "rms": rms}]

    rms_values = []
    ranges = []
    for start in range(0, mono.size - window + 1, hop):
        end = start + window
        rms_values.append(float(np.sqrt(np.mean(mono[start:end] ** 2))))
        ranges.append((start, end))
    low = float(np.percentile(rms_values, 33))
    high = float(np.percentile(rms_values, 67))
    segments = []
    for (start, end), rms in zip(ranges, rms_values):
        segment_type = "quiet" if rms <= low else "dense" if rms >= high else "balanced"
        segments.append(
            {
                "start": float(start / sample_rate),
                "end": float(end / sample_rate),
                "type": segment_type,
                "rms": rms,
            }
        )
    return segments
