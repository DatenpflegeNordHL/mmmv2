"""Segment-based energy automation."""

from __future__ import annotations

import numpy as np

from audio_engine.analysis.segments import energy_segments
from audio_engine.naturalize.segment_contrast import segment_gain_for_type


def build_energy_automation(audio: np.ndarray, sample_rate: int) -> list[dict[str, float | str]]:
    """Build conservative automation points from coarse energy segments."""
    automation = []
    for segment in energy_segments(audio, sample_rate):
        gain_db = segment_gain_for_type(str(segment["type"]))
        automation.append({**segment, "gain_db": gain_db})
    return automation


def apply_energy_automation(
    audio: np.ndarray,
    sample_rate: int,
    automation: list[dict[str, float | str]],
) -> np.ndarray:
    """Apply smooth low-depth energy automation."""
    if not automation:
        return audio.copy()
    gain = np.ones(audio.shape[0], dtype=np.float64)
    for item in automation:
        start = max(0, int(float(item["start"]) * sample_rate))
        end = min(audio.shape[0], int(float(item["end"]) * sample_rate))
        if end <= start:
            continue
        scalar = 10.0 ** (float(item["gain_db"]) / 20.0)
        local = np.full(end - start, scalar)
        fade_len = min(local.size // 2, int(0.5 * sample_rate))
        if fade_len > 1:
            fade_in = np.linspace(gain[start], scalar, fade_len)
            fade_out = np.linspace(scalar, gain[end - 1] if end < gain.size else 1.0, fade_len)
            local[:fade_len] = fade_in
            local[-fade_len:] = fade_out
        gain[start:end] = local
    return audio * gain[:, None]
