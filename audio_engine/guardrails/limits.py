"""Conservative processing limits for release-safe rendering."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GuardrailLimits:
    max_eq_gain_db: float = 2.0
    max_dynamic_eq_gain_reduction_db: float = 3.0
    max_compressor_gain_reduction_db: float = 1.5
    max_limiter_gain_reduction_db: float = 3.0
    max_width_change_percent: float = 10.0
    limiter_ceiling_dbtp: float = -1.5
    preserve_sample_rate: bool = True
    export_default_bit_depth: int = 24

    def to_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


DEFAULT_LIMITS = GuardrailLimits()


def clamp_db(value: float, limit: float) -> float:
    """Clamp positive or negative gain to a symmetric dB limit."""
    return max(-abs(limit), min(abs(limit), float(value)))
