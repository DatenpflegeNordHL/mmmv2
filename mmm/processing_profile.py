"""
Shared processing profile metadata for all sanitizer engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class ProcessingProfile:
    """Describe which sanitizer engine ran and how aggressive it was."""

    engine: str
    paranoid_mode: bool
    methods_used: List[str] = field(default_factory=list)
    passes_run: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "paranoid_mode": self.paranoid_mode,
            "methods_used": list(self.methods_used),
            "passes_run": self.passes_run,
        }


def build_processing_profile(
    engine: str,
    paranoid_mode: bool,
    methods_used: List[str],
    passes_run: int = 1,
) -> ProcessingProfile:
    return ProcessingProfile(
        engine=engine,
        paranoid_mode=paranoid_mode,
        methods_used=list(dict.fromkeys(methods_used)),
        passes_run=passes_run,
    )
