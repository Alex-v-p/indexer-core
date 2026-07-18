from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class VersionSelectionMode(StrEnum):
    """How retrieval should constrain document versions for one query."""

    ALL = "all"
    LATEST = "latest"
    SPECIFIC = "specific"
    PREVIOUS = "previous"
    LATEST_AND_PREVIOUS = "latest_and_previous"


@dataclass(frozen=True, slots=True)
class DocumentVersionConstraint:
    """Explicit version semantics carried from query understanding into retrieval.

    ``ALL`` deliberately applies no recency preference. Newer chunks only receive
    special treatment when the query explicitly asks for latest/previous/specific
    versions.
    """

    mode: VersionSelectionMode = VersionSelectionMode.ALL
    version_numbers: tuple[int, ...] = ()
    confidence: float = 0.0
    rationale: str = "No version-specific constraint was detected."
    detector_name: str = "none"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if any(number <= 0 for number in self.version_numbers):
            raise ValueError("version_numbers must contain positive integers.")
        if len(self.version_numbers) != len(set(self.version_numbers)):
            raise ValueError("version_numbers must be unique.")
        if self.mode is VersionSelectionMode.SPECIFIC and not self.version_numbers:
            raise ValueError("specific version selection requires version_numbers.")
        if self.mode is not VersionSelectionMode.SPECIFIC and self.version_numbers:
            raise ValueError("version_numbers are only valid for specific version selection.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.detector_name.strip():
            raise ValueError("detector_name must not be empty.")

    @property
    def active(self) -> bool:
        return self.mode is not VersionSelectionMode.ALL

    def to_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "version_numbers": list(self.version_numbers),
            "active": self.active,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "detector_name": self.detector_name,
        }
