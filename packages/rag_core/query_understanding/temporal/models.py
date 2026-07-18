from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class DocumentDateField(StrEnum):
    """Structured date metadata that can constrain retrieval."""

    UPLOADED_AT = "uploaded_at"
    PUBLISHED_AT = "published_at"
    ANY_RECORDED_AT = "any_recorded_at"


@dataclass(frozen=True, slots=True)
class DateRange:
    """Half-open UTC datetime interval: ``start <= value < end``."""

    start: datetime | None = None
    end: datetime | None = None

    def __post_init__(self) -> None:
        for value, name in ((self.start, "start"), (self.end, "end")):
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware.")
        if self.start is None and self.end is None:
            raise ValueError("At least one date-range boundary is required.")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("Date-range start must be earlier than end.")

    def to_metadata(self) -> dict[str, str | None]:
        return {
            "start": self.start.isoformat() if self.start is not None else None,
            "end": self.end.isoformat() if self.end is not None else None,
            "end_exclusive": True,
        }


@dataclass(frozen=True, slots=True)
class DocumentDateConstraint:
    """Resolved temporal intent carried from classification into retrieval."""

    field: DocumentDateField
    date_range: DateRange
    original_expression: str
    confidence: float = 1.0
    rationale: str = ""
    detector_name: str = "rule_based_temporal_detector"

    def __post_init__(self) -> None:
        if not self.original_expression.strip():
            raise ValueError("original_expression must not be empty.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")
        if not self.detector_name.strip():
            raise ValueError("detector_name must not be empty.")

    @property
    def active(self) -> bool:
        return True

    def to_metadata(self) -> dict[str, Any]:
        return {
            "field": self.field.value,
            "range": self.date_range.to_metadata(),
            "original_expression": self.original_expression,
            "active": True,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "detector_name": self.detector_name,
        }
