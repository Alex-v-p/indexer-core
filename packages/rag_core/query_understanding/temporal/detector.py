from __future__ import annotations

import re
from datetime import datetime

from packages.rag_core.query_understanding.temporal.models import (
    DocumentDateConstraint,
    DocumentDateField,
)
from packages.rag_core.query_understanding.temporal.resolver import resolve_date_expression

_UPLOAD_FIELD_PATTERN = re.compile(
    r"\b(uploaded|upload date|ingested|ingestion date|added|imported|indexed)\b",
    re.IGNORECASE,
)
_PUBLICATION_FIELD_PATTERN = re.compile(
    r"\b(published|publication date|released|release date|issued|issue date)\b",
    re.IGNORECASE,
)


class RuleBasedTemporalIntentDetector:
    """Resolve explicit upload/publication-date filters conservatively."""

    name = "rule_based_temporal_detector"

    def __init__(self, *, timezone_name: str = "UTC") -> None:
        self._timezone_name = timezone_name

    def detect(
        self,
        question: str,
        *,
        reference_datetime: datetime | None = None,
    ) -> tuple[DocumentDateConstraint, ...]:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        fields: list[DocumentDateField] = []
        if _UPLOAD_FIELD_PATTERN.search(normalized):
            fields.append(DocumentDateField.UPLOADED_AT)
        if _PUBLICATION_FIELD_PATTERN.search(normalized):
            fields.append(DocumentDateField.PUBLISHED_AT)
        if not fields:
            return ()

        try:
            resolved = resolve_date_expression(
                normalized,
                reference_datetime=reference_datetime,
                timezone_name=self._timezone_name,
            )
        except (ValueError, OverflowError):
            return ()
        if resolved is None:
            return ()

        constraints = []
        for field in dict.fromkeys(fields):
            label = "upload/ingestion" if field is DocumentDateField.UPLOADED_AT else "publication"
            constraints.append(
                DocumentDateConstraint(
                    field=field,
                    date_range=resolved.date_range,
                    original_expression=resolved.expression,
                    confidence=resolved.confidence,
                    rationale=f"The query explicitly constrains the document {label} date.",
                    detector_name=self.name,
                ),
            )
        return tuple(constraints)


def detect_document_date_constraints(
    question: str,
    *,
    reference_datetime: datetime | None = None,
    timezone_name: str = "UTC",
) -> tuple[DocumentDateConstraint, ...]:
    return RuleBasedTemporalIntentDetector(timezone_name=timezone_name).detect(
        question,
        reference_datetime=reference_datetime,
    )
