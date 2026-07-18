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
_GENERIC_DATE_SCOPE_PATTERN = re.compile(
    r"(?:\b(?:data|documents?|sources?|files?|records?|evidence|information|chunks?)\b.{0,48}"
    r"\b(?:from|in|during|between|before|after|since|until|through|dated)\b)"
    r"|(?:\b(?:from|during|between|before|after|since|until|through)\b.{0,48}"
    r"\b(?:data|documents?|sources?|files?|records?|evidence|information|chunks?)\b)",
    re.IGNORECASE,
)
_ONLY_USE_DATE_SCOPE_PATTERN = re.compile(
    r"\b(?:only|exclusively)\b.{0,32}\b(?:use|using|consider|search|retrieve|based)\b"
    r"|\b(?:use|using|consider|search|retrieve|based)\b.{0,32}\b(?:only|exclusively)\b",
    re.IGNORECASE,
)


class RuleBasedTemporalIntentDetector:
    """Resolve explicit and conservatively generic document-date filters."""

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

        fields: list[DocumentDateField] = []
        if _UPLOAD_FIELD_PATTERN.search(normalized):
            fields.append(DocumentDateField.UPLOADED_AT)
        if _PUBLICATION_FIELD_PATTERN.search(normalized):
            fields.append(DocumentDateField.PUBLISHED_AT)
        if not fields and (
            _GENERIC_DATE_SCOPE_PATTERN.search(normalized)
            or _ONLY_USE_DATE_SCOPE_PATTERN.search(normalized)
        ):
            fields.append(DocumentDateField.ANY_RECORDED_AT)
        if not fields:
            return ()

        constraints = []
        for field in dict.fromkeys(fields):
            if field is DocumentDateField.UPLOADED_AT:
                rationale = "The query explicitly constrains the document upload or ingestion date."
                confidence = resolved.confidence
            elif field is DocumentDateField.PUBLISHED_AT:
                rationale = "The query explicitly constrains the document publication date."
                confidence = resolved.confidence
            else:
                rationale = (
                    "The query supplies a date scope for documents/data without naming a date field. "
                    "Retrieval therefore requires either publication or upload metadata to match; "
                    "it does not ignore the date."
                )
                confidence = min(resolved.confidence, 0.84)
            constraints.append(
                DocumentDateConstraint(
                    field=field,
                    date_range=resolved.date_range,
                    original_expression=resolved.expression,
                    confidence=confidence,
                    rationale=rationale,
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
