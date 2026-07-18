from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from packages.rag_core.query_understanding.temporal import DocumentDateField
from packages.rag_core.retrieval.constraint_validation import (
    ConstraintValidationReport,
    describe_constraints,
    matching_date_fields,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints


@dataclass(frozen=True, slots=True)
class EvidenceSourceContext:
    """Small, prompt-safe metadata snapshot for one evidence item."""

    evidence_rank: int
    values: tuple[tuple[str, str], ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "evidence_rank": self.evidence_rank,
            "values": {key: value for key, value in self.values},
        }

    def render(self) -> str:
        return ", ".join(f"{key}={value}" for key, value in self.values)


@dataclass(frozen=True, slots=True)
class EvidenceContextBundle:
    """Compact source metadata and constraint state prepared for grading/generation."""

    constraints: RetrievalConstraints
    validation: ConstraintValidationReport
    sources: tuple[EvidenceSourceContext, ...]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "constraint_summary": describe_constraints(self.constraints),
            "constraints": self.constraints.to_metadata(),
            "validation": self.validation.to_metadata(),
            "sources": [source.to_metadata() for source in self.sources],
        }


def build_evidence_context_bundle(
    evidence: list[EvidenceItem] | tuple[EvidenceItem, ...],
    *,
    constraints: RetrievalConstraints,
    validation: ConstraintValidationReport,
) -> EvidenceContextBundle:
    return EvidenceContextBundle(
        constraints=constraints,
        validation=validation,
        sources=tuple(
            build_evidence_source_context(item, constraints=constraints)
            for item in sorted(evidence, key=lambda candidate: candidate.rank)
        ),
    )


def build_evidence_source_context(
    item: EvidenceItem,
    *,
    constraints: RetrievalConstraints | None = None,
) -> EvidenceSourceContext:
    """Select only metadata that helps identify or temporally/version-ground a source."""

    active = constraints or RetrievalConstraints()
    values: list[tuple[str, str]] = []

    filename = _string_metadata(item, "original_filename")
    title = _string_metadata(item, "document_title")
    if filename:
        values.append(("file", filename))
    elif title:
        values.append(("document", title))

    if active.version.active:
        version_label = _string_metadata(item, "document_version_label")
        version_number = _int_metadata(item, "document_version_number")
        if version_label:
            values.append(("version", version_label))
        elif version_number is not None:
            values.append(("version", f"v{version_number}"))
        latest = item.metadata.get("is_latest_version")
        if isinstance(latest, bool):
            values.append(("latest_version", str(latest).lower()))

    requested_date_fields = {constraint.field for constraint in active.dates}
    matched_date_fields = tuple(
        dict.fromkeys(
            field
            for constraint in active.dates
            for field in matching_date_fields(item, constraint)
        )
    )
    if matched_date_fields:
        values.append(("date_scope_match", "+".join(field.value for field in matched_date_fields)))
    if requested_date_fields:
        if (
            DocumentDateField.UPLOADED_AT in requested_date_fields
            or DocumentDateField.ANY_RECORDED_AT in requested_date_fields
        ):
            uploaded = _date_metadata(item, "uploaded_at", "uploaded_at_epoch")
            if uploaded:
                values.append(("uploaded_at", uploaded))
        if (
            DocumentDateField.PUBLISHED_AT in requested_date_fields
            or DocumentDateField.ANY_RECORDED_AT in requested_date_fields
        ):
            published = _date_metadata(item, "published_at", "published_at_epoch")
            if published:
                values.append(("published_at", published))

    section = _string_metadata(item, "section_title")
    if section:
        values.append(("section", section))
    page_start = _int_metadata(item, "page_number", "source_page_start")
    page_end = _int_metadata(item, "source_page_end")
    if page_start is not None and page_end is not None and page_end != page_start:
        values.append(("pages", f"{page_start}-{page_end}"))
    elif page_start is not None:
        values.append(("page", str(page_start)))

    return EvidenceSourceContext(evidence_rank=item.rank, values=tuple(values))


def format_constraint_context(constraints: RetrievalConstraints) -> str:
    if not constraints.active:
        return "No explicit version or date constraint is active."
    return (
        f"Active metadata constraints: {describe_constraints(constraints)}. "
        "Evidence outside this scope must not be used."
    )


def format_evidence_for_prompt(
    item: EvidenceItem,
    *,
    constraints: RetrievalConstraints | None = None,
    max_chars: int | None = None,
) -> str:
    context = build_evidence_source_context(item, constraints=constraints)
    metadata_line = f"Source metadata: {context.render()}\n" if context.values else ""
    text = item.text.strip()
    if max_chars is not None:
        text = text[:max_chars]
    return f"[{item.rank}]\n{metadata_line}Text: {text}"


def _string_metadata(item: EvidenceItem, key: str) -> str | None:
    value = item.metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int_metadata(item: EvidenceItem, *keys: str) -> int | None:
    for key in keys:
        value = item.metadata.get(key)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _date_metadata(item: EvidenceItem, text_key: str, epoch_key: str) -> str | None:
    text_value = _string_metadata(item, text_key)
    if text_value:
        return text_value
    raw_epoch = item.metadata.get(epoch_key)
    try:
        epoch = float(raw_epoch)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(epoch, tz=UTC).isoformat()
