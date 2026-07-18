from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from packages.rag_core.documents import VersionSelectionMode
from packages.rag_core.query_understanding.temporal import DocumentDateConstraint, DocumentDateField
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints


class ConstraintValidationStatus(StrEnum):
    """Outcome of checking retrieved evidence against explicit metadata constraints."""

    NOT_REQUESTED = "not_requested"
    MATCHED = "matched"
    NO_MATCH = "no_match"


@dataclass(frozen=True, slots=True)
class ConstraintValidationReport:
    """Traceable proof that explicit metadata constraints were preserved."""

    status: ConstraintValidationStatus
    constraints: RetrievalConstraints
    candidate_count: int
    matched_count: int
    rejected_count: int
    rationale: str

    def __post_init__(self) -> None:
        if min(self.candidate_count, self.matched_count, self.rejected_count) < 0:
            raise ValueError("Constraint-validation counts cannot be negative.")
        if self.matched_count + self.rejected_count != self.candidate_count:
            raise ValueError("matched_count and rejected_count must equal candidate_count.")
        if self.status is ConstraintValidationStatus.NOT_REQUESTED and self.constraints.active:
            raise ValueError("Active constraints cannot produce a not_requested report.")
        if self.status is ConstraintValidationStatus.NO_MATCH and self.matched_count:
            raise ValueError("A no_match report cannot contain matched evidence.")
        if not self.rationale.strip():
            raise ValueError("rationale must not be empty.")

    @property
    def blocked(self) -> bool:
        return self.status is ConstraintValidationStatus.NO_MATCH

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "blocked": self.blocked,
            "candidate_count": self.candidate_count,
            "matched_count": self.matched_count,
            "rejected_count": self.rejected_count,
            "constraints": self.constraints.to_metadata(),
            "rationale": self.rationale,
        }


def validate_evidence_constraints(
    evidence: list[EvidenceItem] | tuple[EvidenceItem, ...],
    constraints: RetrievalConstraints,
) -> tuple[list[EvidenceItem], ConstraintValidationReport]:
    """Strictly filter evidence and report whether explicit constraints matched anything."""

    candidates = list(evidence)
    if not constraints.active:
        return candidates, ConstraintValidationReport(
            status=ConstraintValidationStatus.NOT_REQUESTED,
            constraints=constraints,
            candidate_count=len(candidates),
            matched_count=len(candidates),
            rejected_count=0,
            rationale="No explicit version or date constraint was requested, so relevance ranking was left unchanged.",
        )

    matched = [item for item in candidates if evidence_matches_constraints(item, constraints)]
    rejected_count = len(candidates) - len(matched)
    status = ConstraintValidationStatus.MATCHED if matched else ConstraintValidationStatus.NO_MATCH
    rationale = (
        f"{len(matched)} retrieved evidence item(s) satisfied the explicit metadata constraints; "
        f"{rejected_count} item(s) were excluded."
        if matched
        else (
            "No retrieved evidence satisfied the explicit metadata constraints. Evidence outside the requested "
            "date/version scope must not be used as a fallback."
        )
    )
    return matched, ConstraintValidationReport(
        status=status,
        constraints=constraints,
        candidate_count=len(candidates),
        matched_count=len(matched),
        rejected_count=rejected_count,
        rationale=rationale,
    )


def evidence_matches_constraints(item: EvidenceItem, constraints: RetrievalConstraints) -> bool:
    if not _matches_date_constraints(item, constraints.dates):
        return False

    version = constraints.version
    if not version.active or version.mode is VersionSelectionMode.ALL:
        return True
    if version.mode is VersionSelectionMode.SPECIFIC:
        return _version_number(item) in set(version.version_numbers)

    # Relative version selection is finalized by VersionAwareRetriever. Here we
    # verify the explicit latest flag where it is available, while allowing the
    # wrapper-selected previous/latest-and-previous evidence through.
    relative_version_scope = bool(constraints.dates)
    if version.mode is VersionSelectionMode.LATEST:
        if relative_version_scope:
            return True
        latest = item.metadata.get("is_latest_version")
        return latest is True if isinstance(latest, bool) else True
    if version.mode is VersionSelectionMode.PREVIOUS:
        if relative_version_scope:
            return True
        latest = item.metadata.get("is_latest_version")
        return latest is False if isinstance(latest, bool) else True
    if version.mode is VersionSelectionMode.LATEST_AND_PREVIOUS:
        return True
    return True


def describe_constraints(constraints: RetrievalConstraints) -> str:
    """Return a concise user-facing description of active constraints."""

    parts: list[str] = []
    version = constraints.version
    if version.active:
        if version.mode is VersionSelectionMode.SPECIFIC:
            labels = ", ".join(f"v{number}" for number in version.version_numbers)
            parts.append(f"document version(s) {labels}")
        elif version.mode is VersionSelectionMode.LATEST:
            parts.append("the latest document version")
        elif version.mode is VersionSelectionMode.PREVIOUS:
            parts.append("the previous document version")
        elif version.mode is VersionSelectionMode.LATEST_AND_PREVIOUS:
            parts.append("the latest and previous document versions")

    for constraint in constraints.dates:
        field = {
            DocumentDateField.UPLOADED_AT: "upload date",
            DocumentDateField.PUBLISHED_AT: "publication date",
            DocumentDateField.ANY_RECORDED_AT: "recorded document date",
        }[constraint.field]
        date_range = constraint.date_range
        if date_range.start is not None and date_range.end is not None:
            parts.append(
                f"{field} between {date_range.start.date()} (inclusive) and "
                f"{date_range.end.date()} (exclusive)"
            )
        elif date_range.start is not None:
            parts.append(f"{field} on or after {date_range.start.date()}")
        elif date_range.end is not None:
            parts.append(f"{field} before {date_range.end.date()}")
    return "; ".join(parts) or "no explicit metadata constraints"


def matching_date_fields(
    item: EvidenceItem,
    constraint: DocumentDateConstraint,
) -> tuple[DocumentDateField, ...]:
    """Return the concrete source date fields that satisfy one date constraint."""

    candidate_fields = (
        (constraint.field,)
        if constraint.field is not DocumentDateField.ANY_RECORDED_AT
        else (DocumentDateField.PUBLISHED_AT, DocumentDateField.UPLOADED_AT)
    )
    matched: list[DocumentDateField] = []
    for field in candidate_fields:
        values = _date_values(item, field)
        if any(_value_in_range(value, constraint) for value in values):
            matched.append(field)
    return tuple(matched)


def _matches_date_constraints(
    item: EvidenceItem,
    constraints: tuple[DocumentDateConstraint, ...],
) -> bool:
    for constraint in constraints:
        values = _date_values(item, constraint.field)
        if not values:
            return False
        if not any(_value_in_range(value, constraint) for value in values):
            return False
    return True


def _date_values(item: EvidenceItem, field: DocumentDateField) -> tuple[float, ...]:
    if field is DocumentDateField.UPLOADED_AT:
        keys = ("uploaded_at_epoch",)
    elif field is DocumentDateField.PUBLISHED_AT:
        keys = ("published_at_epoch",)
    else:
        keys = ("published_at_epoch", "uploaded_at_epoch")

    values: list[float] = []
    for key in keys:
        raw_value = item.metadata.get(key)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue
        if value not in values:
            values.append(value)
    return tuple(values)


def _value_in_range(value: float, constraint: DocumentDateConstraint) -> bool:
    start = constraint.date_range.start
    end = constraint.date_range.end
    if start is not None and value < start.timestamp():
        return False
    if end is not None and value >= end.timestamp():
        return False
    return True


def _version_number(item: EvidenceItem) -> int | None:
    value = item.metadata.get("document_version_number")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
