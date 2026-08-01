from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from packages.rag_core.subjects.models import SubjectKind
from packages.rag_core.subjects.naming import normalize_subject_name


class CoverageMode(StrEnum):
    BEST_EVIDENCE = "best_evidence"
    MULTI_DOCUMENT = "multi_document"


class QueryProductOutcome(StrEnum):
    ANSWERED = "answered"
    CLARIFICATION_REQUIRED = "clarification_required"
    NO_EVIDENCE = "no_evidence"


class ScopeResolutionSource(StrEnum):
    GLOBAL = "global"
    EXPLICIT = "explicit"
    INFERRED_PROJECT = "inferred_project"
    EXPLICIT_AND_INFERRED = "explicit_and_inferred"


class SubjectScopeSnapshotValidationError(ValueError):
    """A persisted subject-scope snapshot is unsafe to reuse."""


@dataclass(frozen=True, slots=True)
class DocumentScope:
    """Immutable hard document boundary; global and strict-empty are distinct."""

    strict: bool = False
    allowed_document_ids: tuple[uuid.UUID, ...] = ()

    def __post_init__(self) -> None:
        unique = tuple(dict.fromkeys(self.allowed_document_ids))
        if unique != self.allowed_document_ids:
            raise ValueError("allowed_document_ids must be unique and stable.")
        if not self.strict and self.allowed_document_ids:
            raise ValueError("A global document scope cannot contain document ids.")

    @classmethod
    def global_scope(cls) -> DocumentScope:
        return cls()

    @classmethod
    def strict_scope(cls, document_ids: tuple[uuid.UUID, ...]) -> DocumentScope:
        return cls(strict=True, allowed_document_ids=tuple(dict.fromkeys(document_ids)))

    @property
    def is_global(self) -> bool:
        return not self.strict

    @property
    def is_strict_empty(self) -> bool:
        return self.strict and not self.allowed_document_ids

    def allows(self, document_id: uuid.UUID | None) -> bool:
        return self.is_global or (
            document_id is not None and document_id in self.allowed_document_ids
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "strict": self.strict,
            "global": self.is_global,
            "strict_empty": self.is_strict_empty,
            "allowed_document_ids": [str(item) for item in self.allowed_document_ids],
            "allowed_document_count": len(self.allowed_document_ids),
        }

    @classmethod
    def from_metadata(cls, value: object) -> DocumentScope:
        if not isinstance(value, dict) or value.get("strict") is not True:
            return cls.global_scope()
        raw_ids = value.get("allowed_document_ids")
        ids = (
            tuple(uuid.UUID(str(item)) for item in raw_ids)
            if isinstance(raw_ids, list)
            else ()
        )
        return cls.strict_scope(ids)


@dataclass(frozen=True, slots=True)
class SubjectDocumentLane:
    """One stable project-specific retrieval lane and its hard document boundary."""

    subject_id: uuid.UUID
    subject_name: str
    document_scope: DocumentScope

    def __post_init__(self) -> None:
        if not self.subject_name.strip():
            raise ValueError("subject_name must not be empty.")
        if not self.document_scope.strict:
            raise ValueError("A subject document lane must have a strict document scope.")

    @property
    def lane_id(self) -> str:
        return f"subject:{self.subject_id}"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "subject_id": str(self.subject_id),
            "subject_name": self.subject_name,
            "document_scope": self.document_scope.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class SubjectScopeCatalogEntry:
    subject_id: uuid.UUID
    kind: SubjectKind
    name: str
    aliases: tuple[str, ...] = ()

    @property
    def normalized_names(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                normalized
                for label in (self.name, *self.aliases)
                if (normalized := normalize_subject_name(label))
            ),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "subject_id": str(self.subject_id),
            "kind": self.kind.value,
            "name": self.name,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class SubjectScopeResolution:
    hard_subject_ids: tuple[uuid.UUID, ...]
    matched_subjects: tuple[SubjectScopeCatalogEntry, ...]
    source: ScopeResolutionSource
    confidence: float
    clarification_reason: str | None = None
    ambiguity_candidates: tuple[SubjectScopeCatalogEntry, ...] = ()
    lane_subjects: tuple[SubjectScopeCatalogEntry, ...] = ()
    comparison_requested: bool = False

    @property
    def clarification_required(self) -> bool:
        return self.clarification_reason is not None

    @property
    def strict(self) -> bool:
        return bool(self.hard_subject_ids) or self.clarification_required


@dataclass(frozen=True, slots=True)
class ResolvedSubjectScopeSnapshot:
    requested_subject_ids: tuple[uuid.UUID, ...]
    catalog: tuple[SubjectScopeCatalogEntry, ...]
    matched_subject_ids: tuple[uuid.UUID, ...]
    document_scope: DocumentScope
    source: ScopeResolutionSource
    confidence: float
    catalog_revision: str
    policy_revision: str
    coverage_mode: CoverageMode = CoverageMode.BEST_EVIDENCE
    product_outcome: QueryProductOutcome | None = None
    clarification_reason: str | None = None
    ambiguity_candidates: tuple[SubjectScopeCatalogEntry, ...] = ()
    subject_lanes: tuple[SubjectDocumentLane, ...] = ()
    comparison_requested: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requested_subject_ids": [str(item) for item in self.requested_subject_ids],
            "catalog": [item.to_metadata() for item in self.catalog],
            "matched_subject_ids": [str(item) for item in self.matched_subject_ids],
            "document_scope": self.document_scope.to_metadata(),
            "source": self.source.value,
            "confidence": self.confidence,
            "strict": self.document_scope.strict,
            "catalog_revision": self.catalog_revision,
            "policy_revision": self.policy_revision,
            "coverage_mode": self.coverage_mode.value,
            "product_outcome": (
                self.product_outcome.value if self.product_outcome is not None else None
            ),
            "clarification_reason": self.clarification_reason,
            "ambiguity_candidates": [
                item.to_metadata() for item in self.ambiguity_candidates
            ],
            "subject_lanes": [item.to_metadata() for item in self.subject_lanes],
            "comparison_requested": self.comparison_requested,
        }

    @classmethod
    def from_metadata(cls, value: object) -> ResolvedSubjectScopeSnapshot:
        try:
            return _snapshot_from_metadata(value)
        except SubjectScopeSnapshotValidationError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SubjectScopeSnapshotValidationError(
                "Resolved subject scope snapshot is malformed."
            ) from exc


def resolve_subject_scope(
    *,
    question: str,
    requested_subject_ids: tuple[uuid.UUID, ...],
    catalog: tuple[SubjectScopeCatalogEntry, ...],
    max_project_lanes: int = 4,
) -> SubjectScopeResolution:
    """Resolve explicit hard filters and conservative project-name inference."""

    if max_project_lanes <= 0:
        raise ValueError("max_project_lanes must be positive.")
    by_id = {item.subject_id: item for item in catalog}
    missing = tuple(item for item in requested_subject_ids if item not in by_id)
    if missing:
        return SubjectScopeResolution(
            hard_subject_ids=(),
            matched_subjects=(),
            source=ScopeResolutionSource.EXPLICIT,
            confidence=1.0,
            clarification_reason="unknown_explicit_subject",
        )

    explicit = tuple(by_id[item] for item in dict.fromkeys(requested_subject_ids))
    normalized_question = normalize_subject_name(question)
    project_matches: dict[uuid.UUID, SubjectScopeCatalogEntry] = {}
    ambiguous: dict[uuid.UUID, SubjectScopeCatalogEntry] = {}
    names_to_projects: dict[str, list[SubjectScopeCatalogEntry]] = {}
    for item in catalog:
        if item.kind is not SubjectKind.PROJECT:
            continue
        for name in item.normalized_names:
            names_to_projects.setdefault(name, []).append(item)
    padded_question = f" {normalized_question} "
    for normalized_name, projects in names_to_projects.items():
        if f" {normalized_name} " not in padded_question:
            continue
        if len(projects) > 1:
            ambiguous.update((item.subject_id, item) for item in projects)
        else:
            project_matches[projects[0].subject_id] = projects[0]
    if ambiguous:
        return SubjectScopeResolution(
            hard_subject_ids=tuple(item.subject_id for item in explicit),
            matched_subjects=explicit,
            source=(
                ScopeResolutionSource.EXPLICIT_AND_INFERRED
                if explicit
                else ScopeResolutionSource.INFERRED_PROJECT
            ),
            confidence=0.5,
            clarification_reason="ambiguous_project_name",
            ambiguity_candidates=tuple(
                sorted(ambiguous.values(), key=lambda item: (item.name.casefold(), str(item.subject_id)))
            ),
        )

    inferred = tuple(project_matches.values())
    if not inferred and _contains_unknown_named_project(question, names_to_projects):
        return SubjectScopeResolution(
            hard_subject_ids=tuple(item.subject_id for item in explicit),
            matched_subjects=explicit,
            source=(
                ScopeResolutionSource.EXPLICIT_AND_INFERRED
                if explicit
                else ScopeResolutionSource.INFERRED_PROJECT
            ),
            confidence=0.45,
            clarification_reason="unknown_named_project",
        )

    lane_subjects = tuple(
        dict.fromkeys(
            item
            for item in (*explicit, *inferred)
            if item.kind is SubjectKind.PROJECT
        )
    )
    comparison_requested = len(lane_subjects) > 1 and (
        sum(item.kind is SubjectKind.PROJECT for item in explicit) > 1
        or _has_project_comparison_cue(question, lane_subjects)
    )
    if len(lane_subjects) > max_project_lanes:
        return SubjectScopeResolution(
            hard_subject_ids=tuple(item.subject_id for item in explicit),
            matched_subjects=explicit,
            source=(
                ScopeResolutionSource.EXPLICIT_AND_INFERRED
                if explicit and inferred
                else ScopeResolutionSource.EXPLICIT
                if explicit
                else ScopeResolutionSource.INFERRED_PROJECT
            ),
            confidence=1.0,
            clarification_reason="comparison_project_limit_exceeded",
            lane_subjects=lane_subjects[:max_project_lanes],
            comparison_requested=True,
        )
    if len(lane_subjects) > 1 and not comparison_requested:
        return SubjectScopeResolution(
            hard_subject_ids=tuple(item.subject_id for item in explicit),
            matched_subjects=explicit,
            source=(
                ScopeResolutionSource.EXPLICIT_AND_INFERRED
                if explicit and inferred
                else ScopeResolutionSource.EXPLICIT
                if explicit
                else ScopeResolutionSource.INFERRED_PROJECT
            ),
            confidence=0.6,
            clarification_reason="multiple_projects_require_comparison_intent",
            lane_subjects=lane_subjects,
        )

    matched = tuple(dict.fromkeys((*explicit, *inferred)))
    hard_ids = tuple(item.subject_id for item in matched)
    if explicit and inferred:
        source = ScopeResolutionSource.EXPLICIT_AND_INFERRED
    elif explicit:
        source = ScopeResolutionSource.EXPLICIT
    elif inferred:
        source = ScopeResolutionSource.INFERRED_PROJECT
    else:
        source = ScopeResolutionSource.GLOBAL
    return SubjectScopeResolution(
        hard_subject_ids=hard_ids,
        matched_subjects=matched,
        source=source,
        confidence=1.0 if explicit else 0.9 if inferred else 1.0,
        lane_subjects=lane_subjects,
        comparison_requested=comparison_requested,
    )


def _has_project_comparison_cue(
    question: str,
    projects: tuple[SubjectScopeCatalogEntry, ...],
) -> bool:
    normalized = normalize_subject_name(question)
    if re.search(
        r"\b(compare|comparison|versus|vs|difference|differences|differ|between)\b",
        normalized,
    ):
        return True
    if " and " not in f" {normalized} ":
        return False
    mentioned = [
        name
        for project in projects
        for name in project.normalized_names
        if f" {name} " in f" {normalized} "
    ]
    return len(set(mentioned)) >= 2


def _contains_unknown_named_project(
    question: str,
    known_names: dict[str, list[SubjectScopeCatalogEntry]],
) -> bool:
    known = set(known_names)
    return any(
        (normalized := normalize_subject_name(candidate))
        and normalized not in known
        and (deliberately_named or _looks_like_project_identifier(candidate))
        for candidate, deliberately_named in _named_project_candidates(question)
    )


def _named_project_candidates(question: str) -> tuple[tuple[str, bool], ...]:
    """Extract conservative prefix/postfix project names from natural language."""

    candidates: list[tuple[str, bool]] = []
    patterns = (
        # Explicit naming language and quoted names are deliberate even when
        # the label itself resembles an ordinary word.
        (
            r"\b[Pp]roject\s+(?:called|named)\s+"
            r"(?:[\"']([^\"']{2,80})[\"']|([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2}))",
            True,
        ),
        (r"\b[Pp]roject\s+[\"']([^\"']{2,80})[\"']", True),
        (r"[\"']([^\"']{2,80})[\"']\s+[Pp]roject\b", True),
        # Unquoted prefix/postfix forms must also look like a genuine title,
        # acronym, or code rather than generic project vocabulary.
        (
            r"\b[Pp]roject\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})",
            False,
        ),
        (
            r"\b([A-Z][\w-]*(?:\s+[A-Z][\w-]*){0,2})\s+[Pp]roject\b",
            False,
        ),
    )
    for pattern, deliberately_named in patterns:
        for match in re.findall(pattern, question):
            parts = match if isinstance(match, tuple) else (match,)
            candidate = next((part for part in parts if part), "")
            if candidate:
                candidates.append((candidate, deliberately_named))
    return tuple(dict.fromkeys(candidates))


_GENERIC_PROJECT_LABELS = frozenset(
    {
        "documentation",
        "guidance",
        "information",
        "management",
        "overview",
        "plan",
        "planning",
        "roadmap",
        "schedule",
        "software",
        "status",
        "timeline",
        "update",
        "updates",
        "work",
    }
)


def _looks_like_project_identifier(candidate: str) -> bool:
    normalized = normalize_subject_name(candidate)
    tokens = normalized.split()
    if not tokens or any(token in _GENERIC_PROJECT_LABELS for token in tokens):
        return False
    compact = candidate.strip().replace(" ", "")
    if re.fullmatch(r"[A-Z]{2,12}", compact):
        return True
    if re.fullmatch(r"[A-Z][A-Za-z]*[-_]?[A-Z0-9][A-Za-z0-9_-]*", compact):
        return True
    words = candidate.strip().split()
    return bool(words) and all(word[0].isupper() for word in words)


def _catalog_entry_from_metadata(value: object) -> SubjectScopeCatalogEntry:
    if not isinstance(value, dict):
        raise ValueError("Subject scope catalog entries must be objects.")
    return SubjectScopeCatalogEntry(
        subject_id=uuid.UUID(str(value["subject_id"])),
        kind=SubjectKind(str(value["kind"])),
        name=str(value["name"]),
        aliases=tuple(str(item) for item in value.get("aliases", [])),
    )


def _subject_lane_from_metadata(value: object) -> SubjectDocumentLane:
    if not isinstance(value, dict):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot subject lane entries must be objects."
        )
    subject_id = uuid.UUID(str(value["subject_id"]))
    lane = SubjectDocumentLane(
        subject_id=subject_id,
        subject_name=str(value["subject_name"]),
        document_scope=_document_scope_from_snapshot_metadata(value["document_scope"]),
    )
    if value.get("lane_id") != lane.lane_id:
        raise SubjectScopeSnapshotValidationError(
            "Snapshot subject lane id does not match its subject id."
        )
    return lane


def _snapshot_from_metadata(value: object) -> ResolvedSubjectScopeSnapshot:
    if not isinstance(value, dict):
        raise SubjectScopeSnapshotValidationError(
            "Resolved subject scope snapshot must be an object."
        )
    required = {
        "requested_subject_ids",
        "catalog",
        "matched_subject_ids",
        "document_scope",
        "source",
        "confidence",
        "strict",
        "catalog_revision",
        "policy_revision",
        "coverage_mode",
        "product_outcome",
        "clarification_reason",
        "ambiguity_candidates",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise SubjectScopeSnapshotValidationError(
            "Resolved subject scope snapshot is incomplete: " + ", ".join(missing)
        )

    requested_ids = _uuid_list(value["requested_subject_ids"], "requested_subject_ids")
    catalog = tuple(
        _catalog_entry_from_metadata(item)
        for item in _object_list(value["catalog"], "catalog")
    )
    matched_ids = _uuid_list(value["matched_subject_ids"], "matched_subject_ids")
    candidates = tuple(
        _catalog_entry_from_metadata(item)
        for item in _object_list(value["ambiguity_candidates"], "ambiguity_candidates")
    )
    raw_lanes = value.get("subject_lanes")
    subject_lanes = (
        tuple(
            _subject_lane_from_metadata(item)
            for item in _object_list(raw_lanes, "subject_lanes")
        )
        if raw_lanes is not None
        else ()
    )
    comparison_requested = value.get("comparison_requested", False)
    if not isinstance(comparison_requested, bool):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot comparison_requested must be boolean."
        )
    source = ScopeResolutionSource(str(value["source"]))
    document_scope = _document_scope_from_snapshot_metadata(value["document_scope"])
    if not isinstance(value["strict"], bool) or value["strict"] != document_scope.strict:
        raise SubjectScopeSnapshotValidationError(
            "Snapshot strictness does not match its document scope."
        )

    strict_source = source in {
        ScopeResolutionSource.EXPLICIT,
        ScopeResolutionSource.INFERRED_PROJECT,
        ScopeResolutionSource.EXPLICIT_AND_INFERRED,
    }
    method = value.get("method")
    if method is not None and not isinstance(method, str):
        raise SubjectScopeSnapshotValidationError("Snapshot method must be text.")
    strict_method = isinstance(method, str) and any(
        marker in method.casefold() for marker in ("explicit", "inferred", "project")
    )
    if (strict_source or strict_method) and not document_scope.strict:
        raise SubjectScopeSnapshotValidationError(
            "An explicit or inferred subject selection requires a strict document scope."
        )
    if source is ScopeResolutionSource.GLOBAL and document_scope.strict:
        raise SubjectScopeSnapshotValidationError(
            "A global subject selection cannot contain a strict document scope."
        )

    confidence = float(value["confidence"])
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise SubjectScopeSnapshotValidationError(
            "Snapshot confidence must be between zero and one."
        )
    catalog_revision = _non_empty_text(value["catalog_revision"], "catalog_revision")
    policy_revision = _non_empty_text(value["policy_revision"], "policy_revision")
    coverage_mode = CoverageMode(str(value["coverage_mode"]))
    outcome_value = value["product_outcome"]
    outcome = (
        QueryProductOutcome(str(outcome_value)) if outcome_value is not None else None
    )
    clarification = value["clarification_reason"]
    if clarification is not None and not isinstance(clarification, str):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot clarification_reason must be text or null."
        )
    _validate_snapshot_outcome(
        outcome=outcome,
        clarification_reason=clarification,
        document_scope=document_scope,
    )
    _validate_snapshot_subject_relationships(
        requested_ids=requested_ids,
        catalog=catalog,
        matched_ids=matched_ids,
        candidates=candidates,
        source=source,
        outcome=outcome,
        clarification_reason=clarification,
        document_scope=document_scope,
        subject_lanes=subject_lanes,
        comparison_requested=comparison_requested,
    )

    return ResolvedSubjectScopeSnapshot(
        requested_subject_ids=requested_ids,
        catalog=catalog,
        matched_subject_ids=matched_ids,
        document_scope=document_scope,
        source=source,
        confidence=confidence,
        catalog_revision=catalog_revision,
        policy_revision=policy_revision,
        coverage_mode=coverage_mode,
        product_outcome=outcome,
        clarification_reason=clarification,
        ambiguity_candidates=candidates,
        subject_lanes=subject_lanes,
        comparison_requested=comparison_requested,
    )


def _validate_snapshot_outcome(
    *,
    outcome: QueryProductOutcome | None,
    clarification_reason: str | None,
    document_scope: DocumentScope,
) -> None:
    if clarification_reason is not None and not clarification_reason.strip():
        raise SubjectScopeSnapshotValidationError(
            "Snapshot clarification_reason must not be empty."
        )
    if outcome is QueryProductOutcome.CLARIFICATION_REQUIRED:
        if clarification_reason is None or not document_scope.is_strict_empty:
            raise SubjectScopeSnapshotValidationError(
                "A clarification outcome requires a reason and strict-empty scope."
            )
    elif clarification_reason is not None:
        raise SubjectScopeSnapshotValidationError(
            "A clarification reason requires a clarification_required outcome."
        )

    if outcome is QueryProductOutcome.NO_EVIDENCE and not document_scope.is_strict_empty:
        raise SubjectScopeSnapshotValidationError(
            "A no_evidence snapshot requires a strict-empty document scope."
        )
    if outcome is QueryProductOutcome.ANSWERED and document_scope.is_strict_empty:
        raise SubjectScopeSnapshotValidationError(
            "An answered snapshot cannot contain a strict-empty document scope."
        )
    if outcome is None and document_scope.is_strict_empty:
        raise SubjectScopeSnapshotValidationError(
            "A strict-empty document scope requires a safe product outcome."
        )


def _validate_snapshot_subject_relationships(
    *,
    requested_ids: tuple[uuid.UUID, ...],
    catalog: tuple[SubjectScopeCatalogEntry, ...],
    matched_ids: tuple[uuid.UUID, ...],
    candidates: tuple[SubjectScopeCatalogEntry, ...],
    source: ScopeResolutionSource,
    outcome: QueryProductOutcome | None,
    clarification_reason: str | None,
    document_scope: DocumentScope,
    subject_lanes: tuple[SubjectDocumentLane, ...],
    comparison_requested: bool,
) -> None:
    catalog_by_id = {item.subject_id: item for item in catalog}
    if len(catalog_by_id) != len(catalog):
        raise SubjectScopeSnapshotValidationError("Snapshot catalog ids must be unique.")
    candidate_ids = tuple(item.subject_id for item in candidates)
    if len(set(candidate_ids)) != len(candidate_ids):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot ambiguity candidate ids must be unique."
        )
    if any(
        item.kind is not SubjectKind.PROJECT
        or catalog_by_id.get(item.subject_id) != item
        for item in candidates
    ):
        raise SubjectScopeSnapshotValidationError(
            "Ambiguity candidates must be project entries from the snapshot catalog."
        )
    if not set(matched_ids).issubset(catalog_by_id):
        raise SubjectScopeSnapshotValidationError(
            "Matched subject ids must exist in the snapshot catalog."
        )

    requested = set(requested_ids)
    matched = set(matched_ids)
    missing_requested = requested - catalog_by_id.keys()
    coherent_unknown_explicit = (
        source is ScopeResolutionSource.EXPLICIT
        and outcome is QueryProductOutcome.CLARIFICATION_REQUIRED
        and clarification_reason == "unknown_explicit_subject"
        and bool(missing_requested)
        and not matched
        and not candidates
    )
    if outcome is QueryProductOutcome.CLARIFICATION_REQUIRED:
        if clarification_reason not in {
            "unknown_explicit_subject",
            "unknown_named_project",
            "ambiguous_project_name",
            "hard_scope_document_limit_exceeded",
            "comparison_project_limit_exceeded",
            "multiple_projects_require_comparison_intent",
        }:
            raise SubjectScopeSnapshotValidationError(
                "Snapshot clarification reason is not recognized."
            )
        if clarification_reason == "unknown_explicit_subject" and not coherent_unknown_explicit:
            raise SubjectScopeSnapshotValidationError(
                "An unknown-explicit clarification requires absent requested ids and no matches."
            )
        if clarification_reason == "unknown_named_project" and (
            source
            not in {
                ScopeResolutionSource.INFERRED_PROJECT,
                ScopeResolutionSource.EXPLICIT_AND_INFERRED,
            }
            or candidates
            or matched
            != (
                requested
                if source is ScopeResolutionSource.EXPLICIT_AND_INFERRED
                else set()
            )
        ):
            raise SubjectScopeSnapshotValidationError(
                "An unknown-project clarification has inconsistent source or candidates."
            )
        if clarification_reason == "ambiguous_project_name" and (
            source
            not in {
                ScopeResolutionSource.INFERRED_PROJECT,
                ScopeResolutionSource.EXPLICIT_AND_INFERRED,
            }
            or len(candidates) < 2
            or matched
            != (
                requested
                if source is ScopeResolutionSource.EXPLICIT_AND_INFERRED
                else set()
            )
        ):
            raise SubjectScopeSnapshotValidationError(
                "An ambiguous-project clarification requires inferred project candidates."
            )
        if clarification_reason == "hard_scope_document_limit_exceeded" and not matched:
            raise SubjectScopeSnapshotValidationError(
                "A document-limit clarification requires matched subjects."
            )
    lane_ids = tuple(item.subject_id for item in subject_lanes)
    if len(set(lane_ids)) != len(lane_ids):
        raise SubjectScopeSnapshotValidationError("Snapshot subject lane ids must be unique.")
    if any(
        item.subject_id not in catalog_by_id
        or catalog_by_id[item.subject_id].kind is not SubjectKind.PROJECT
        or catalog_by_id[item.subject_id].name != item.subject_name
        for item in subject_lanes
    ):
        raise SubjectScopeSnapshotValidationError(
            "Subject lanes must reference project entries from the snapshot catalog."
        )
    if not set(lane_ids).issubset(matched):
        raise SubjectScopeSnapshotValidationError(
            "Subject lanes must be a subset of matched subjects."
        )
    matched_project_ids = {
        subject_id
        for subject_id in matched
        if catalog_by_id[subject_id].kind is SubjectKind.PROJECT
    }
    if subject_lanes and set(lane_ids) != matched_project_ids:
        raise SubjectScopeSnapshotValidationError(
            "Subject lanes must cover every matched project exactly once."
        )
    if (
        len(matched_project_ids) > 1
        and not subject_lanes
        and outcome is not QueryProductOutcome.CLARIFICATION_REQUIRED
    ):
        raise SubjectScopeSnapshotValidationError(
            "A persisted multi-project scope cannot omit its per-project lanes."
        )
    lane_documents = {
        document_id
        for lane in subject_lanes
        for document_id in lane.document_scope.allowed_document_ids
    }
    if not lane_documents.issubset(document_scope.allowed_document_ids):
        raise SubjectScopeSnapshotValidationError(
            "Subject lane documents must remain inside the resolved document scope."
        )
    if (
        comparison_requested
        and lane_documents != set(document_scope.allowed_document_ids)
    ):
        raise SubjectScopeSnapshotValidationError(
            "A comparison scope must equal the union of its subject lane documents."
        )
    if comparison_requested and len(subject_lanes) < 2 and outcome is not QueryProductOutcome.CLARIFICATION_REQUIRED:
        raise SubjectScopeSnapshotValidationError(
            "A comparison snapshot requires at least two subject lanes."
        )
    if not comparison_requested and len(subject_lanes) > 1:
        raise SubjectScopeSnapshotValidationError(
            "Multiple subject lanes require comparison_requested."
        )
    if missing_requested and not coherent_unknown_explicit:
        raise SubjectScopeSnapshotValidationError(
            "Requested subject ids absent from the catalog require an unknown-explicit clarification."
        )

    if source is ScopeResolutionSource.GLOBAL:
        if requested or matched or candidates:
            raise SubjectScopeSnapshotValidationError(
                "A global snapshot cannot contain requested, matched, or candidate subjects."
            )
        return

    if source is ScopeResolutionSource.EXPLICIT:
        if not requested:
            raise SubjectScopeSnapshotValidationError(
                "An explicit snapshot requires requested subject ids."
            )
        if coherent_unknown_explicit:
            return
        if missing_requested or matched != requested:
            raise SubjectScopeSnapshotValidationError(
                "Explicit matched subjects must equal the requested catalog subjects."
            )
        if candidates:
            raise SubjectScopeSnapshotValidationError(
                "An explicit-only snapshot cannot contain inferred ambiguity candidates."
            )
        return

    if source is ScopeResolutionSource.EXPLICIT_AND_INFERRED:
        if not requested or missing_requested or not requested.issubset(matched):
            raise SubjectScopeSnapshotValidationError(
                "A combined snapshot must match every requested catalog subject."
            )
        inferred_ids = matched - requested
        if any(catalog_by_id[item].kind is not SubjectKind.PROJECT for item in inferred_ids):
            raise SubjectScopeSnapshotValidationError(
                "Combined inferred matches must be project subjects."
            )
        return

    if requested:
        raise SubjectScopeSnapshotValidationError(
            "An inferred-project snapshot cannot contain requested subject ids."
        )
    if matched:
        if any(catalog_by_id[item].kind is not SubjectKind.PROJECT for item in matched):
            raise SubjectScopeSnapshotValidationError(
                "Inferred matches must be project subjects."
            )
        return
    if not (
        outcome is QueryProductOutcome.CLARIFICATION_REQUIRED
        and clarification_reason
        in {
            "unknown_named_project",
            "ambiguous_project_name",
            "comparison_project_limit_exceeded",
            "multiple_projects_require_comparison_intent",
        }
    ):
        raise SubjectScopeSnapshotValidationError(
            "An inferred-project snapshot requires a matched project or named-project clarification."
        )
    if clarification_reason == "ambiguous_project_name" and len(candidates) < 2:
        raise SubjectScopeSnapshotValidationError(
            "An ambiguous-project clarification requires at least two candidates."
        )
    if clarification_reason == "unknown_named_project" and candidates:
        raise SubjectScopeSnapshotValidationError(
            "An unknown-project clarification cannot contain ambiguity candidates."
        )


def _document_scope_from_snapshot_metadata(value: object) -> DocumentScope:
    if not isinstance(value, dict):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot document_scope must be an object."
        )
    required = {
        "strict",
        "global",
        "strict_empty",
        "allowed_document_ids",
        "allowed_document_count",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise SubjectScopeSnapshotValidationError(
            "Snapshot document_scope is incomplete: " + ", ".join(missing)
        )
    strict = value["strict"]
    if not isinstance(strict, bool):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot document_scope.strict must be boolean."
        )
    ids = _uuid_list(value["allowed_document_ids"], "allowed_document_ids")
    count = value["allowed_document_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count != len(ids):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot allowed_document_count does not match its document ids."
        )
    if value["global"] is not (not strict):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot global flag does not match document scope strictness."
        )
    if value["strict_empty"] is not (strict and not ids):
        raise SubjectScopeSnapshotValidationError(
            "Snapshot strict_empty flag does not match its document ids."
        )
    try:
        return DocumentScope(strict=strict, allowed_document_ids=ids)
    except ValueError as exc:
        raise SubjectScopeSnapshotValidationError(str(exc)) from exc


def _uuid_list(value: object, field: str) -> tuple[uuid.UUID, ...]:
    if not isinstance(value, list):
        raise SubjectScopeSnapshotValidationError(f"Snapshot {field} must be a list.")
    ids = tuple(uuid.UUID(str(item)) for item in value)
    if len(set(ids)) != len(ids):
        raise SubjectScopeSnapshotValidationError(
            f"Snapshot {field} must contain unique ids."
        )
    return ids


def _object_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise SubjectScopeSnapshotValidationError(f"Snapshot {field} must be a list.")
    return value


def _non_empty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SubjectScopeSnapshotValidationError(
            f"Snapshot {field} must be non-empty text."
        )
    return value
