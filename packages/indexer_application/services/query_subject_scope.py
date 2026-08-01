from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass

from packages.indexer_application.dto import QueryRunRecord, SubjectRecord
from packages.indexer_application.ports import UnitOfWork
from packages.rag_core.document_scope import (
    CoverageMode,
    DocumentScope,
    QueryProductOutcome,
    ResolvedSubjectScopeSnapshot,
    SubjectDocumentLane,
    SubjectScopeCatalogEntry,
    resolve_subject_scope,
)


@dataclass(frozen=True, slots=True)
class QuerySubjectScopeConfig:
    max_document_ids: int = 10_000
    max_project_lanes: int = 4
    policy_revision: str = "subject-scope-policy/1.0"

    def __post_init__(self) -> None:
        if self.max_document_ids <= 0:
            raise ValueError("max_document_ids must be positive.")
        if self.max_project_lanes <= 0:
            raise ValueError("max_project_lanes must be positive.")
        if not self.policy_revision.strip():
            raise ValueError("policy_revision must not be empty.")


async def resolve_or_reuse_query_subject_scope(
    *,
    uow: UnitOfWork,
    query_run_id: uuid.UUID,
    config: QuerySubjectScopeConfig,
) -> tuple[QueryRunRecord, ResolvedSubjectScopeSnapshot, bool]:
    """Row-lock and create one immutable catalog/document-scope snapshot."""

    query_run = await uow.query_runs.get_for_update(query_run_id)
    if query_run is None:
        raise LookupError(f"Query run {query_run_id} was not found.")
    existing = query_run.metadata.get("resolved_subject_scope")
    if existing is not None:
        return query_run, ResolvedSubjectScopeSnapshot.from_metadata(existing), True

    requested = query_run.metadata.get("requested_subject_scope")
    requested_payload = requested if isinstance(requested, dict) else {}
    requested_ids = tuple(
        uuid.UUID(str(item))
        for item in requested_payload.get("subject_ids", [])
    )
    coverage_mode = CoverageMode(
        str(requested_payload.get("coverage_mode", CoverageMode.BEST_EVIDENCE.value)),
    )
    snapshot = await resolve_query_subject_scope_snapshot(
        uow=uow,
        question=query_run.question,
        requested_subject_ids=requested_ids,
        coverage_mode=coverage_mode,
        config=config,
    )
    await uow.query_runs.set_subject_scope_snapshot(
        query_run_id=query_run_id,
        snapshot=snapshot.to_metadata(),
    )
    return query_run, snapshot, False


async def resolve_query_subject_scope_snapshot(
    *,
    uow: UnitOfWork,
    question: str,
    requested_subject_ids: tuple[uuid.UUID, ...],
    coverage_mode: CoverageMode,
    config: QuerySubjectScopeConfig,
) -> ResolvedSubjectScopeSnapshot:
    """Resolve the production subject catalog and membership boundary without persistence."""

    subjects = await _active_subjects(uow)
    catalog = tuple(_catalog_entry(item) for item in subjects)
    resolution = resolve_subject_scope(
        question=question,
        requested_subject_ids=requested_subject_ids,
        catalog=catalog,
        max_project_lanes=config.max_project_lanes,
    )

    product_outcome: QueryProductOutcome | None = None
    clarification_reason = resolution.clarification_reason
    subject_lanes: tuple[SubjectDocumentLane, ...] = ()
    if resolution.clarification_required:
        document_scope = DocumentScope.strict_scope(())
        product_outcome = QueryProductOutcome.CLARIFICATION_REQUIRED
    elif resolution.hard_subject_ids:
        if resolution.lane_subjects:
            lane_subject_ids = {item.subject_id for item in resolution.lane_subjects}
            shared_subject_ids = tuple(
                item
                for item in resolution.hard_subject_ids
                if item not in lane_subject_ids
            )
            lanes: list[SubjectDocumentLane] = []
            allowed_union: list[uuid.UUID] = []
            for lane_subject in resolution.lane_subjects:
                lane_allowed = await uow.subjects.list_assigned_document_ids(
                    subject_ids=(*shared_subject_ids, lane_subject.subject_id),
                    require_all=True,
                )
                lanes.append(
                    SubjectDocumentLane(
                        subject_id=lane_subject.subject_id,
                        subject_name=lane_subject.name,
                        document_scope=DocumentScope.strict_scope(lane_allowed),
                    )
                )
                allowed_union.extend(lane_allowed)
            subject_lanes = tuple(lanes)
            allowed_ids = tuple(dict.fromkeys(allowed_union))
        else:
            allowed_ids = await uow.subjects.list_assigned_document_ids(
                subject_ids=resolution.hard_subject_ids,
                require_all=True,
            )
        if len(allowed_ids) > config.max_document_ids:
            document_scope = DocumentScope.strict_scope(())
            subject_lanes = ()
            product_outcome = QueryProductOutcome.CLARIFICATION_REQUIRED
            clarification_reason = "hard_scope_document_limit_exceeded"
        else:
            document_scope = DocumentScope.strict_scope(allowed_ids)
            if document_scope.is_strict_empty:
                product_outcome = QueryProductOutcome.NO_EVIDENCE
    else:
        document_scope = DocumentScope.global_scope()

    snapshot = ResolvedSubjectScopeSnapshot(
        requested_subject_ids=requested_subject_ids,
        catalog=catalog,
        matched_subject_ids=resolution.hard_subject_ids,
        document_scope=document_scope,
        source=resolution.source,
        confidence=resolution.confidence,
        catalog_revision=_catalog_revision(subjects),
        policy_revision=config.policy_revision,
        coverage_mode=coverage_mode,
        product_outcome=product_outcome,
        clarification_reason=clarification_reason,
        ambiguity_candidates=resolution.ambiguity_candidates,
        subject_lanes=subject_lanes,
        comparison_requested=resolution.comparison_requested,
    )
    return snapshot


async def _active_subjects(uow: UnitOfWork) -> list[SubjectRecord]:
    subjects: list[SubjectRecord] = []
    offset = 0
    while True:
        batch = await uow.subjects.list(limit=100, offset=offset)
        subjects.extend(batch)
        if len(batch) < 100:
            return subjects
        offset += len(batch)


def _catalog_entry(subject: SubjectRecord) -> SubjectScopeCatalogEntry:
    return SubjectScopeCatalogEntry(
        subject_id=subject.id,
        kind=subject.kind,
        name=subject.name,
        aliases=tuple(
            alias.name for alias in subject.aliases if alias.archived_at is None
        ),
    )


def _catalog_revision(subjects: list[SubjectRecord]) -> str:
    payload = [
        {
            "id": str(subject.id),
            "kind": subject.kind.value,
            "name": subject.name,
            "updated_at": subject.updated_at.isoformat(),
            "aliases": [
                {
                    "id": str(alias.id),
                    "name": alias.name,
                    "archived_at": (
                        alias.archived_at.isoformat() if alias.archived_at is not None else None
                    ),
                }
                for alias in subject.aliases
            ],
        }
        for subject in subjects
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]
