from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from packages.indexer_application.dto import (
    QueryRunRecord,
    QueryRunStatus,
    SubjectAliasRecord,
    SubjectRecord,
)
from packages.indexer_application.services.query_subject_scope import (
    QuerySubjectScopeConfig,
    resolve_or_reuse_query_subject_scope,
    resolve_query_subject_scope_snapshot,
)
from packages.rag_core.document_scope import (
    CoverageMode,
    DocumentScope,
    QueryProductOutcome,
    ResolvedSubjectScopeSnapshot,
    ScopeResolutionSource,
    SubjectScopeCatalogEntry,
    SubjectDocumentLane,
    SubjectScopeSnapshotValidationError,
)
from packages.rag_core.subjects import SubjectKind

NOW = datetime(2026, 8, 1, tzinfo=UTC)


class QueryRuns:
    def __init__(self, record: QueryRunRecord) -> None:
        self.record = record
        self.lock_count = 0
        self.snapshot_write_count = 0

    async def get_for_update(self, query_run_id):
        assert query_run_id == self.record.id
        self.lock_count += 1
        return self.record

    async def set_subject_scope_snapshot(self, *, query_run_id, snapshot):
        assert query_run_id == self.record.id
        assert "resolved_subject_scope" not in self.record.metadata
        self.snapshot_write_count += 1
        self.record = replace(
            self.record,
            metadata={**self.record.metadata, "resolved_subject_scope": snapshot},
        )


class Subjects:
    def __init__(self, subjects, allowed_ids) -> None:
        self.subjects = subjects
        self.allowed_ids = allowed_ids
        self.list_calls = 0
        self.assignment_calls = 0

    async def list(self, *, limit, offset, **kwargs):
        self.list_calls += 1
        return self.subjects[offset : offset + limit]

    async def list_assigned_document_ids(self, *, subject_ids, require_all=False):
        self.assignment_calls += 1
        assert require_all is True
        return self.allowed_ids


class Uow:
    def __init__(self, query_run, subjects) -> None:
        self.query_runs = QueryRuns(query_run)
        self.subjects = subjects


async def test_shared_scope_resolver_handles_explicit_and_clarification_paths() -> None:
    project = _subject("DAF")
    allowed = (uuid.uuid4(),)
    explicit_subjects = Subjects([project], allowed)
    explicit = await resolve_query_subject_scope_snapshot(
        uow=Uow(_query("DAF status"), explicit_subjects),
        question="DAF status",
        requested_subject_ids=(project.id,),
        coverage_mode=CoverageMode.BEST_EVIDENCE,
        config=QuerySubjectScopeConfig(),
    )

    assert explicit.requested_subject_ids == (project.id,)
    assert explicit.matched_subject_ids == (project.id,)
    assert explicit.document_scope == DocumentScope.strict_scope(allowed)
    assert explicit.product_outcome is None
    assert explicit_subjects.assignment_calls == 1

    clarification_subjects = Subjects([project], allowed)
    clarification = await resolve_query_subject_scope_snapshot(
        uow=Uow(_query("Project Zephyr status"), clarification_subjects),
        question="Project Zephyr status",
        requested_subject_ids=(),
        coverage_mode=CoverageMode.BEST_EVIDENCE,
        config=QuerySubjectScopeConfig(),
    )

    assert clarification.product_outcome is QueryProductOutcome.CLARIFICATION_REQUIRED
    assert clarification.clarification_reason == "unknown_named_project"
    assert clarification.document_scope.is_strict_empty
    assert clarification_subjects.assignment_calls == 0

    archived_alias = SubjectAliasRecord(
        id=uuid.uuid4(),
        subject_id=project.id,
        name="Legacy",
        normalized_name="legacy",
        created_at=NOW,
        archived_at=NOW,
    )
    archived_subjects = Subjects(
        [replace(project, aliases=(archived_alias,))],
        allowed,
    )
    archived = await resolve_query_subject_scope_snapshot(
        uow=Uow(_query("Project Legacy status"), archived_subjects),
        question="Project Legacy status",
        requested_subject_ids=(),
        coverage_mode=CoverageMode.BEST_EVIDENCE,
        config=QuerySubjectScopeConfig(),
    )

    assert archived.product_outcome is QueryProductOutcome.CLARIFICATION_REQUIRED
    assert archived.clarification_reason == "unknown_named_project"
    assert archived_subjects.assignment_calls == 0


async def test_scope_snapshot_is_created_once_and_reused_by_retry_or_stale_worker() -> None:
    project = _subject("DAF")
    allowed = (uuid.uuid4(), uuid.uuid4())
    subjects = Subjects([project], allowed)
    uow = Uow(_query("What changed in DAF?"), subjects)

    _, first, first_reused = await resolve_or_reuse_query_subject_scope(
        uow=uow,
        query_run_id=uow.query_runs.record.id,
        config=QuerySubjectScopeConfig(),
    )
    _, second, second_reused = await resolve_or_reuse_query_subject_scope(
        uow=uow,
        query_run_id=uow.query_runs.record.id,
        config=QuerySubjectScopeConfig(),
    )

    assert first_reused is False
    assert second_reused is True
    assert second == first
    assert first.document_scope.allowed_document_ids == allowed
    assert uow.query_runs.snapshot_write_count == 1
    assert subjects.assignment_calls == 1
    assert subjects.list_calls == 1


async def test_strict_empty_and_document_limit_produce_safe_outcomes() -> None:
    project = _subject("DAF")
    empty_uow = Uow(_query("DAF status"), Subjects([project], ()))
    _, empty, _ = await resolve_or_reuse_query_subject_scope(
        uow=empty_uow,
        query_run_id=empty_uow.query_runs.record.id,
        config=QuerySubjectScopeConfig(),
    )

    large_uow = Uow(
        _query("DAF status"),
        Subjects([project], (uuid.uuid4(), uuid.uuid4())),
    )
    _, large, _ = await resolve_or_reuse_query_subject_scope(
        uow=large_uow,
        query_run_id=large_uow.query_runs.record.id,
        config=QuerySubjectScopeConfig(max_document_ids=1),
    )

    assert empty.document_scope.is_strict_empty
    assert empty.product_outcome is QueryProductOutcome.NO_EVIDENCE
    assert large.document_scope.is_strict_empty
    assert large.product_outcome is QueryProductOutcome.CLARIFICATION_REQUIRED
    assert large.clarification_reason == "hard_scope_document_limit_exceeded"


async def test_oversized_comparison_snapshot_clears_lanes_and_reuses_coherently() -> None:
    first = _subject("DAF")
    second = _subject("Large Internship")
    uow = Uow(
        _query("Compare DAF and Large Internship"),
        Subjects([first, second], (uuid.uuid4(), uuid.uuid4())),
    )

    _, created, created_reused = await resolve_or_reuse_query_subject_scope(
        uow=uow,
        query_run_id=uow.query_runs.record.id,
        config=QuerySubjectScopeConfig(max_document_ids=1),
    )
    _, reused, retry_reused = await resolve_or_reuse_query_subject_scope(
        uow=uow,
        query_run_id=uow.query_runs.record.id,
        config=QuerySubjectScopeConfig(max_document_ids=1),
    )

    assert created_reused is False
    assert retry_reused is True
    assert reused == created
    assert created.document_scope.is_strict_empty
    assert created.subject_lanes == ()
    assert created.product_outcome is QueryProductOutcome.CLARIFICATION_REQUIRED
    assert created.clarification_reason == "hard_scope_document_limit_exceeded"


@pytest.mark.parametrize(
    "outcome",
    (
        None,
        QueryProductOutcome.ANSWERED,
        QueryProductOutcome.NO_EVIDENCE,
        QueryProductOutcome.CLARIFICATION_REQUIRED,
    ),
)
@pytest.mark.parametrize("mismatch", ("extra_overall", "omitted_lane_union"))
def test_live_comparison_snapshot_requires_exact_lane_document_union(
    outcome: QueryProductOutcome | None,
    mismatch: str,
) -> None:
    first = _catalog_subject("DAF")
    second = _catalog_subject("Large Internship")
    first_document, second_document, extra_document = (
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )
    terminal_empty = outcome in {
        QueryProductOutcome.NO_EVIDENCE,
        QueryProductOutcome.CLARIFICATION_REQUIRED,
    }
    first_scope = DocumentScope.strict_scope(() if terminal_empty else (first_document,))
    second_scope = DocumentScope.strict_scope(() if terminal_empty else (second_document,))
    overall_scope = DocumentScope.strict_scope(
        () if terminal_empty else (first_document, second_document)
    )
    snapshot = ResolvedSubjectScopeSnapshot(
        requested_subject_ids=(),
        catalog=(first, second),
        matched_subject_ids=(first.subject_id, second.subject_id),
        document_scope=overall_scope,
        source=ScopeResolutionSource.INFERRED_PROJECT,
        confidence=0.9,
        catalog_revision="catalog",
        policy_revision="policy",
        product_outcome=outcome,
        clarification_reason=(
            "hard_scope_document_limit_exceeded"
            if outcome is QueryProductOutcome.CLARIFICATION_REQUIRED
            else None
        ),
        subject_lanes=(
            SubjectDocumentLane(
                first.subject_id,
                first.name,
                first_scope,
            ),
            SubjectDocumentLane(
                second.subject_id,
                second.name,
                second_scope,
            ),
        ),
        comparison_requested=True,
    )
    malformed = snapshot.to_metadata()
    assert ResolvedSubjectScopeSnapshot.from_metadata(malformed) == snapshot
    if mismatch == "extra_overall":
        malformed["document_scope"] = DocumentScope.strict_scope(
            (*overall_scope.allowed_document_ids, extra_document)
        ).to_metadata()
    else:
        malformed["subject_lanes"][0]["document_scope"] = DocumentScope.strict_scope(
            (*first_scope.allowed_document_ids, extra_document)
        ).to_metadata()

    with pytest.raises(SubjectScopeSnapshotValidationError):
        ResolvedSubjectScopeSnapshot.from_metadata(malformed)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda snapshot: snapshot.pop("document_scope"),
        lambda snapshot: snapshot.__setitem__("document_scope", None),
        lambda snapshot: snapshot["document_scope"].__setitem__("strict", False),
        lambda snapshot: snapshot["document_scope"].pop("allowed_document_ids"),
        lambda snapshot: snapshot["document_scope"].__setitem__(
            "allowed_document_count", 0
        ),
        lambda snapshot: snapshot.__setitem__("matched_subject_ids", [str(uuid.uuid4())]),
        lambda snapshot: snapshot.__setitem__(
            "requested_subject_ids", [str(uuid.uuid4())]
        ),
        lambda snapshot: snapshot.__setitem__("source", "explicit"),
        lambda snapshot: (
            snapshot.__setitem__("source", "explicit"),
            snapshot.__setitem__(
                "requested_subject_ids", [snapshot["catalog"][0]["subject_id"]]
            ),
            snapshot.__setitem__("matched_subject_ids", []),
        ),
        lambda snapshot: snapshot["catalog"][0].__setitem__("kind", "topic"),
        lambda snapshot: snapshot.__setitem__(
            "clarification_reason", "unknown_named_project"
        ),
        lambda snapshot: snapshot.__setitem__(
            "product_outcome", "clarification_required"
        ),
        lambda snapshot: snapshot.__setitem__("product_outcome", "no_evidence"),
        lambda snapshot: (
            snapshot.__setitem__("product_outcome", "answered"),
            snapshot.__setitem__(
                "document_scope", DocumentScope.strict_scope(()).to_metadata()
            ),
        ),
        lambda snapshot: (
            snapshot.__setitem__("strict", False),
            snapshot.__setitem__(
                "document_scope", DocumentScope.global_scope().to_metadata()
            ),
        ),
        lambda snapshot: (
            snapshot.__setitem__("source", "global"),
            snapshot.__setitem__("method", "explicit_subject_selection"),
            snapshot.__setitem__("matched_subject_ids", []),
            snapshot.__setitem__("strict", False),
            snapshot.__setitem__(
                "document_scope", DocumentScope.global_scope().to_metadata()
            ),
        ),
    ),
)
async def test_retry_rejects_partial_or_inconsistent_strict_snapshots(mutate) -> None:
    project = _subject("DAF")
    uow = Uow(_query("DAF status"), Subjects([project], (uuid.uuid4(),)))
    _, snapshot, _ = await resolve_or_reuse_query_subject_scope(
        uow=uow,
        query_run_id=uow.query_runs.record.id,
        config=QuerySubjectScopeConfig(),
    )
    malformed = deepcopy(snapshot.to_metadata())
    mutate(malformed)
    uow.query_runs.record = replace(
        uow.query_runs.record,
        metadata={
            **uow.query_runs.record.metadata,
            "resolved_subject_scope": malformed,
        },
    )

    with pytest.raises(SubjectScopeSnapshotValidationError):
        await resolve_or_reuse_query_subject_scope(
            uow=uow,
            query_run_id=uow.query_runs.record.id,
            config=QuerySubjectScopeConfig(),
        )

    assert uow.query_runs.snapshot_write_count == 1


def test_valid_unknown_and_ambiguous_clarification_snapshots_round_trip() -> None:
    first = _catalog_subject("DAF One")
    second = _catalog_subject("DAF Two")
    unknown_explicit_id = uuid.uuid4()
    snapshots = (
        ResolvedSubjectScopeSnapshot(
            requested_subject_ids=(unknown_explicit_id,),
            catalog=(first,),
            matched_subject_ids=(),
            document_scope=DocumentScope.strict_scope(()),
            source=ScopeResolutionSource.EXPLICIT,
            confidence=1.0,
            catalog_revision="catalog",
            policy_revision="policy",
            coverage_mode=CoverageMode.BEST_EVIDENCE,
            product_outcome=QueryProductOutcome.CLARIFICATION_REQUIRED,
            clarification_reason="unknown_explicit_subject",
        ),
        ResolvedSubjectScopeSnapshot(
            requested_subject_ids=(),
            catalog=(first,),
            matched_subject_ids=(),
            document_scope=DocumentScope.strict_scope(()),
            source=ScopeResolutionSource.INFERRED_PROJECT,
            confidence=0.45,
            catalog_revision="catalog",
            policy_revision="policy",
            product_outcome=QueryProductOutcome.CLARIFICATION_REQUIRED,
            clarification_reason="unknown_named_project",
        ),
        ResolvedSubjectScopeSnapshot(
            requested_subject_ids=(),
            catalog=(first, second),
            matched_subject_ids=(),
            document_scope=DocumentScope.strict_scope(()),
            source=ScopeResolutionSource.INFERRED_PROJECT,
            confidence=0.5,
            catalog_revision="catalog",
            policy_revision="policy",
            product_outcome=QueryProductOutcome.CLARIFICATION_REQUIRED,
            clarification_reason="ambiguous_project_name",
            ambiguity_candidates=(first, second),
        ),
    )

    for snapshot in snapshots:
        assert ResolvedSubjectScopeSnapshot.from_metadata(snapshot.to_metadata()) == snapshot


async def test_absent_snapshot_is_freshly_resolved_for_historical_record() -> None:
    query = replace(_query("How does retrieval work?"), metadata={})
    uow = Uow(query, Subjects([], ()))

    _, snapshot, reused = await resolve_or_reuse_query_subject_scope(
        uow=uow,
        query_run_id=query.id,
        config=QuerySubjectScopeConfig(),
    )

    assert reused is False
    assert snapshot.document_scope.is_global
    assert uow.query_runs.snapshot_write_count == 1


def _query(question: str) -> QueryRunRecord:
    return QueryRunRecord(
        id=uuid.uuid4(),
        question=question,
        answer=None,
        status=QueryRunStatus.PENDING,
        pipeline_name="baseline_rag",
        pipeline_version="1",
        top_k=5,
        started_at=NOW,
        completed_at=None,
        error_message=None,
        metadata={
            "requested_subject_scope": {
                "subject_ids": [],
                "coverage_mode": "best_evidence",
            },
        },
    )


def _subject(name: str) -> SubjectRecord:
    return SubjectRecord(
        id=uuid.uuid4(),
        kind=SubjectKind.PROJECT,
        name=name,
        normalized_name=name.casefold(),
        description=None,
        metadata={},
        created_at=NOW,
        updated_at=NOW,
    )


def _catalog_subject(name: str) -> SubjectScopeCatalogEntry:
    return SubjectScopeCatalogEntry(
        subject_id=uuid.uuid4(),
        kind=SubjectKind.PROJECT,
        name=name,
    )
