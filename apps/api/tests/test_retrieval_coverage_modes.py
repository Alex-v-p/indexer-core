from __future__ import annotations

import uuid
from types import SimpleNamespace

from packages.rag_core.agents.shared.retrieval import (
    RetrievalPlanExecution,
    RetrievalPlanExecutor,
)
from packages.rag_core.agents.information_need_graph.nodes.grade import (
    _coverage_satisfied,
    preserve_information_need_multi_document_coverage,
)
from packages.rag_core.agents.query_graph.nodes.arbitrate_evidence import (
    preserve_multi_document_coverage,
)
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.document_scope import CoverageMode, DocumentScope, SubjectDocumentLane
from packages.rag_core.ports import KeywordSearchResult, VectorSearchResult
from packages.rag_core.query_understanding.classification import QueryType
from packages.rag_core.query_understanding.planning import RetrievalPlan, RetrievalStrategy
from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.retrievers import (
    HybridRetriever,
    KeywordRetriever,
    VectorRetriever,
)


async def test_multi_document_searches_per_document_while_best_evidence_searches_lane() -> None:
    documents = tuple(uuid.uuid4() for _ in range(5))
    lane = SubjectDocumentLane(
        uuid.uuid4(), "DAF", DocumentScope.strict_scope(documents)
    )
    retriever = _RecordingRetriever()
    executor = RetrievalPlanExecutor(
        {
            "baseline": RetrievalPlanExecution(
                "baseline", "1", RetrievalStrategy.BASELINE, retriever
            )
        },
        multi_document_fanout=3,
    )

    best, _ = await executor.execute_lookup(
        plan=_plan(lane, CoverageMode.BEST_EVIDENCE),
        query="status",
        top_k=4,
    )
    assert retriever.scopes == [documents]
    assert len({item.document_id for item in best}) == 1

    retriever.scopes.clear()
    multi, metadata = await executor.execute_lookup(
        plan=_plan(lane, CoverageMode.MULTI_DOCUMENT),
        query="status",
        top_k=4,
    )

    assert len(retriever.scopes) == 3
    assert all(len(scope) == 1 for scope in retriever.scopes)
    assert len({item.document_id for item in multi}) == 3
    assert metadata["coverage"]["fallback_reason"] == "multi_document_fanout_truncated"
    assert metadata["coverage"]["searched_document_count"] == 3


def test_final_coverage_preserves_prior_approved_distinct_documents_only_in_multi_mode() -> None:
    first_document, second_document, outside_document = (
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )
    lane = SubjectDocumentLane(
        uuid.uuid4(),
        "DAF",
        DocumentScope.strict_scope((first_document, second_document)),
    )
    evidence = [
        _lane_evidence(1, first_document, lane),
        _lane_evidence(2, second_document, lane),
        _lane_evidence(3, outside_document, lane),
        _lane_evidence(4, second_document, lane),
        EvidenceItem(
            rank=5,
            text="wrong lane attribution",
            document_id=second_document,
            subject_lane_id=lane.lane_id,
            subject_id=uuid.uuid4(),
            subject_name="Other",
        ),
    ]
    prior = _coverage_report(relevant_ranks=(1, 2, 3, 5), grader_name="prior")
    final = _coverage_report(relevant_ranks=(1,), grader_name="final")

    best_state = QueryState(
        question="DAF",
        top_k=5,
        document_scope=lane.document_scope,
        subject_lanes=(lane,),
        coverage_mode=CoverageMode.BEST_EVIDENCE,
        retrieved_evidence=list(evidence),
    )
    best, best_restored = preserve_multi_document_coverage(
        best_state,
        prior=prior,
        final=final,
    )
    assert best.relevant_evidence_ranks == (1,)
    assert best_restored == ()

    multi_state = QueryState(
        question="DAF",
        top_k=5,
        document_scope=lane.document_scope,
        subject_lanes=(lane,),
        coverage_mode=CoverageMode.MULTI_DOCUMENT,
        retrieved_evidence=list(evidence),
    )
    multi, restored = preserve_multi_document_coverage(
        multi_state,
        prior=prior,
        final=final,
    )

    assert restored == (2,)
    assert multi.relevant_evidence_ranks == (1, 2)
    assert 3 not in multi.relevant_evidence_ranks
    assert 4 not in multi.relevant_evidence_ranks
    assert 5 not in multi.relevant_evidence_ranks


def test_need_grading_retains_best_safe_fanout_candidate_before_pruning() -> None:
    first_document, second_document, outside_document = (
        uuid.uuid4(),
        uuid.uuid4(),
        uuid.uuid4(),
    )
    lane = SubjectDocumentLane(
        uuid.uuid4(),
        "DAF",
        DocumentScope.strict_scope((first_document, second_document)),
    )
    evidence = [
        _lane_evidence(1, first_document, lane),
        _lane_evidence(2, second_document, lane),
        _lane_evidence(3, outside_document, lane),
        EvidenceItem(
            rank=4,
            text="wrong lane attribution",
            document_id=second_document,
            subject_lane_id=lane.lane_id,
            subject_id=uuid.uuid4(),
            subject_name="Other",
        ),
    ]
    report = EvidenceGradingReport(
        status=EvidenceSufficiency.SUFFICIENT,
        coverage_score=1.0,
        grades=tuple(
            EvidenceGrade(
                evidence_rank=rank,
                relevance_score=0.9 if rank == 1 else 0.2,
                relevant=rank == 1,
                rationale="accepted" if rank == 1 else "rejected",
                supports_information_need_ids=("need_1",) if rank == 1 else (),
            )
            for rank in range(1, 5)
        ),
        information_need_grades=(
            InformationNeedGrade(
                information_need_id="need_1",
                description="Summarize delivery status",
                status=InformationNeedSupport.SUPPORTED,
                coverage_score=1.0,
                supporting_evidence_ranks=(1,),
                rationale="The first document supports the need.",
            ),
        ),
        rationale="Only the first document was accepted by the model grader.",
        grader_name="test",
    )
    execution = SimpleNamespace(
        current_plan=SimpleNamespace(
            coverage_mode=CoverageMode.MULTI_DOCUMENT,
            document_scope=lane.document_scope,
            subject_lane=lane,
        ),
        pending_retrieval_metadata=SimpleNamespace(
            coverage={
                "effective_mode": CoverageMode.MULTI_DOCUMENT.value,
                "searched_document_ids": [
                    str(first_document),
                    str(second_document),
                ],
            },
        ),
        information_need=SimpleNamespace(need_id="need_1"),
    )
    state = QueryState(
        question="DAF status",
        document_scope=lane.document_scope,
        subject_lanes=(lane,),
        coverage_mode=CoverageMode.MULTI_DOCUMENT,
        active_information_need_id="need_1",
        information_need_executions={"need_1": execution},
    )

    retained_report, retained_ranks = preserve_information_need_multi_document_coverage(
        state,
        evidence=evidence,
        report=report,
    )

    assert retained_ranks == (2,)
    assert retained_report.relevant_evidence_ranks == (1, 2)
    assert retained_report.information_need_grades[0].supporting_evidence_ranks == (1, 2)
    assert retained_report.status is EvidenceSufficiency.SUFFICIENT

    execution.current_plan.coverage_mode = CoverageMode.BEST_EVIDENCE
    best_report, best_ranks = preserve_information_need_multi_document_coverage(
        state,
        evidence=evidence,
        report=report,
    )
    assert best_report is report
    assert best_ranks == ()


def test_need_grading_does_not_promote_all_rejected_fanout_candidates() -> None:
    first_document, second_document = uuid.uuid4(), uuid.uuid4()
    lane = SubjectDocumentLane(
        uuid.uuid4(),
        "DAF",
        DocumentScope.strict_scope((first_document, second_document)),
    )
    evidence = [
        _lane_evidence(1, first_document, lane),
        _lane_evidence(2, second_document, lane),
    ]
    report = EvidenceGradingReport(
        status=EvidenceSufficiency.MISSING,
        coverage_score=0.0,
        grades=tuple(
            EvidenceGrade(
                evidence_rank=rank,
                relevance_score=0.1,
                relevant=False,
                rationale="rejected",
            )
            for rank in range(1, 3)
        ),
        information_need_grades=(
            InformationNeedGrade(
                information_need_id="need_1",
                description="Summarize delivery status",
                status=InformationNeedSupport.MISSING,
                coverage_score=0.0,
                supporting_evidence_ranks=(),
                rationale="No evidence supports the need.",
            ),
        ),
        rationale="All candidates were rejected by the model grader.",
        grader_name="test",
    )
    execution = SimpleNamespace(
        current_plan=SimpleNamespace(
            coverage_mode=CoverageMode.MULTI_DOCUMENT,
            document_scope=lane.document_scope,
            subject_lane=lane,
        ),
        pending_retrieval_metadata=SimpleNamespace(
            coverage={
                "effective_mode": CoverageMode.MULTI_DOCUMENT.value,
                "searched_document_ids": [
                    str(first_document),
                    str(second_document),
                ],
            },
        ),
        information_need=SimpleNamespace(need_id="need_1"),
    )
    state = QueryState(
        question="DAF status",
        document_scope=lane.document_scope,
        subject_lanes=(lane,),
        coverage_mode=CoverageMode.MULTI_DOCUMENT,
        active_information_need_id="need_1",
        information_need_executions={"need_1": execution},
    )

    retained_report, retained_ranks = preserve_information_need_multi_document_coverage(
        state,
        evidence=evidence,
        report=report,
    )

    assert retained_report is report
    assert retained_ranks == ()
    assert retained_report.status is EvidenceSufficiency.MISSING
    assert retained_report.relevant_evidence_ranks == ()
    assert retained_report.information_need_grades[0].supporting_evidence_ranks == ()


def test_final_coverage_recomputes_two_required_needs_after_restoration() -> None:
    first_document, second_document = uuid.uuid4(), uuid.uuid4()
    lane = SubjectDocumentLane(
        uuid.uuid4(),
        "DAF",
        DocumentScope.strict_scope((first_document, second_document)),
    )
    state = QueryState(
        question="Compare DAF documents",
        top_k=2,
        document_scope=lane.document_scope,
        subject_lanes=(lane,),
        coverage_mode=CoverageMode.MULTI_DOCUMENT,
        retrieved_evidence=[
            _lane_evidence(1, first_document, lane),
            _lane_evidence(2, second_document, lane),
        ],
    )
    prior = _two_need_report(
        second_status=InformationNeedSupport.SUPPORTED,
        second_supporting_ranks=(2,),
        second_relevant=True,
        status=EvidenceSufficiency.SUFFICIENT,
    )
    final = _two_need_report(
        second_status=InformationNeedSupport.MISSING,
        second_supporting_ranks=(),
        second_relevant=False,
        status=EvidenceSufficiency.WEAK,
    )

    reconciled, restored = preserve_multi_document_coverage(
        state,
        prior=prior,
        final=final,
    )

    assert restored == (2,)
    assert reconciled.status is EvidenceSufficiency.SUFFICIENT
    assert abs(reconciled.coverage_score - 0.7) < 1e-9
    assert reconciled.partial_answer_available is False
    assert reconciled.unresolved_information == ()


async def test_best_evidence_soft_cap_bounds_large_document_domination() -> None:
    dominant, secondary = uuid.uuid4(), uuid.uuid4()
    lane = SubjectDocumentLane(
        uuid.uuid4(), "DAF", DocumentScope.strict_scope((dominant, secondary))
    )
    retriever = _DominantRetriever(dominant, secondary)
    executor = RetrievalPlanExecutor(
        {
            "baseline": RetrievalPlanExecution(
                "baseline", "1", RetrievalStrategy.BASELINE, retriever
            )
        }
    )

    evidence, _ = await executor.execute_lookup(
        plan=_plan(lane, CoverageMode.BEST_EVIDENCE),
        query="status",
        top_k=4,
    )

    assert sum(item.document_id == dominant for item in evidence) == 3
    assert any(item.document_id == secondary for item in evidence)


async def test_global_multi_document_fallback_uses_effective_best_evidence_coverage() -> None:
    retriever = _GlobalRetriever()
    executor = RetrievalPlanExecutor(
        {
            "baseline": RetrievalPlanExecution(
                "baseline", "1", RetrievalStrategy.BASELINE, retriever
            )
        }
    )
    plan = RetrievalPlan(
        strategy=RetrievalStrategy.BASELINE,
        selected_pipeline_name="baseline",
        rationale="test",
        planner_name="test",
        based_on_query_type=QueryType.FACTUAL_LOOKUP,
        coverage_mode=CoverageMode.MULTI_DOCUMENT,
    )

    evidence, metadata = await executor.execute_lookup(
        plan=plan,
        query="status",
        top_k=3,
    )

    assert evidence
    assert metadata["coverage"]["requested_mode"] == "multi_document"
    assert metadata["coverage"]["effective_mode"] == "best_evidence"
    assert _coverage_satisfied(
        effective_mode=metadata["coverage"]["effective_mode"],
        searched_document_count=metadata["coverage"]["searched_document_count"],
        contributing_document_count=1,
    ) is True


async def test_leaky_retriever_reports_prefilter_rejection_in_compact_coverage_trace() -> None:
    allowed, outside = uuid.uuid4(), uuid.uuid4()
    lane = SubjectDocumentLane(
        uuid.uuid4(), "DAF", DocumentScope.strict_scope((allowed,))
    )
    executor = RetrievalPlanExecutor(
        {
            "baseline": RetrievalPlanExecution(
                "baseline", "1", RetrievalStrategy.BASELINE, _LeakyRetriever(allowed, outside)
            )
        }
    )

    evidence, metadata = await executor.execute_lookup(
        plan=_plan(lane, CoverageMode.BEST_EVIDENCE),
        query="status",
        top_k=3,
    )

    assert [item.document_id for item in evidence] == [allowed]
    assert metadata["coverage"]["out_of_scope_rejected_count"] == 1
    assert metadata["coverage"]["allowed_document_ids"] == [str(allowed)]


async def test_vector_and_keyword_retrievers_report_exact_scope_rejections() -> None:
    allowed, outside = uuid.uuid4(), uuid.uuid4()
    lane = SubjectDocumentLane(
        uuid.uuid4(), "DAF", DocumentScope.strict_scope((allowed,))
    )
    vector = VectorRetriever(
        embedding_provider=_EmbeddingProvider(),
        vector_store=_VectorStore(allowed, outside),
        vector_name="content",
    )
    keyword = KeywordRetriever(keyword_store=_KeywordStore(allowed, outside))

    for retriever in (vector, keyword):
        executor = _executor(retriever)
        evidence, metadata = await executor.execute_lookup(
            plan=_plan(lane, CoverageMode.BEST_EVIDENCE),
            query="status",
            top_k=3,
        )

        assert [item.document_id for item in evidence] == [allowed]
        assert metadata["coverage"]["out_of_scope_rejected_count"] == 1


async def test_hybrid_composite_aggregates_child_scope_rejections_exactly() -> None:
    allowed, outside = uuid.uuid4(), uuid.uuid4()
    lane = SubjectDocumentLane(
        uuid.uuid4(), "DAF", DocumentScope.strict_scope((allowed,))
    )
    retriever = HybridRetriever(
        vector_retriever=VectorRetriever(
            embedding_provider=_EmbeddingProvider(),
            vector_store=_VectorStore(allowed, outside),
            vector_name="content",
        ),
        keyword_retriever=KeywordRetriever(
            keyword_store=_KeywordStore(allowed, outside)
        ),
    )

    evidence, metadata = await _executor(retriever).execute_lookup(
        plan=_plan(lane, CoverageMode.BEST_EVIDENCE),
        query="status",
        top_k=3,
    )

    assert evidence and all(item.document_id == allowed for item in evidence)
    assert metadata["coverage"]["out_of_scope_rejected_count"] == 2


class _RecordingRetriever:
    def __init__(self) -> None:
        self.scopes: list[tuple[uuid.UUID, ...]] = []

    async def retrieve(self, question, *, top_k, constraints=None):
        scope = constraints.document_scope.allowed_document_ids
        self.scopes.append(scope)
        # A lane-wide best-evidence lookup is intentionally dominated by its
        # first document; per-document mode yields one useful candidate per call.
        document_id = scope[0]
        return [
            EvidenceItem(rank=index, text=f"{document_id}-{index}", document_id=document_id)
            for index in range(1, top_k + 1)
        ]


def _lane_evidence(
    rank: int,
    document_id: uuid.UUID,
    lane: SubjectDocumentLane,
) -> EvidenceItem:
    return EvidenceItem(
        rank=rank,
        text=f"evidence-{rank}",
        document_id=document_id,
        subject_lane_id=lane.lane_id,
        subject_id=lane.subject_id,
        subject_name=lane.subject_name,
    )


def _coverage_report(
    *,
    relevant_ranks: tuple[int, ...],
    grader_name: str,
) -> EvidenceGradingReport:
    return EvidenceGradingReport(
        status=EvidenceSufficiency.SUFFICIENT,
        coverage_score=1.0,
        grades=tuple(
            EvidenceGrade(
                evidence_rank=rank,
                relevance_score=0.9 if rank in relevant_ranks else 0.1,
                relevant=rank in relevant_ranks,
                rationale="approved" if rank in relevant_ranks else "rejected",
                supports_information_need_ids=("need",) if rank in relevant_ranks else (),
            )
            for rank in range(1, 6)
        ),
        rationale="coverage test",
        grader_name=grader_name,
    )


def _two_need_report(
    *,
    second_status: InformationNeedSupport,
    second_supporting_ranks: tuple[int, ...],
    second_relevant: bool,
    status: EvidenceSufficiency,
) -> EvidenceGradingReport:
    return EvidenceGradingReport(
        status=status,
        coverage_score=0.7 if second_relevant else 0.5,
        grades=(
            EvidenceGrade(
                evidence_rank=1,
                relevance_score=0.9,
                relevant=True,
                rationale="supports the first need",
                supports_information_need_ids=("need_1",),
            ),
            EvidenceGrade(
                evidence_rank=2,
                relevance_score=0.8 if second_relevant else 0.2,
                relevant=second_relevant,
                rationale="second need assessment",
                supports_information_need_ids=("need_2",) if second_relevant else (),
            ),
        ),
        information_need_grades=(
            InformationNeedGrade(
                information_need_id="need_1",
                description="First required need",
                status=InformationNeedSupport.SUPPORTED,
                coverage_score=0.8,
                supporting_evidence_ranks=(1,),
                rationale="The first need is supported.",
            ),
            InformationNeedGrade(
                information_need_id="need_2",
                description="Second required need",
                status=second_status,
                coverage_score=0.6 if second_relevant else 0.2,
                supporting_evidence_ranks=second_supporting_ranks,
                rationale="The second need depends on the second document.",
            ),
        ),
        rationale="Two-need reconciliation test.",
        grader_name="test",
    )


class _DominantRetriever:
    def __init__(self, dominant: uuid.UUID, secondary: uuid.UUID) -> None:
        self.dominant = dominant
        self.secondary = secondary

    async def retrieve(self, question, *, top_k, constraints=None):
        return [
            EvidenceItem(rank=1, text="dominant-1", document_id=self.dominant),
            EvidenceItem(rank=2, text="dominant-2", document_id=self.dominant),
            EvidenceItem(rank=3, text="dominant-3", document_id=self.dominant),
            EvidenceItem(rank=4, text="dominant-4", document_id=self.dominant),
            EvidenceItem(rank=5, text="secondary", document_id=self.secondary),
        ]


class _GlobalRetriever:
    async def retrieve(self, question, *, top_k, constraints=None):
        return [EvidenceItem(rank=1, text="relevant", document_id=uuid.uuid4())]


class _LeakyRetriever:
    def __init__(self, allowed: uuid.UUID, outside: uuid.UUID) -> None:
        self.allowed = allowed
        self.outside = outside

    async def retrieve(self, question, *, top_k, constraints=None):
        return [
            EvidenceItem(rank=1, text="allowed", document_id=self.allowed),
            EvidenceItem(rank=2, text="outside", document_id=self.outside),
        ]


class _EmbeddingProvider:
    async def embed_texts(self, texts):
        return [[1.0] for _ in texts]


class _VectorStore:
    def __init__(self, allowed: uuid.UUID, outside: uuid.UUID) -> None:
        self.allowed = allowed
        self.outside = outside

    async def ensure_collection(self):
        return None

    async def search_by_vector(self, vector, **kwargs):
        return [
            VectorSearchResult(
                id="allowed-vector",
                score=1.0,
                payload={"text": "allowed", "document_id": str(self.allowed)},
            ),
            VectorSearchResult(
                id="outside-vector",
                score=0.9,
                payload={"text": "outside", "document_id": str(self.outside)},
            ),
        ]


class _KeywordStore:
    def __init__(self, allowed: uuid.UUID, outside: uuid.UUID) -> None:
        self.allowed = allowed
        self.outside = outside

    async def search(self, query, **kwargs):
        return [
            KeywordSearchResult(
                id="allowed-keyword",
                score=1.0,
                payload={"text": "allowed", "document_id": str(self.allowed)},
            ),
            KeywordSearchResult(
                id="outside-keyword",
                score=0.9,
                payload={"text": "outside", "document_id": str(self.outside)},
            ),
        ]


def _executor(retriever) -> RetrievalPlanExecutor:
    return RetrievalPlanExecutor(
        {
            "baseline": RetrievalPlanExecution(
                "baseline", "1", RetrievalStrategy.BASELINE, retriever
            )
        }
    )


def _plan(lane: SubjectDocumentLane, mode: CoverageMode) -> RetrievalPlan:
    return RetrievalPlan(
        strategy=RetrievalStrategy.BASELINE,
        selected_pipeline_name="baseline",
        rationale="test",
        planner_name="test",
        based_on_query_type=QueryType.COMPARISON,
        document_scope=lane.document_scope,
        subject_lane=lane,
        coverage_mode=mode,
    )
