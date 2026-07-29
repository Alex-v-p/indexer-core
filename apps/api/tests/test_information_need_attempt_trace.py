from __future__ import annotations

import uuid
from copy import deepcopy

from app.api.routes.queries import _to_information_need_resolution_response
from packages.rag_core.agents.information_need_graph.models import InformationNeedExecution
from packages.rag_core.agents.information_need_graph.nodes.execute import ExecuteInformationNeedPlanNode
from packages.rag_core.agents.information_need_graph.nodes.grade import GradeInformationNeedNode
from packages.rag_core.agents.information_need_graph.nodes.validate_constraints import (
    ValidateInformationNeedConstraintsNode,
)
from packages.rag_core.agents.information_need_graph.reporting import InformationNeedResolutionReport
from packages.rag_core.agents.information_need_graph.routes import InformationNeedExecutionStatus
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.query_understanding.classification import QueryType
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.query_understanding.planning import (
    InformationNeedRetrievalPlan,
    RetrievalStrategy,
)
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem


CHUNK_A_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
CHUNK_B_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
CHUNK_C_ID = uuid.UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
DOCUMENT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
VERSION_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class QueuedRetrievalExecutor:
    def __init__(self) -> None:
        self.results = [
            (
                [
                    _item(
                        rank=1,
                        chunk_id=CHUNK_A_ID,
                        text="Accepted deployment evidence.",
                        metadata={
                            "original_filename": "deployment.pdf",
                            "page_number": 3,
                            "retrieval_source": "hybrid",
                            "fusion": {
                                "method": "weighted_reciprocal_rank_fusion",
                                "sources": {
                                    "vector": {"weight": 1.2},
                                    "keyword": {"weight": 0.8},
                                },
                            },
                        },
                    ),
                    _item(
                        rank=2,
                        chunk_id=CHUNK_B_ID,
                        text="Rejected unrelated evidence.",
                        metadata={"original_filename": "unrelated.pdf", "page_number": 8},
                    ),
                ],
                {
                    "top_k": 2,
                    "reranking_applied": True,
                    "preferred_document": {"document": {"key": f"document:{DOCUMENT_ID}"}},
                    "retrieval": {
                        "requested_top_k": 2,
                        "candidate_top_k": 6,
                        "strategy": "multi_query",
                        "fusion_method": "weighted_reciprocal_rank_fusion",
                        "candidate_top_k_per_query": 4,
                        "query_count": 3,
                        "queries": [
                            {"text": "deployment", "kind": "original"},
                            {"text": "deployment requirements", "kind": "variant"},
                            {"text": "production deployment", "kind": "variant"},
                        ],
                        "result_counts": {"0": 2, "1": 2, "2": 1},
                    },
                    "reranking": {
                        "candidate_count": 6,
                        "result_count": 2,
                        "providers": ["test-cross-encoder"],
                    },
                    "document_balancing": {
                        "selector_name": "document_balanced_candidate_selector",
                        "candidate_count": 6,
                        "requested_top_k": 2,
                        "selected_count": 2,
                        "selected_document_counts": {f"document:{DOCUMENT_ID}": 2},
                        "primary_document_key": f"document:{DOCUMENT_ID}",
                        "primary_document_quota": 2,
                        "primary_selected_count": 2,
                        "quota_relaxed": False,
                    },
                },
            ),
            (
                [
                    _item(
                        rank=1,
                        chunk_id=CHUNK_C_ID,
                        text="Accepted retry evidence.",
                        metadata={"original_filename": "retry.pdf", "page_number": 4},
                    ),
                ],
                {
                    "top_k": 1,
                    "reranking_applied": False,
                    "preferred_document": None,
                    "retrieval": {
                        "requested_top_k": 1,
                        "candidate_top_k": 1,
                        "strategy": "vector",
                    },
                    "document_balancing": {
                        "selector_name": "passthrough_document_candidate_selector",
                        "candidate_count": 1,
                        "requested_top_k": 1,
                        "selected_count": 1,
                        "selected_document_counts": {f"document:{DOCUMENT_ID}": 1},
                        "primary_document_key": None,
                        "primary_document_quota": None,
                        "primary_selected_count": 0,
                        "quota_relaxed": False,
                    },
                },
            ),
        ]

    async def execute_lookup(self, *, plan, query: str, top_k: int):
        del plan, query, top_k
        return self.results.pop(0)


class SelectiveGrader:
    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
    ) -> EvidenceGradingReport:
        del question
        need = information_needs[0]
        grades = tuple(
            EvidenceGrade(
                evidence_rank=item.rank,
                relevance_score=0.9 if "Accepted" in item.text else 0.1,
                relevant="Accepted" in item.text,
                rationale=(
                    "Directly answers the information need."
                    if "Accepted" in item.text
                    else "Does not answer the information need."
                ),
                supports_information_need_ids=(
                    (need.need_id,) if "Accepted" in item.text else ()
                ),
            )
            for item in evidence
        )
        supporting_ranks = tuple(grade.evidence_rank for grade in grades if grade.relevant)
        return EvidenceGradingReport(
            status=EvidenceSufficiency.SUFFICIENT,
            coverage_score=0.9,
            grades=grades,
            information_need_grades=(
                InformationNeedGrade(
                    information_need_id=need.need_id,
                    description=need.description,
                    status=InformationNeedSupport.SUPPORTED,
                    coverage_score=0.9,
                    supporting_evidence_ranks=supporting_ranks,
                    rationale="The accepted evidence is sufficient.",
                    required=need.required,
                ),
            ),
            rationale="The accepted evidence resolves the need.",
            grader_name="selective_test",
        )


async def test_attempt_snapshots_preserve_rejected_evidence_and_structured_metadata() -> None:
    state, execution = _state()
    executor = QueuedRetrievalExecutor()

    await _run_attempt(state, execution, executor, attempt_number=1, top_k=2)

    attempt = execution.attempts[0]
    assert attempt.retrieved_count == 2
    assert [item.retrieval_order for item in attempt.evidence] == [1, 2]
    assert [item.aggregate_rank for item in attempt.evidence] == [1, 2]
    assert [item.text for item in attempt.evidence] == [
        "Accepted deployment evidence.",
        "Rejected unrelated evidence.",
    ]
    accepted, rejected = attempt.evidence
    assert accepted.qdrant_chunk_index_id == CHUNK_A_ID
    assert accepted.document_id == DOCUMENT_ID
    assert accepted.document_version_id == VERSION_ID
    assert accepted.metadata["original_filename"] == "deployment.pdf"
    assert accepted.relevance_score == 0.9
    assert accepted.relevant is True
    assert accepted.grading_rationale == "Directly answers the information need."
    assert accepted.supports_information_need_ids == ("need_1",)
    assert accepted.retained_after_need_grading is True
    assert rejected.relevance_score == 0.1
    assert rejected.relevant is False
    assert rejected.grading_rationale == "Does not answer the information need."
    assert rejected.retained_after_need_grading is False
    assert [item.rank for item in state.retrieved_evidence] == [1]

    retrieval = attempt.retrieval_metadata
    assert retrieval.requested_top_k == 2
    assert retrieval.candidate_top_k == 6
    assert retrieval.retriever_type == "multi_query"
    assert retrieval.fusion_method == "weighted_reciprocal_rank_fusion"
    assert retrieval.vector_contribution == 1.2
    assert retrieval.keyword_contribution == 0.8
    assert retrieval.query_variants == (
        "deployment requirements",
        "production deployment",
    )
    assert retrieval.multi_query_query_count == 3
    assert retrieval.multi_query_candidate_top_k_per_query == 4
    assert retrieval.multi_query_result_counts == {"0": 2, "1": 2, "2": 1}

    reranking = attempt.reranking_metadata
    assert reranking.applied is True
    assert reranking.provider == "test-cross-encoder"
    assert reranking.candidate_count_before == 6
    assert reranking.candidate_count_after == 2

    balancing = attempt.document_balancing
    assert balancing.candidate_count == 6
    assert balancing.selected_count == 2
    assert balancing.selected_chunks_per_document == {f"document:{DOCUMENT_ID}": 2}
    assert balancing.primary_document_quota == 2
    assert balancing.primary_document_selected_count == 2
    assert balancing.preferred_document_active is True
    assert balancing.quota_relaxed is False


async def test_aggregate_ranks_are_not_reused_after_rejected_evidence_is_pruned() -> None:
    state, execution = _state()
    executor = QueuedRetrievalExecutor()

    await _run_attempt(state, execution, executor, attempt_number=1, top_k=2)
    await _run_attempt(state, execution, executor, attempt_number=2, top_k=1)

    assert execution.attempts[0].evidence[1].aggregate_rank == 2
    assert execution.attempts[0].evidence[1].retained_after_need_grading is False
    assert execution.attempts[1].evidence[0].aggregate_rank == 3
    assert [item.rank for item in state.retrieved_evidence] == [1, 3]
    assert {grade.evidence_rank for grade in execution.last_grading.grades} == {1, 3}


async def test_query_api_serializes_new_attempt_fields_and_accepts_older_runs() -> None:
    state, execution = _state()
    executor = QueuedRetrievalExecutor()
    await _run_attempt(state, execution, executor, attempt_number=1, top_k=2)
    execution.status = InformationNeedExecutionStatus.SUPPORTED

    report = InformationNeedResolutionReport(
        graph_name="information_need_resolution",
        executions=(execution,),
        total_retrieval_attempts=1,
        max_total_retrieval_attempts=4,
        max_attempts_per_information_need=2,
    )
    response = _to_information_need_resolution_response(
        {"information_need_resolution": report.to_metadata()},
    )

    assert response is not None
    api_attempt = response.executions[0].attempts[0]
    assert len(api_attempt.evidence) == 2
    assert api_attempt.evidence[1].relevant is False
    assert api_attempt.retrieval_metadata.query_variants == [
        "deployment requirements",
        "production deployment",
    ]
    assert api_attempt.reranking_metadata.provider == "test-cross-encoder"
    assert api_attempt.document_balancing.primary_document_quota == 2

    older_metadata = deepcopy(report.to_metadata())
    older_attempt = older_metadata["executions"][0]["attempts"][0]
    older_attempt.pop("evidence")
    older_attempt.pop("retrieval_metadata")
    older_attempt.pop("reranking_metadata")
    older_attempt.pop("document_balancing")
    older_response = _to_information_need_resolution_response(
        {"information_need_resolution": older_metadata},
    )

    assert older_response is not None
    old_api_attempt = older_response.executions[0].attempts[0]
    assert old_api_attempt.evidence == []
    assert old_api_attempt.retrieval_metadata.requested_top_k is None
    assert old_api_attempt.reranking_metadata.applied is False
    assert old_api_attempt.document_balancing.selected_chunks_per_document == {}


async def _run_attempt(
    state: QueryState,
    execution: InformationNeedExecution,
    executor: QueuedRetrievalExecutor,
    *,
    attempt_number: int,
    top_k: int,
) -> None:
    execution.current_plan = _plan(attempt_number=attempt_number, top_k=top_k)
    await ExecuteInformationNeedPlanNode(
        executor,  # type: ignore[arg-type]
        max_total_attempts=4,
        max_accumulated_evidence=10,
    )(state)
    await ValidateInformationNeedConstraintsNode()(state)
    await GradeInformationNeedNode(SelectiveGrader())(state)


def _state() -> tuple[QueryState, InformationNeedExecution]:
    need = InformationNeed(
        need_id="need_1",
        description="Identify the deployment requirements.",
        retrieval_query="deployment requirements",
    )
    execution = InformationNeedExecution(information_need=need, max_attempts=2)
    state = QueryState(
        question="What are the deployment requirements?",
        information_need_executions={need.need_id: execution},
        active_information_need_id=need.need_id,
    )
    return state, execution


def _plan(*, attempt_number: int, top_k: int) -> InformationNeedRetrievalPlan:
    return InformationNeedRetrievalPlan(
        information_need_id="need_1",
        strategy=RetrievalStrategy.RERANK,
        selected_pipeline_name="test_rerank",
        query="deployment requirements",
        top_k=top_k,
        rationale="Use reranking for focused evidence.",
        planner_name="test",
        based_on_query_type=QueryType.FACTUAL_LOOKUP,
        attempt_number=attempt_number,
        requires_reranking=True,
    )


def _item(
    *,
    rank: int,
    chunk_id: uuid.UUID,
    text: str,
    metadata: dict[str, object],
) -> EvidenceItem:
    return EvidenceItem(
        rank=rank,
        text=text,
        score=1.0 / rank,
        qdrant_chunk_index_id=chunk_id,
        document_id=DOCUMENT_ID,
        document_version_id=VERSION_ID,
        metadata=dict(metadata),
    )
