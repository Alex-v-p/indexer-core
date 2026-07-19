from __future__ import annotations

from dataclasses import replace

import pytest

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.shared.retrieval import RetrievalPlanExecution
from packages.rag_core.pipelines import (
    AGENTIC_RAG_NAME,
    BASELINE_RAG_NAME,
    CONTEXTUAL_RAG_NAME,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_RAG_NAME,
    MULTI_QUERY_RAG_NAME,
    build_agentic_rag_graph,
)
from packages.rag_core.query_understanding.classification import (
    MetadataFilterHint,
    QueryClassification,
    QueryType,
)
from packages.rag_core.query_understanding.decomposition import (
    HeuristicInformationNeedDecomposer,
    InformationNeed,
    InformationNeedDecomposition,
)
from packages.rag_core.query_understanding.planning import (
    RetrievalStrategy,
    RuleBasedClaimRetrievalPlanner,
    RuleBasedRetrievalPlanner,
)
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.retry import RuleBasedRetrievalRetryPolicy
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)


def build_planner(*, contextual_available: bool = True) -> RuleBasedRetrievalPlanner:
    return RuleBasedRetrievalPlanner(
        baseline_pipeline_name=BASELINE_RAG_NAME,
        hybrid_pipeline_name=HYBRID_RAG_NAME,
        contextual_pipeline_name=CONTEXTUAL_RAG_NAME,
        multi_query_pipeline_name=MULTI_QUERY_RAG_NAME,
        rerank_pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        low_confidence_threshold=0.55,
        contextual_available=contextual_available,
    )


def build_claim_planner() -> RuleBasedClaimRetrievalPlanner:
    return RuleBasedClaimRetrievalPlanner(max_claims_per_retry=3, max_query_chars=1_200)


def build_retry_policy(*, max_retries: int = 2) -> RuleBasedRetrievalRetryPolicy:
    return RuleBasedRetrievalRetryPolicy(
        pipeline_names={
            RetrievalStrategy.BASELINE: BASELINE_RAG_NAME,
            RetrievalStrategy.HYBRID: HYBRID_RAG_NAME,
            RetrievalStrategy.CONTEXTUAL: CONTEXTUAL_RAG_NAME,
            RetrievalStrategy.MULTI_QUERY: MULTI_QUERY_RAG_NAME,
            RetrievalStrategy.RERANK: HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        },
        max_retries=max_retries,
        top_k_multiplier=2.0,
        max_top_k=20,
    )


def classification(
    query_type: QueryType,
    *,
    confidence: float = 0.9,
    hints: tuple[MetadataFilterHint, ...] = (),
) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        confidence=confidence,
        needs_metadata_filters=bool(hints),
        metadata_filter_hints=hints,
        rationale="Test classification.",
        classifier_name="test",
    )


def decomposition(*retrieval_queries: str) -> InformationNeedDecomposition:
    needs = tuple(
        InformationNeed(
            need_id=f"need_{index}",
            description=query[0].upper() + query[1:] if query else query,
            retrieval_query=query,
        )
        for index, query in enumerate(retrieval_queries, start=1)
    )
    return InformationNeedDecomposition(
        information_needs=needs,
        rationale="Test decomposition.",
        decomposer_name="test",
    )


@pytest.mark.parametrize(
    ("question", "query_classification", "expected_strategy", "expected_pipeline"),
    [
        (
            "What port does the API use?",
            classification(QueryType.FACTUAL_LOOKUP),
            RetrievalStrategy.BASELINE,
            BASELINE_RAG_NAME,
        ),
        (
            "What does deployment-guide.pdf say?",
            classification(QueryType.FACTUAL_LOOKUP, hints=(MetadataFilterHint.DOCUMENT,)),
            RetrievalStrategy.HYBRID,
            HYBRID_RAG_NAME,
        ),
        (
            "Explain the ingestion architecture.",
            classification(QueryType.BROAD_EXPLANATION),
            RetrievalStrategy.CONTEXTUAL,
            CONTEXTUAL_RAG_NAME,
        ),
        (
            "Compare vector and hybrid retrieval.",
            classification(QueryType.COMPARISON),
            RetrievalStrategy.MULTI_QUERY,
            MULTI_QUERY_RAG_NAME,
        ),
        (
            "What changed in the 2026 version?",
            classification(QueryType.VERSION_SPECIFIC, hints=(MetadataFilterHint.DOCUMENT_VERSION,)),
            RetrievalStrategy.HYBRID,
            HYBRID_RAG_NAME,
        ),
        (
            "Which evidence is most relevant to the deployment decision?",
            classification(QueryType.FACTUAL_LOOKUP),
            RetrievalStrategy.RERANK,
            HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        ),
    ],
)
async def test_rule_based_planner_selects_explainable_strategy(
    question: str,
    query_classification: QueryClassification,
    expected_strategy: RetrievalStrategy,
    expected_pipeline: str,
) -> None:
    plan = await build_planner().plan(question, query_classification, decomposition(question))

    assert plan.strategy is expected_strategy
    assert plan.selected_pipeline_name == expected_pipeline
    assert plan.based_on_query_type is query_classification.query_type
    assert plan.rationale
    assert plan.target_information_need_ids == ("need_1",)
    assert plan.requires_reranking is (expected_strategy is RetrievalStrategy.RERANK)


async def test_planner_uses_multi_query_when_contextual_retrieval_is_disabled() -> None:
    plan = await build_planner(contextual_available=False).plan(
        "Explain the complete ingestion workflow.",
        classification(QueryType.BROAD_EXPLANATION),
        decomposition("Explain the complete ingestion workflow."),
    )

    assert plan.strategy is RetrievalStrategy.MULTI_QUERY
    assert plan.selected_pipeline_name == MULTI_QUERY_RAG_NAME
    assert "unavailable" in plan.rationale


async def test_low_classification_confidence_selects_reranking() -> None:
    plan = await build_planner().plan(
        "What is the deployment requirement?",
        replace(classification(QueryType.FACTUAL_LOOKUP), confidence=0.4),
        decomposition("What is the deployment requirement?"),
    )

    assert plan.strategy is RetrievalStrategy.RERANK
    assert plan.selected_pipeline_name == HYBRID_CROSS_ENCODER_RERANK_RAG_NAME


async def test_compound_broad_question_selects_multi_query_and_preserves_needs() -> None:
    plan = await build_planner().plan(
        "What are the pipeline flows and how do they function?",
        classification(QueryType.BROAD_EXPLANATION),
        decomposition("What are the pipeline flows", "how do they function"),
    )

    assert plan.strategy is RetrievalStrategy.MULTI_QUERY
    assert plan.selected_pipeline_name == MULTI_QUERY_RAG_NAME
    assert plan.target_information_need_ids == ("need_1", "need_2")
    assert "2 gradable information needs" in plan.rationale


class StaticClassifier:
    def __init__(self, value: QueryClassification) -> None:
        self.value = value

    async def classify(self, question: str) -> QueryClassification:
        del question
        return self.value


class RecordingRetriever:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        self.calls.append((question, top_k))
        return [
            EvidenceItem(rank=index, text=f"{self.name} evidence {index}", score=1.0 / index)
            for index in range(1, top_k + 1)
        ]


class RecordingReranker:
    def __init__(self) -> None:
        self.candidate_counts: list[int] = []

    async def rerank(
        self,
        question: str,
        evidence: list[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        del question
        self.candidate_counts.append(len(evidence))
        selected = list(reversed(evidence))[:top_k]
        for rank, item in enumerate(selected, start=1):
            item.rank = rank
            item.metadata["rerank"] = {"provider": "test", "fallback_used": False}
        return selected


class StaticAnswerLLM:
    async def generate(self, prompt: str) -> str:
        del prompt
        return "Planned answer [1]."


class SingleNeedDecomposer:
    def __init__(self, query: str) -> None:
        self._query = query

    async def decompose(self, question: str) -> InformationNeedDecomposition:
        del question
        return decomposition(self._query)


class StaticEvidenceGrader:
    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
    ) -> EvidenceGradingReport:
        del question
        need = information_needs[0]
        ranks = tuple(item.rank for item in evidence)
        return EvidenceGradingReport(
            status=EvidenceSufficiency.SUFFICIENT,
            coverage_score=0.9,
            grades=tuple(
                EvidenceGrade(
                    evidence_rank=item.rank,
                    relevance_score=0.9,
                    relevant=True,
                    rationale="Test evidence is relevant.",
                    supports_information_need_ids=(need.need_id,),
                )
                for item in evidence
            ),
            information_need_grades=(
                InformationNeedGrade(
                    information_need_id=need.need_id,
                    description=need.description,
                    status=InformationNeedSupport.SUPPORTED,
                    coverage_score=0.9,
                    supporting_evidence_ranks=ranks,
                    rationale="The information need is supported.",
                    required=need.required,
                ),
            ),
            rationale="Test evidence is sufficient.",
            grader_name="test",
        )


async def test_agentic_graph_executes_only_the_pipeline_planned_for_the_information_need() -> None:
    baseline = RecordingRetriever("baseline")
    multi_query = RecordingRetriever("multi_query")
    graph = build_agentic_rag_graph(
        query_classifier=StaticClassifier(classification(QueryType.COMPARISON)),
        information_need_decomposer=SingleNeedDecomposer("Compare the two retrieval approaches."),
        retrieval_planner=build_planner(),
        executions={
            BASELINE_RAG_NAME: RetrievalPlanExecution(
                pipeline_name=BASELINE_RAG_NAME,
                pipeline_version="test",
                strategy=RetrievalStrategy.BASELINE,
                retriever=baseline,
            ),
            MULTI_QUERY_RAG_NAME: RetrievalPlanExecution(
                pipeline_name=MULTI_QUERY_RAG_NAME,
                pipeline_version="test",
                strategy=RetrievalStrategy.MULTI_QUERY,
                retriever=multi_query,
            ),
        },
        evidence_grader=StaticEvidenceGrader(),
        retry_policy=build_retry_policy(),
        llm_provider=StaticAnswerLLM(),
    )

    state = await graph.run(QueryState(question="Compare the two retrieval approaches.", top_k=2))

    execution = state.information_need_executions["need_1"]
    assert state.pipeline_name == AGENTIC_RAG_NAME
    assert execution.current_plan is not None
    assert execution.current_plan.selected_pipeline_name == MULTI_QUERY_RAG_NAME
    assert baseline.calls == []
    assert multi_query.calls == [("Compare the two retrieval approaches.", 2)]
    assert state.information_need_resolution is not None
    assert state.information_need_resolution.complete is True
    assert [step.name for step in state.trace] == [
        "select_pipeline",
        "classify_query",
        "decompose_information_needs",
        "initialize_information_need_work",
        "select_information_need",
        "classify_information_need",
        "plan_information_need",
        "execute_information_need_plan",
        "validate_information_need_constraints",
        "grade_information_need",
        "decide_information_need",
        "complete_information_need",
        "select_information_need",
        "resolve_information_needs",
        "aggregate_information_needs",
        "prepare_evidence_context",
        "generate_answer",
    ]
    subgraph_steps = [step for step in state.trace if step.metadata.get("graph_depth") == 1]
    assert subgraph_steps
    assert all(step.metadata["graph_name"] == "information_need_resolution" for step in subgraph_steps)
    planning_step = next(step for step in state.trace if step.name == "plan_information_need")
    assert planning_step.metadata["information_need_id"] == "need_1"
    assert planning_step.metadata["information_need_execution"]["current_plan"]["strategy"] == "multi_query"


async def test_agentic_information_need_plan_applies_rerank_candidate_expansion() -> None:
    retriever = RecordingRetriever("hybrid")
    reranker = RecordingReranker()
    graph = build_agentic_rag_graph(
        query_classifier=StaticClassifier(classification(QueryType.FACTUAL_LOOKUP)),
        information_need_decomposer=SingleNeedDecomposer("Which evidence is most relevant to deployment?"),
        retrieval_planner=build_planner(),
        executions={
            HYBRID_CROSS_ENCODER_RERANK_RAG_NAME: RetrievalPlanExecution(
                pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
                pipeline_version="test",
                strategy=RetrievalStrategy.RERANK,
                retriever=retriever,
                reranker=reranker,
                candidate_multiplier=3,
                max_candidates=10,
            ),
        },
        evidence_grader=StaticEvidenceGrader(),
        retry_policy=build_retry_policy(),
        llm_provider=StaticAnswerLLM(),
    )

    state = await graph.run(QueryState(question="Which evidence is most relevant to deployment?", top_k=2))

    assert retriever.calls == [("Which evidence is most relevant to deployment?", 6)]
    assert reranker.candidate_counts == [6]
    assert len(state.retrieved_evidence) == 2
    lookup = state.metadata["active_information_need_lookup"]
    assert lookup["execution"]["reranking_applied"] is True
    assert lookup["execution"]["reranking"]["candidate_count"] == 6



async def test_agentic_subgraph_preserves_date_scope_and_never_answers_from_outside_it() -> None:
    from datetime import UTC, datetime

    from packages.rag_core.query_understanding.temporal import DateRange, DocumentDateConstraint, DocumentDateField
    from packages.rag_core.retrieval.constraint_validation import ConstraintValidationStatus
    from packages.rag_core.retrieval.graders import HeuristicEvidenceGrader

    date_constraint = DocumentDateConstraint(
        field=DocumentDateField.ANY_RECORDED_AT,
        date_range=DateRange(
            start=datetime(2026, 5, 1, tzinfo=UTC),
            end=datetime(2026, 6, 1, tzinfo=UTC),
        ),
        original_expression="May 2026",
        rationale="The question restricts sources to May 2026.",
        detector_name="test",
    )
    scoped_classification = replace(
        classification(QueryType.FACTUAL_LOOKUP, hints=(MetadataFilterHint.DATE_RANGE,)),
        date_constraints=(date_constraint,),
    )

    class OutsideMonthRetriever:
        async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
            del question, top_k
            uploaded = datetime(2026, 6, 10, tzinfo=UTC)
            return [
                EvidenceItem(
                    rank=1,
                    text="This evidence is semantically relevant but outside the requested month.",
                    metadata={
                        "original_filename": "outside.md",
                        "uploaded_at": uploaded.isoformat(),
                        "uploaded_at_epoch": uploaded.timestamp(),
                    },
                ),
            ]

    class RecordingAnswerLLM:
        def __init__(self) -> None:
            self.prompts: list[str] = []

        async def generate(self, prompt: str) -> str:
            self.prompts.append(prompt)
            return "This must not be used."

    llm = RecordingAnswerLLM()
    graph = build_agentic_rag_graph(
        query_classifier=StaticClassifier(scoped_classification),
        information_need_decomposer=SingleNeedDecomposer("Explain metadata filtering."),
        retrieval_planner=build_planner(),
        executions={
            HYBRID_RAG_NAME: RetrievalPlanExecution(
                pipeline_name=HYBRID_RAG_NAME,
                pipeline_version="test",
                strategy=RetrievalStrategy.HYBRID,
                retriever=OutsideMonthRetriever(),
            ),
        },
        evidence_grader=HeuristicEvidenceGrader(),
        retry_policy=build_retry_policy(max_retries=0),
        llm_provider=llm,
        max_retries_per_information_need=0,
    )

    state = await graph.run(
        QueryState(question="Only use data from May 2026 to explain metadata filtering."),
    )

    execution = state.information_need_executions["need_1"]
    assert execution.constraint_validation_history
    assert execution.constraint_validation_history[0].status is ConstraintValidationStatus.NO_MATCH
    assert execution.attempts[0].constraint_validation.status is ConstraintValidationStatus.NO_MATCH
    assert state.constraint_validation is not None
    assert state.constraint_validation.status is ConstraintValidationStatus.NO_MATCH
    assert state.retrieved_evidence == []
    assert state.citations == []
    assert llm.prompts == []
    assert "No indexed evidence matched" in (state.answer or "")
    assert "validate_information_need_constraints" in [step.name for step in state.trace]
    assert "prepare_evidence_context" in [step.name for step in state.trace]
