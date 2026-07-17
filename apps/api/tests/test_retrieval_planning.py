from __future__ import annotations

from dataclasses import replace

import pytest

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.nodes import RetrievalPlanExecution
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
from packages.rag_core.query_understanding.planning import (
    RetrievalStrategy,
    RuleBasedRetrievalPlanner,
)
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
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
    plan = await build_planner().plan(question, query_classification)

    assert plan.strategy is expected_strategy
    assert plan.selected_pipeline_name == expected_pipeline
    assert plan.based_on_query_type is query_classification.query_type
    assert plan.rationale
    assert plan.information_needs
    assert plan.decomposition_rationale
    assert plan.requires_reranking is (expected_strategy is RetrievalStrategy.RERANK)


async def test_planner_uses_multi_query_when_contextual_retrieval_is_disabled() -> None:
    plan = await build_planner(contextual_available=False).plan(
        "Explain the complete ingestion workflow.",
        classification(QueryType.BROAD_EXPLANATION),
    )

    assert plan.strategy is RetrievalStrategy.MULTI_QUERY
    assert plan.selected_pipeline_name == MULTI_QUERY_RAG_NAME
    assert "unavailable" in plan.rationale


async def test_low_classification_confidence_selects_reranking() -> None:
    plan = await build_planner().plan(
        "What is the deployment requirement?",
        replace(classification(QueryType.FACTUAL_LOOKUP), confidence=0.4),
    )

    assert plan.strategy is RetrievalStrategy.RERANK
    assert plan.selected_pipeline_name == HYBRID_CROSS_ENCODER_RERANK_RAG_NAME


async def test_compound_broad_question_selects_multi_query_and_preserves_needs() -> None:
    plan = await build_planner().plan(
        "What are the pipeline flows and how do they function?",
        classification(QueryType.BROAD_EXPLANATION),
    )

    assert plan.strategy is RetrievalStrategy.MULTI_QUERY
    assert plan.selected_pipeline_name == MULTI_QUERY_RAG_NAME
    assert [need.need_id for need in plan.information_needs] == ["need_1", "need_2"]
    assert plan.information_needs[1].retrieval_query == "how do they function"
    assert "2 independently gradable information needs" in plan.rationale


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


class StaticEvidenceGrader:
    async def grade(self, question: str, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
        del question
        return EvidenceGradingReport(
            status=EvidenceSufficiency.SUFFICIENT,
            coverage_score=0.9,
            grades=tuple(
                EvidenceGrade(
                    evidence_rank=item.rank,
                    relevance_score=0.9,
                    relevant=True,
                    rationale="Test evidence is relevant.",
                )
                for item in evidence
            ),
            rationale="Test evidence is sufficient.",
            grader_name="test",
        )


async def test_agentic_graph_executes_only_the_planned_retrieval_pipeline() -> None:
    baseline = RecordingRetriever("baseline")
    multi_query = RecordingRetriever("multi_query")
    graph = build_agentic_rag_graph(
        query_classifier=StaticClassifier(classification(QueryType.COMPARISON)),
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
        llm_provider=StaticAnswerLLM(),
    )

    state = await graph.run(QueryState(question="Compare the two retrieval approaches.", top_k=2))

    assert state.pipeline_name == AGENTIC_RAG_NAME
    assert state.retrieval_plan is not None
    assert state.retrieval_plan.selected_pipeline_name == MULTI_QUERY_RAG_NAME
    assert baseline.calls == []
    assert multi_query.calls == [("Compare the two retrieval approaches.", 2)]
    assert state.metadata["retrieval_plan_execution"]["selected_pipeline_name"] == MULTI_QUERY_RAG_NAME
    assert [step.name for step in state.trace] == [
        "select_pipeline",
        "classify_query",
        "plan_retrieval",
        "execute_retrieval_plan",
        "grade_evidence",
        "generate_answer",
    ]
    planning_step = state.trace[2]
    assert planning_step.step_type == "planning"
    assert planning_step.metadata["retrieval_plan"]["strategy"] == "multi_query"
    assert "selected_pipeline=multi_query_rag" in (planning_step.output_summary or "")
    grading_step = state.trace[4]
    assert grading_step.step_type == "evidence_grading"
    assert grading_step.metadata["evidence_grading"]["status"] == "sufficient"
    assert state.retrieved_evidence[0].metadata["evidence_grade"]["relevance_score"] == 0.9


async def test_agentic_graph_applies_rerank_candidate_expansion_for_rerank_plan() -> None:
    retriever = RecordingRetriever("hybrid")
    reranker = RecordingReranker()
    graph = build_agentic_rag_graph(
        query_classifier=StaticClassifier(classification(QueryType.FACTUAL_LOOKUP)),
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
        llm_provider=StaticAnswerLLM(),
    )

    state = await graph.run(
        QueryState(question="Which evidence is most relevant to deployment?", top_k=2),
    )

    assert retriever.calls == [("Which evidence is most relevant to deployment?", 6)]
    assert reranker.candidate_counts == [6]
    assert len(state.retrieved_evidence) == 2
    assert state.metadata["retrieval_plan_execution"]["reranking_applied"] is True
    assert state.metadata["reranking"]["candidate_count"] == 6
