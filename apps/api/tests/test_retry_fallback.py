from __future__ import annotations

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.nodes import RetrievalPlanExecution
from packages.rag_core.pipelines import (
    BASELINE_RAG_NAME,
    CONTEXTUAL_RAG_NAME,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_RAG_NAME,
    MULTI_QUERY_RAG_NAME,
    build_agentic_rag_graph,
)
from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.decomposition import (
    InformationNeed,
    InformationNeedDecomposition,
)
from packages.rag_core.query_understanding.planning import (
    RetrievalPlan,
    RetrievalStrategy,
    RuleBasedRetrievalPlanner,
)
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.retry import (
    RetryAction,
    RetryStopReason,
    RetrievalRetryContext,
    RuleBasedRetrievalRetryPolicy,
)


class StaticClassifier:
    async def classify(self, question: str) -> QueryClassification:
        del question
        return QueryClassification(
            query_type=QueryType.FACTUAL_LOOKUP,
            confidence=0.95,
            needs_metadata_filters=False,
            rationale="Focused factual lookup.",
            classifier_name="test",
        )


class StaticDecomposer:
    async def decompose(self, question: str) -> InformationNeedDecomposition:
        del question
        return InformationNeedDecomposition(
            information_needs=(
                InformationNeed(
                    need_id="need_1",
                    description="Identify the API port and its configuration source.",
                    retrieval_query="API port configuration environment variable service binding",
                ),
            ),
            rationale="One required fact with its configuration source.",
            decomposer_name="test",
        )


class RecordingRetriever:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        self.calls.append((question, top_k))
        return [
            EvidenceItem(
                rank=rank,
                text=f"{self.name} evidence {rank} for {question}",
                score=1.0 / rank,
            )
            for rank in range(1, top_k + 1)
        ]


class SequencedEvidenceGrader:
    def __init__(self, statuses: tuple[EvidenceSufficiency, ...]) -> None:
        self._statuses = statuses
        self.calls = 0

    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
    ) -> EvidenceGradingReport:
        del question
        status = self._statuses[min(self.calls, len(self._statuses) - 1)]
        self.calls += 1
        need = information_needs[0]

        if status is EvidenceSufficiency.MISSING:
            grades = tuple(
                EvidenceGrade(
                    evidence_rank=item.rank,
                    relevance_score=0.1,
                    relevant=False,
                    rationale="The chunk does not establish the required fact.",
                )
                for item in evidence
            )
            need_grade = InformationNeedGrade(
                information_need_id=need.need_id,
                description=need.description,
                status=InformationNeedSupport.MISSING,
                coverage_score=0.0,
                supporting_evidence_ranks=(),
                rationale="No supporting evidence was retrieved.",
            )
            coverage = 0.0
        elif status is EvidenceSufficiency.WEAK:
            grades = tuple(
                EvidenceGrade(
                    evidence_rank=item.rank,
                    relevance_score=0.7 if item.rank == 1 else 0.4,
                    relevant=item.rank == 1,
                    rationale="The first chunk is related but incomplete.",
                    supports_information_need_ids=(need.need_id,) if item.rank == 1 else (),
                )
                for item in evidence
            )
            need_grade = InformationNeedGrade(
                information_need_id=need.need_id,
                description=need.description,
                status=InformationNeedSupport.PARTIAL,
                coverage_score=0.55,
                supporting_evidence_ranks=(1,),
                rationale="The port is suggested but its configuration source is missing.",
            )
            coverage = 0.55
        else:
            grades = tuple(
                EvidenceGrade(
                    evidence_rank=item.rank,
                    relevance_score=0.95 if item.rank == 1 else 0.65,
                    relevant=True,
                    rationale="The evidence supports the required fact.",
                    supports_information_need_ids=(need.need_id,),
                )
                for item in evidence
            )
            need_grade = InformationNeedGrade(
                information_need_id=need.need_id,
                description=need.description,
                status=InformationNeedSupport.SUPPORTED,
                coverage_score=0.95,
                supporting_evidence_ranks=(1,),
                rationale="The API port and configuration source are both established.",
            )
            coverage = 0.95

        return EvidenceGradingReport(
            status=status,
            coverage_score=coverage,
            grades=grades,
            information_need_grades=(need_grade,),
            rationale=f"Evidence is {status.value}.",
            grader_name="sequenced_test_grader",
        )


class RecordingAnswerLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        assert "multi_query evidence" in prompt
        return "The API port is configured by the service environment [1]."


def build_planner() -> RuleBasedRetrievalPlanner:
    return RuleBasedRetrievalPlanner(
        baseline_pipeline_name=BASELINE_RAG_NAME,
        hybrid_pipeline_name=HYBRID_RAG_NAME,
        contextual_pipeline_name=CONTEXTUAL_RAG_NAME,
        multi_query_pipeline_name=MULTI_QUERY_RAG_NAME,
        rerank_pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    )


def build_retry_policy(*, max_retries: int) -> RuleBasedRetrievalRetryPolicy:
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
        max_top_k=10,
        expand_query=True,
    )


def build_executions(
    baseline: RecordingRetriever,
    hybrid: RecordingRetriever,
    multi_query: RecordingRetriever,
) -> dict[str, RetrievalPlanExecution]:
    return {
        BASELINE_RAG_NAME: RetrievalPlanExecution(
            pipeline_name=BASELINE_RAG_NAME,
            pipeline_version="test",
            strategy=RetrievalStrategy.BASELINE,
            retriever=baseline,
        ),
        HYBRID_RAG_NAME: RetrievalPlanExecution(
            pipeline_name=HYBRID_RAG_NAME,
            pipeline_version="test",
            strategy=RetrievalStrategy.HYBRID,
            retriever=hybrid,
        ),
        MULTI_QUERY_RAG_NAME: RetrievalPlanExecution(
            pipeline_name=MULTI_QUERY_RAG_NAME,
            pipeline_version="test",
            strategy=RetrievalStrategy.MULTI_QUERY,
            retriever=multi_query,
        ),
    }


async def test_agentic_retry_escalates_query_top_k_and_pipeline_until_evidence_is_sufficient() -> None:
    baseline = RecordingRetriever("baseline")
    hybrid = RecordingRetriever("hybrid")
    multi_query = RecordingRetriever("multi_query")
    grader = SequencedEvidenceGrader(
        (
            EvidenceSufficiency.MISSING,
            EvidenceSufficiency.WEAK,
            EvidenceSufficiency.SUFFICIENT,
        ),
    )
    llm = RecordingAnswerLLM()
    graph = build_agentic_rag_graph(
        query_classifier=StaticClassifier(),
        information_need_decomposer=StaticDecomposer(),
        retrieval_planner=build_planner(),
        executions=build_executions(baseline, hybrid, multi_query),
        evidence_grader=grader,
        retry_policy=build_retry_policy(max_retries=2),
        llm_provider=llm,
    )

    state = await graph.run(QueryState(question="What port does the API use?", top_k=2))

    expanded_query = (
        "What port does the API use? | "
        "API port configuration environment variable service binding"
    )
    assert baseline.calls == [("What port does the API use?", 2)]
    assert hybrid.calls == [(expanded_query, 4)]
    assert multi_query.calls == [(expanded_query, 8)]
    assert grader.calls == 3
    assert llm.calls == 1
    assert state.evidence_grading is not None and state.evidence_grading.sufficient
    assert state.retrieval_plan is not None
    assert state.retrieval_plan.selected_pipeline_name == BASELINE_RAG_NAME
    assert state.effective_retrieval_plan is not None
    assert state.effective_retrieval_plan.selected_pipeline_name == MULTI_QUERY_RAG_NAME

    report = state.retrieval_retry
    assert report is not None
    assert report.retries_used == 2
    assert report.stop_reason is RetryStopReason.EVIDENCE_SUFFICIENT
    assert report.final_sufficient is True
    assert [attempt.retrieval_plan.strategy for attempt in report.attempts] == [
        RetrievalStrategy.BASELINE,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.MULTI_QUERY,
    ]
    retry_metadata = state.metadata["retrieval_retry"]
    first_retry = retry_metadata["attempts"][1]
    assert first_retry["actions"] == ["expand_query", "increase_top_k", "switch_pipeline"]
    assert "switch_pipeline" in first_retry["decision_rationale"]
    assert retry_metadata["final_top_k"] == 8
    assert retry_metadata["query_changed"] is True

    retry_step = next(step for step in state.trace if step.name == "retry_retrieval")
    assert retry_step.status == "succeeded"
    assert retry_step.metadata["retrieval_retry"]["attempt_count"] == 3
    assert "retries_used=2/2" in (retry_step.output_summary or "")


async def test_agentic_retry_stops_at_limit_and_keeps_generation_blocked() -> None:
    baseline = RecordingRetriever("baseline")
    hybrid = RecordingRetriever("hybrid")
    multi_query = RecordingRetriever("multi_query")
    grader = SequencedEvidenceGrader((EvidenceSufficiency.MISSING,))
    llm = RecordingAnswerLLM()
    graph = build_agentic_rag_graph(
        query_classifier=StaticClassifier(),
        information_need_decomposer=StaticDecomposer(),
        retrieval_planner=build_planner(),
        executions=build_executions(baseline, hybrid, multi_query),
        evidence_grader=grader,
        retry_policy=build_retry_policy(max_retries=1),
        llm_provider=llm,
    )

    state = await graph.run(QueryState(question="What port does the API use?", top_k=2))

    assert baseline.calls == [("What port does the API use?", 2)]
    assert len(hybrid.calls) == 1
    assert multi_query.calls == []
    assert llm.calls == 0
    assert state.retrieval_retry is not None
    assert state.retrieval_retry.retries_used == 1
    assert state.retrieval_retry.stop_reason is RetryStopReason.RETRY_LIMIT_REACHED
    assert state.answer is not None and "not sufficient" in state.answer
    assert state.metadata["answer_blocked_by_evidence_grading"] is True


def test_retry_policy_reports_all_adjustments_for_missing_evidence() -> None:
    policy = build_retry_policy(max_retries=2)
    plan = RetrievalPlan(
        strategy=RetrievalStrategy.BASELINE,
        selected_pipeline_name=BASELINE_RAG_NAME,
        rationale="Initial plan.",
        planner_name="test",
        based_on_query_type=QueryType.FACTUAL_LOOKUP,
        target_information_need_ids=("need_1",),
    )
    report = EvidenceGradingReport(
        status=EvidenceSufficiency.MISSING,
        coverage_score=0.0,
        grades=(),
        information_need_grades=(
            InformationNeedGrade(
                information_need_id="need_1",
                description="Find the port.",
                status=InformationNeedSupport.MISSING,
                coverage_score=0.0,
                supporting_evidence_ranks=(),
                rationale="Missing.",
            ),
        ),
        rationale="Missing.",
        grader_name="test",
    )
    decision = policy.decide(
        RetrievalRetryContext(
            original_question="What port is used?",
            current_query="What port is used?",
            current_top_k=2,
            current_plan=plan,
            evidence_grading=report,
            retries_used=0,
            attempted_strategies=(RetrievalStrategy.BASELINE,),
            unresolved_retrieval_queries=("API port environment setting",),
            available_pipeline_names=(BASELINE_RAG_NAME, HYBRID_RAG_NAME),
        ),
    )

    assert decision.should_retry is True
    assert decision.actions == (
        RetryAction.EXPAND_QUERY,
        RetryAction.INCREASE_TOP_K,
        RetryAction.SWITCH_PIPELINE,
    )
