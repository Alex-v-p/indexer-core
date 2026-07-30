from __future__ import annotations

from collections.abc import Callable

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.information_need_graph.models import InformationNeedExecution
from packages.rag_core.agents.information_need_graph.tracing import active_need_classification_summary
from packages.rag_core.agents.shared.retrieval import RetrievalPlanExecution
from packages.rag_core.pipelines import (
    BASELINE_RAG_NAME,
    CONTEXTUAL_RAG_NAME,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_RAG_NAME,
    MULTI_QUERY_RAG_NAME,
    build_agentic_rag_graph,
)
from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.decomposition import InformationNeed, InformationNeedDecomposition
from packages.rag_core.query_understanding.planning import RetrievalStrategy, RuleBasedRetrievalPlanner
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.retry import RuleBasedRetrievalRetryPolicy


def query_classification(query_type: QueryType, *, confidence: float = 0.95) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        confidence=confidence,
        needs_metadata_filters=False,
        rationale="Test classification.",
        classifier_name="test",
    )


class MappingClassifier:
    def __init__(self, resolver: Callable[[str, int], QueryClassification]) -> None:
        self._resolver = resolver
        self.calls: list[str] = []

    async def classify(self, question: str) -> QueryClassification:
        self.calls.append(question)
        return self._resolver(question, len(self.calls))


class StaticDecomposer:
    def __init__(self, needs: tuple[InformationNeed, ...]) -> None:
        self._needs = needs

    async def decompose(self, question: str) -> InformationNeedDecomposition:
        del question
        return InformationNeedDecomposition(
            information_needs=self._needs,
            rationale="Test decomposition.",
            decomposer_name="test",
        )


class RecordingRetriever:
    def __init__(self, name: str, responder: Callable[[str, int], list[EvidenceItem]]) -> None:
        self.name = name
        self._responder = responder
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        self.calls.append((question, top_k))
        return self._responder(question, top_k)


class KeywordEvidenceGrader:
    def __init__(self, required_terms: dict[str, str]) -> None:
        self._required_terms = required_terms
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
    ) -> EvidenceGradingReport:
        del question
        need = information_needs[0]
        self.calls.append((need.need_id, tuple(item.rank for item in evidence)))
        term = self._required_terms[need.need_id].lower()
        supporting = tuple(item.rank for item in evidence if term in item.text.lower())
        grades = tuple(
            EvidenceGrade(
                evidence_rank=item.rank,
                relevance_score=0.95 if item.rank in supporting else 0.05,
                relevant=item.rank in supporting,
                rationale="Contains the required fact." if item.rank in supporting else "Does not support this item.",
                supports_information_need_ids=(need.need_id,) if item.rank in supporting else (),
            )
            for item in evidence
        )
        supported = bool(supporting)
        need_grade = InformationNeedGrade(
            information_need_id=need.need_id,
            description=need.description,
            status=InformationNeedSupport.SUPPORTED if supported else InformationNeedSupport.MISSING,
            coverage_score=0.95 if supported else 0.0,
            supporting_evidence_ranks=supporting,
            rationale="The required fact is present." if supported else "The required fact is absent.",
            required=need.required,
        )
        return EvidenceGradingReport(
            status=EvidenceSufficiency.SUFFICIENT if supported else EvidenceSufficiency.MISSING,
            coverage_score=need_grade.coverage_score,
            grades=grades,
            information_need_grades=(need_grade,),
            rationale=need_grade.rationale,
            grader_name="keyword_test_grader",
        )


class RecordingAnswerLLM:
    def __init__(self, answer: str = "Supported answer [1].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def build_planner() -> RuleBasedRetrievalPlanner:
    return RuleBasedRetrievalPlanner(
        baseline_pipeline_name=BASELINE_RAG_NAME,
        hybrid_pipeline_name=HYBRID_RAG_NAME,
        contextual_pipeline_name=CONTEXTUAL_RAG_NAME,
        multi_query_pipeline_name=MULTI_QUERY_RAG_NAME,
        rerank_pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        top_k_multiplier=2.0,
        max_top_k=20,
        expand_query=True,
    )


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


def execution(name: str, strategy: RetrievalStrategy, retriever: RecordingRetriever) -> RetrievalPlanExecution:
    return RetrievalPlanExecution(
        pipeline_name=name,
        pipeline_version="test",
        strategy=strategy,
        retriever=retriever,
    )


def test_classification_trace_summary_tolerates_missing_legacy_source_history() -> None:
    need = InformationNeed("need_1", "Identify the API port.", "API port")
    classification = query_classification(QueryType.FACTUAL_LOOKUP)
    execution_state = InformationNeedExecution(
        information_need=need,
        max_attempts=2,
        classification=classification,
        classification_history=[classification],
    )
    state = QueryState(
        question="Which port does the API use?",
        information_need_executions={need.need_id: execution_state},
        active_information_need_id=need.need_id,
    )

    summary = active_need_classification_summary(state)

    assert "source=unknown" in summary
    assert "classification_count=1" in summary


async def test_exact_single_need_reuses_top_level_classification_with_normalized_query() -> None:
    retriever = RecordingRetriever(
        "baseline",
        lambda query, top_k: [EvidenceItem(rank=1, text="The API listens on port 8000.")],
    )
    classifier = MappingClassifier(
        lambda question, call: query_classification(QueryType.FACTUAL_LOOKUP),
    )
    question = "What Port Does The API Use?"
    graph = build_agentic_rag_graph(
        query_classifier=classifier,
        information_need_decomposer=StaticDecomposer(
            (
                InformationNeed(
                    need_id="need_port",
                    description="Identify the API port.",
                    retrieval_query="  what   port does the api use?  ",
                ),
            ),
        ),
        retrieval_planner=build_planner(),
        executions={
            BASELINE_RAG_NAME: execution(
                BASELINE_RAG_NAME,
                RetrievalStrategy.BASELINE,
                retriever,
            ),
        },
        evidence_grader=KeywordEvidenceGrader({"need_port": "8000"}),
        retry_policy=build_retry_policy(),
        llm_provider=RecordingAnswerLLM("The API listens on port 8000 [1]."),
    )

    state = await graph.run(QueryState(question=question, top_k=2))

    execution_state = state.information_need_executions["need_port"]
    assert classifier.calls == [question]
    assert execution_state.classification is state.query_classification
    assert execution_state.classification_source_history == ["top_level_reuse"]
    assert execution_state.reclassifications_used == 0
    classification_step = next(
        step for step in state.trace if step.name == "classify_information_need"
    )
    assert classification_step.metadata["information_need_execution"][
        "classification_source_history"
    ] == ["top_level_reuse"]
    assert "source=top_level_reuse" in (classification_step.output_summary or "")


async def test_information_need_subgraph_replans_and_retries_only_the_active_item() -> None:
    baseline = RecordingRetriever(
        "baseline",
        lambda query, top_k: [EvidenceItem(rank=1, text="Unrelated baseline result.")],
    )
    hybrid = RecordingRetriever(
        "hybrid",
        lambda query, top_k: [EvidenceItem(rank=1, text="The API listens on port 8000.")],
    )
    classifier = MappingClassifier(lambda question, call: query_classification(QueryType.FACTUAL_LOOKUP))
    llm = RecordingAnswerLLM("The API listens on port 8000 [1].")
    graph = build_agentic_rag_graph(
        query_classifier=classifier,
        information_need_decomposer=StaticDecomposer(
            (
                InformationNeed(
                    need_id="need_port",
                    description="Identify the API port.",
                    retrieval_query="API listening port",
                ),
            ),
        ),
        retrieval_planner=build_planner(),
        executions={
            BASELINE_RAG_NAME: execution(BASELINE_RAG_NAME, RetrievalStrategy.BASELINE, baseline),
            HYBRID_RAG_NAME: execution(HYBRID_RAG_NAME, RetrievalStrategy.HYBRID, hybrid),
        },
        evidence_grader=KeywordEvidenceGrader({"need_port": "8000"}),
        retry_policy=build_retry_policy(max_retries=2),
        llm_provider=llm,
        max_retries_per_information_need=2,
        max_total_retrieval_attempts=6,
    )

    state = await graph.run(QueryState(question="What port does the API use?", top_k=2))

    need = state.information_need_executions["need_port"]
    assert baseline.calls == [("API listening port", 2)]
    assert len(hybrid.calls) == 1
    retry_query, retry_top_k = hybrid.calls[0]
    assert retry_top_k == 4
    assert "API listening port" in retry_query
    assert retry_query == "API listening port"
    assert "previous evidence gap" not in retry_query
    assert "answer requirement" not in retry_query
    assert need.attempts_used == 2
    assert [plan.strategy for plan in need.plan_history] == [RetrievalStrategy.BASELINE, RetrievalStrategy.HYBRID]
    assert need.status.value == "supported"
    assert state.information_need_resolution is not None
    assert state.information_need_resolution.total_retrieval_attempts == 2
    assert state.information_need_resolution.complete is True
    assert [step.name for step in state.trace].count("plan_information_need") == 2
    assert [step.name for step in state.trace].count("grade_information_need") == 2
    assert [item.text for item in state.retrieved_evidence] == ["The API listens on port 8000."]
    assert classifier.calls == ["What port does the API use?", "API listening port"]
    assert need.classification_source_history == ["model"]


async def test_each_information_need_gets_independent_classification_plan_and_retry_budget() -> None:
    baseline = RecordingRetriever(
        "baseline",
        lambda query, top_k: [EvidenceItem(rank=1, text="No person identity is documented.")],
    )
    hybrid = RecordingRetriever(
        "hybrid",
        lambda query, top_k: [EvidenceItem(rank=1, text="Still no identity information.")],
    )
    multi_query = RecordingRetriever(
        "multi_query",
        lambda query, top_k: [EvidenceItem(rank=1, text="No evidence about Alex.")],
    )
    contextual = RecordingRetriever(
        "contextual",
        lambda query, top_k: [
            EvidenceItem(rank=1, text="Indexer Core is an agentic retrieval and evaluation project."),
        ],
    )

    def classify(question: str, call: int) -> QueryClassification:
        del call
        return query_classification(
            QueryType.BROAD_EXPLANATION if "project" in question.lower() else QueryType.FACTUAL_LOOKUP,
        )

    llm = RecordingAnswerLLM("Indexer Core is an agentic retrieval project [1].")
    classifier = MappingClassifier(classify)
    graph = build_agentic_rag_graph(
        query_classifier=classifier,
        information_need_decomposer=StaticDecomposer(
            (
                InformationNeed(
                    need_id="need_project",
                    description="Explain what the project is.",
                    retrieval_query="Explain the Indexer Core project architecture and purpose",
                ),
                InformationNeed(
                    need_id="need_alex",
                    description="Identify who Alex is.",
                    retrieval_query="Who is Alex",
                ),
            ),
        ),
        retrieval_planner=build_planner(),
        executions={
            BASELINE_RAG_NAME: execution(BASELINE_RAG_NAME, RetrievalStrategy.BASELINE, baseline),
            HYBRID_RAG_NAME: execution(HYBRID_RAG_NAME, RetrievalStrategy.HYBRID, hybrid),
            CONTEXTUAL_RAG_NAME: execution(CONTEXTUAL_RAG_NAME, RetrievalStrategy.CONTEXTUAL, contextual),
            MULTI_QUERY_RAG_NAME: execution(MULTI_QUERY_RAG_NAME, RetrievalStrategy.MULTI_QUERY, multi_query),
        },
        evidence_grader=KeywordEvidenceGrader(
            {
                "need_project": "agentic retrieval",
                "need_alex": "alex is",
            },
        ),
        retry_policy=build_retry_policy(max_retries=2),
        llm_provider=llm,
        max_retries_per_information_need=2,
        max_total_retrieval_attempts=8,
    )

    state = await graph.run(QueryState(question="Tell me about the project and who is Alex.", top_k=2))

    project = state.information_need_executions["need_project"]
    alex = state.information_need_executions["need_alex"]
    assert [plan.strategy for plan in project.plan_history] == [RetrievalStrategy.CONTEXTUAL]
    assert project.attempts_used == 1
    assert project.status.value == "supported"
    assert [plan.strategy for plan in alex.plan_history] == [
        RetrievalStrategy.BASELINE,
        RetrievalStrategy.HYBRID,
        RetrievalStrategy.MULTI_QUERY,
    ]
    assert alex.attempts_used == 3
    assert alex.status.value == "exhausted"
    assert state.information_need_resolution is not None
    assert state.information_need_resolution.supported_information_need_ids == ("need_project",)
    assert state.information_need_resolution.unresolved_information_need_ids == ("need_alex",)
    assert state.metadata["answer_is_partial"] is True
    assert "Identify who Alex is." in (state.answer or "")
    assert [item.text for item in state.retrieved_evidence] == [
        "Indexer Core is an agentic retrieval and evaluation project.",
    ]
    assert llm.prompts and "Supported required claims" in llm.prompts[0]
    assert classifier.calls == [
        "Tell me about the project and who is Alex.",
        "Explain the Indexer Core project architecture and purpose",
        "Who is Alex",
    ]
    assert project.classification_source_history == ["model"]
    assert alex.classification_source_history == ["model"]


async def test_low_confidence_missing_item_routes_back_through_classification_before_retry() -> None:
    baseline = RecordingRetriever("baseline", lambda query, top_k: [EvidenceItem(rank=1, text="Irrelevant.")])
    hybrid = RecordingRetriever("hybrid", lambda query, top_k: [EvidenceItem(rank=1, text="Release 2026 is current.")])

    def classify(question: str, call: int) -> QueryClassification:
        if question == "Which release is current?" and call == 1:
            return query_classification(QueryType.FACTUAL_LOOKUP, confidence=0.4)
        if question == "Which release is current?":
            return query_classification(QueryType.VERSION_SPECIFIC, confidence=0.95)
        return query_classification(QueryType.FACTUAL_LOOKUP)

    classifier = MappingClassifier(classify)
    graph = build_agentic_rag_graph(
        query_classifier=classifier,
        information_need_decomposer=StaticDecomposer(
            (
                InformationNeed(
                    need_id="need_release",
                    description="Identify the current release.",
                    retrieval_query="Which release is current?",
                ),
            ),
        ),
        retrieval_planner=build_planner(),
        executions={
            BASELINE_RAG_NAME: execution(BASELINE_RAG_NAME, RetrievalStrategy.BASELINE, baseline),
            HYBRID_RAG_NAME: execution(HYBRID_RAG_NAME, RetrievalStrategy.HYBRID, hybrid),
        },
        evidence_grader=KeywordEvidenceGrader({"need_release": "release 2026"}),
        retry_policy=build_retry_policy(max_retries=2),
        llm_provider=RecordingAnswerLLM("Release 2026 is current [1]."),
        max_retries_per_information_need=2,
        max_total_retrieval_attempts=5,
        max_reclassifications_per_information_need=1,
    )

    state = await graph.run(QueryState(question="Which release is current?", top_k=2))

    execution_state = state.information_need_executions["need_release"]
    assert len(execution_state.classification_history) == 2
    assert execution_state.classification_source_history == ["top_level_reuse", "model"]
    assert execution_state.reclassifications_used == 1
    assert classifier.calls == ["Which release is current?", "Which release is current?"]
    assert [step.name for step in state.trace].count("classify_information_need") == 2
    classify_steps = [step for step in state.trace if step.name == "classify_information_need"]
    assert classify_steps[0].metadata["information_need_execution"][
        "classification_source_history"
    ] == ["top_level_reuse"]
    assert classify_steps[1].metadata["information_need_execution"][
        "classification_source_history"
    ] == ["top_level_reuse", "model"]
    assert execution_state.status.value == "supported"


async def test_global_attempt_budget_exhausts_remaining_items_without_unbounded_loop() -> None:
    retriever = RecordingRetriever("baseline", lambda query, top_k: [EvidenceItem(rank=1, text="Irrelevant.")])
    graph = build_agentic_rag_graph(
        query_classifier=MappingClassifier(lambda question, call: query_classification(QueryType.FACTUAL_LOOKUP)),
        information_need_decomposer=StaticDecomposer(
            (
                InformationNeed("need_1", "Find fact one.", "fact one"),
                InformationNeed("need_2", "Find fact two.", "fact two"),
            ),
        ),
        retrieval_planner=build_planner(),
        executions={BASELINE_RAG_NAME: execution(BASELINE_RAG_NAME, RetrievalStrategy.BASELINE, retriever)},
        evidence_grader=KeywordEvidenceGrader({"need_1": "never", "need_2": "never"}),
        retry_policy=build_retry_policy(max_retries=2),
        llm_provider=RecordingAnswerLLM(),
        max_retries_per_information_need=2,
        max_total_retrieval_attempts=2,
    )

    state = await graph.run(QueryState(question="Find both facts.", top_k=1))

    assert state.total_information_need_retrieval_attempts == 2
    assert state.information_need_resolution is not None
    assert state.information_need_resolution.complete is False
    assert all(
        execution_state.status.value == "exhausted"
        for execution_state in state.information_need_executions.values()
    )
    assert len(state.trace) < 40
