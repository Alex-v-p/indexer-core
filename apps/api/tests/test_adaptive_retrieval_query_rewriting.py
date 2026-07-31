from __future__ import annotations

from packages.rag_core.pipelines import (
    BASELINE_RAG_NAME,
    CONTEXTUAL_RAG_NAME,
    HIERARCHICAL_RAG_NAME,
    HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
    HYBRID_RAG_NAME,
    MULTI_QUERY_RAG_NAME,
)
from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.decomposition import (
    InformationNeed,
    parse_information_need_decomposition,
)
from packages.rag_core.query_understanding.planning import (
    InformationNeedPlanningContext,
    LLMRetrievalQueryRewriter,
    RetrievalAttemptEvidenceFeedback,
    RetrievalAttemptFeedback,
    RetrievalStrategy,
    RuleBasedRetrievalPlanner,
)
from packages.rag_core.retrieval.graders import InformationNeedGrade, InformationNeedSupport


class RecordingStructuredLLM:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[0]

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        self.prompts.append(prompt)
        self.schemas.append(response_schema)
        return self.responses.pop(0)


def _classification() -> QueryClassification:
    return QueryClassification(
        query_type=QueryType.BROAD_EXPLANATION,
        confidence=0.95,
        needs_metadata_filters=False,
        rationale="Broad project explanation.",
        classifier_name="test",
    )


def _planner(rewriter: LLMRetrievalQueryRewriter) -> RuleBasedRetrievalPlanner:
    return RuleBasedRetrievalPlanner(
        baseline_pipeline_name=BASELINE_RAG_NAME,
        hybrid_pipeline_name=HYBRID_RAG_NAME,
        contextual_pipeline_name=CONTEXTUAL_RAG_NAME,
        multi_query_pipeline_name=MULTI_QUERY_RAG_NAME,
        rerank_pipeline_name=HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
        hierarchical_pipeline_name=HIERARCHICAL_RAG_NAME,
        contextual_available=True,
        hierarchical_available=True,
        query_rewriter=rewriter,
    )


def _grade(status: InformationNeedSupport, coverage: float, rationale: str) -> InformationNeedGrade:
    return InformationNeedGrade(
        information_need_id="need_overview",
        description="Explain the LLMguidance project purpose, scope, and overview.",
        status=status,
        coverage_score=coverage,
        supporting_evidence_ranks=(() if status is InformationNeedSupport.MISSING else (1,)),
        rationale=rationale,
    )


def test_decomposition_repairs_context_free_project_overview_query() -> None:
    result = parse_information_need_decomposition(
        '''{
          "information_needs": [
            {
              "description": "Explain the project overview.",
              "retrieval_query": "project overview",
              "subject_context": "project"
            }
          ],
          "rationale": "The overview is one atomic requirement."
        }''',
        original_question="Tell me about the LLMguidance project and key points of it.",
    )

    need = result.information_needs[0]
    assert need.subject_context == "LLMguidance project"
    assert need.retrieval_query == "LLMguidance project overview"


async def test_retry_rewriter_uses_previous_evidence_and_improves_each_attempt() -> None:
    llm = RecordingStructuredLLM(
        '''{
          "rewritten_query": "LLMguidance project purpose scope problem statement",
          "failure_mode": "query_too_generic",
          "missing_aspects": ["project purpose", "project scope"],
          "rationale": "The earlier query retrieved unrelated projects and omitted the project identity."
        }''',
        '''{
          "rewritten_query": "LLMguidance project goals intended users expected outcomes",
          "failure_mode": "partial_coverage",
          "missing_aspects": ["intended users", "expected outcomes"],
          "rationale": "Architecture evidence was useful, but the project goals and audience remain unsupported."
        }''',
    )
    rewriter = LLMRetrievalQueryRewriter(llm_provider=llm, fail_open=False)
    planner = _planner(rewriter)
    need = InformationNeed(
        need_id="need_overview",
        description="Explain the LLMguidance project purpose, scope, and overview.",
        retrieval_query="project overview",
        subject_context="LLMguidance project",
    )
    classification = _classification()

    first = await planner.plan_information_need(
        InformationNeedPlanningContext(
            original_question="Tell me about the LLMguidance project and key points of it.",
            information_need=need,
            classification=classification,
            previous_grade=None,
            previous_plans=(),
            previous_queries=(),
            previous_attempts=(),
            available_pipeline_names=(
                HIERARCHICAL_RAG_NAME,
                CONTEXTUAL_RAG_NAME,
                HYBRID_RAG_NAME,
                MULTI_QUERY_RAG_NAME,
                HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
                BASELINE_RAG_NAME,
            ),
            attempts_used=0,
            max_attempts=3,
            current_top_k=5,
        ),
    )
    assert first.query == "LLMguidance project overview"
    assert first.query_rewrite is None

    first_feedback = RetrievalAttemptFeedback(
        attempt_number=1,
        query=first.query,
        pipeline_name=first.selected_pipeline_name,
        strategy=first.strategy,
        top_k=first.top_k,
        grade_status="missing",
        coverage_score=0.0,
        grading_rationale="The chunks describe unrelated projects and do not identify LLMguidance.",
        evidence=(
            RetrievalAttemptEvidenceFeedback(
                text="This project overview describes a warehouse migration initiative.",
                document_name="WarehouseMigration.md",
                relevant=False,
                relevance_score=0.05,
                grading_rationale="Wrong project.",
            ),
        ),
    )
    second_grade = _grade(
        InformationNeedSupport.MISSING,
        0.0,
        "The chunks describe unrelated projects and do not identify LLMguidance.",
    )
    second = await planner.plan_information_need(
        InformationNeedPlanningContext(
            original_question="Tell me about the LLMguidance project and key points of it.",
            information_need=need,
            classification=classification,
            previous_grade=second_grade,
            previous_plans=(first,),
            previous_queries=(first.query,),
            previous_attempts=(first_feedback,),
            available_pipeline_names=(
                HIERARCHICAL_RAG_NAME,
                CONTEXTUAL_RAG_NAME,
                HYBRID_RAG_NAME,
                MULTI_QUERY_RAG_NAME,
                HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
                BASELINE_RAG_NAME,
            ),
            attempts_used=1,
            max_attempts=3,
            current_top_k=first.top_k,
        ),
    )
    assert second.query == "LLMguidance project purpose scope problem statement"
    assert second.query_rewrite is not None
    assert second.query_rewrite.failure_mode == "query_too_generic"
    assert "adaptive_query_rewrite" in second.adjustments
    assert "WarehouseMigration.md" in llm.prompts[0]
    assert "Wrong project." in llm.prompts[0]

    second_feedback = RetrievalAttemptFeedback(
        attempt_number=2,
        query=second.query,
        pipeline_name=second.selected_pipeline_name,
        strategy=second.strategy,
        top_k=second.top_k,
        grade_status="partial",
        coverage_score=0.45,
        grading_rationale="The architecture is described, but goals and intended users are missing.",
        evidence=(
            RetrievalAttemptEvidenceFeedback(
                text="LLMguidance uses a local retrieval and generation architecture.",
                document_name="LLMguidance-architecture.md",
                relevant=True,
                relevance_score=0.82,
                grading_rationale="Useful architecture evidence, incomplete overview.",
            ),
        ),
    )
    third_grade = _grade(
        InformationNeedSupport.PARTIAL,
        0.45,
        "The architecture is described, but goals and intended users are missing.",
    )
    third = await planner.plan_information_need(
        InformationNeedPlanningContext(
            original_question="Tell me about the LLMguidance project and key points of it.",
            information_need=need,
            classification=classification,
            previous_grade=third_grade,
            previous_plans=(first, second),
            previous_queries=(first.query, second.query),
            previous_attempts=(first_feedback, second_feedback),
            available_pipeline_names=(
                HIERARCHICAL_RAG_NAME,
                CONTEXTUAL_RAG_NAME,
                HYBRID_RAG_NAME,
                MULTI_QUERY_RAG_NAME,
                HYBRID_CROSS_ENCODER_RERANK_RAG_NAME,
                BASELINE_RAG_NAME,
            ),
            attempts_used=2,
            max_attempts=3,
            current_top_k=second.top_k,
        ),
    )
    assert third.query == "LLMguidance project goals intended users expected outcomes"
    assert third.query not in {first.query, second.query}
    assert third.query_rewrite is not None
    assert third.query_rewrite.failure_mode == "partial_coverage"
    assert "LLMguidance-architecture.md" in llm.prompts[1]
    assert "goals and intended users are missing" in llm.prompts[1]


async def test_retry_rewriter_removes_sibling_intent_without_removing_shared_subject() -> None:
    llm = RecordingStructuredLLM(
        '''{
          "rewritten_query": "who was the main contributor LLMguidance LLM guidance project key points architecture",
          "failure_mode": "partial_coverage",
          "missing_aspects": ["project architecture"],
          "rationale": "The earlier result lacked architectural key points."
        }''',
    )
    rewriter = LLMRetrievalQueryRewriter(llm_provider=llm, fail_open=False)
    active_need = InformationNeed(
        need_id="need_key_points",
        description="Summarize the key points of the LLM guidance project.",
        retrieval_query="LLMguidance LLM guidance project key points",
        subject_context="LLMguidance LLM guidance project",
    )
    sibling_need = InformationNeed(
        need_id="need_contributor",
        description="Identify the main contributor to the LLM guidance project.",
        retrieval_query="LLMguidance LLM guidance project main contributor",
        subject_context="LLMguidance LLM guidance project",
    )
    previous_attempt = RetrievalAttemptFeedback(
        attempt_number=1,
        query=active_need.retrieval_query,
        pipeline_name=HIERARCHICAL_RAG_NAME,
        strategy=RetrievalStrategy.HIERARCHICAL,
        top_k=5,
        grade_status="partial",
        coverage_score=0.5,
        grading_rationale="The project identity is correct but architecture details are missing.",
    )
    context = InformationNeedPlanningContext(
        original_question=(
            "Who was the main contributor to the LLMguidance project and what are its key points?"
        ),
        information_need=active_need,
        classification=_classification(),
        previous_grade=_grade(
            InformationNeedSupport.PARTIAL,
            0.5,
            "The project identity is correct but architecture details are missing.",
        ),
        previous_plans=(),
        previous_queries=(active_need.retrieval_query,),
        previous_attempts=(previous_attempt,),
        available_pipeline_names=(HIERARCHICAL_RAG_NAME,),
        attempts_used=1,
        max_attempts=3,
        current_top_k=5,
        sibling_information_needs=(sibling_need,),
    )

    result = await rewriter.rewrite(context)

    assert result.query == "LLMguidance LLM guidance project key points architecture"
    assert "main contributor" not in result.query.casefold()
    assert "llmguidance" in result.query.casefold()
    assert "llm guidance project" in result.query.casefold()
    assert "excluded_sibling_information_needs" in llm.prompts[0]
    assert "Identify the main contributor" in llm.prompts[0]


async def test_initial_planner_defensively_isolates_sibling_lane_intent() -> None:
    planner = _planner(
        LLMRetrievalQueryRewriter(
            llm_provider=RecordingStructuredLLM(),
            fail_open=False,
        ),
    )
    active_need = InformationNeed(
        need_id="need_key_points",
        description="Summarize the key points of the LLM guidance project.",
        retrieval_query=(
            "who was the main contributor LLMguidance LLM guidance project key points"
        ),
        subject_context="LLMguidance LLM guidance project",
    )
    sibling_need = InformationNeed(
        need_id="need_contributor",
        description="Identify the main contributor to the LLM guidance project.",
        retrieval_query="LLMguidance LLM guidance project main contributor",
        subject_context="LLMguidance LLM guidance project",
    )

    plan = await planner.plan_information_need(
        InformationNeedPlanningContext(
            original_question=(
                "Who was the main contributor to the LLMguidance project and what are its key points?"
            ),
            information_need=active_need,
            classification=_classification(),
            previous_grade=None,
            previous_plans=(),
            previous_queries=(),
            previous_attempts=(),
            available_pipeline_names=(HIERARCHICAL_RAG_NAME,),
            attempts_used=0,
            max_attempts=3,
            current_top_k=5,
            sibling_information_needs=(sibling_need,),
        ),
    )

    assert plan.query == "LLMguidance LLM guidance project key points"
