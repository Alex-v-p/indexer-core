from __future__ import annotations

import uuid

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.shared.retrieval import RetrievalPlanExecution
from packages.rag_core.documents import DocumentPreference, document_reference_from_values
from packages.rag_core.pipelines import BASELINE_RAG_NAME, build_agentic_rag_graph
from packages.rag_core.query_understanding.classification import QueryClassification, QueryType
from packages.rag_core.query_understanding.decomposition import (
    InformationNeed,
    InformationNeedDecomposition,
)
from packages.rag_core.query_understanding.planning import RetrievalStrategy, RuleBasedRetrievalPlanner
from packages.rag_core.retrieval.document_selection import (
    DocumentBalancedCandidateSelector,
    RuleBasedPrimaryDocumentDetector,
)
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem
from packages.rag_core.retrieval.retry import RuleBasedRetrievalRetryPolicy

FUNCTIONAL_ID = uuid.uuid4()
REALIZATION_ID = uuid.uuid4()


def _item(rank: int, document_id: uuid.UUID, filename: str, text: str) -> EvidenceItem:
    return EvidenceItem(
        rank=rank,
        text=text,
        score=1.0 - rank * 0.01,
        document_id=document_id,
        document_version_id=uuid.uuid4(),
        metadata={"original_filename": filename},
    )


def _grading(need: InformationNeed, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
    ranks = tuple(item.rank for item in evidence)
    return EvidenceGradingReport(
        status=EvidenceSufficiency.SUFFICIENT,
        coverage_score=0.9,
        grades=tuple(
            EvidenceGrade(
                evidence_rank=item.rank,
                relevance_score=0.9,
                relevant=True,
                rationale="Direct project evidence.",
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
                rationale="The need is supported.",
            ),
        ),
        rationale="Relevant project evidence was found.",
        grader_name="test",
    )


def test_primary_document_detector_prefers_title_overlap_without_hard_filtering() -> None:
    need = InformationNeed(
        "need_document",
        "Identify the project Alex made for DAF.",
        "Alex DAF project",
    )
    evidence = [
        _item(1, REALIZATION_ID, "Realization_Draft4.pdf", "The work followed three phases."),
        _item(2, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "Dashboarding requirements for DAF."),
    ]

    preference = RuleBasedPrimaryDocumentDetector().detect(
        question="What was the project Alex made for DAF?",
        information_need=need,
        evidence=evidence,
        grading=_grading(need, evidence),
    )

    assert preference is not None
    assert preference.document.document_id == FUNCTIONAL_ID
    assert preference.document.display_name == "FunctionalSpecDAF_AVP.pdf"
    assert preference.to_metadata()["semantics"] == "soft_preference_not_filter"


def test_document_balancing_makes_primary_dominate_but_keeps_supporting_documents() -> None:
    primary_example = _item(1, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "Primary evidence 1")
    reference = document_reference_from_values(
        rank=primary_example.rank,
        document_id=primary_example.document_id,
        document_version_id=primary_example.document_version_id,
        metadata=primary_example.metadata,
    )
    preference = DocumentPreference(
        document=reference,
        score=0.9,
        confidence=0.9,
        margin=0.2,
        supporting_information_need_ids=("need_1",),
        supporting_evidence_ranks=(1,),
        rationale="Strong document match.",
        detector_name="test",
    )
    candidates = [
        _item(1, REALIZATION_ID, "Realization_Draft4.pdf", "Secondary 1"),
        _item(2, REALIZATION_ID, "Realization_Draft4.pdf", "Secondary 2"),
        _item(3, REALIZATION_ID, "Realization_Draft4.pdf", "Secondary 3"),
        _item(4, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "Primary 1"),
        _item(5, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "Primary 2"),
        _item(6, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "Primary 3"),
        _item(7, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "Primary 4"),
    ]

    selection = DocumentBalancedCandidateSelector().select(
        candidates,
        top_k=5,
        preference=preference,
    )

    counts = dict(selection.selected_document_counts)
    assert counts[f"document:{FUNCTIONAL_ID}"] == 3
    assert counts[f"document:{REALIZATION_ID}"] == 2
    assert selection.primary_document_key == f"document:{FUNCTIONAL_ID}"
    assert [item.text for item in selection.evidence] == [
        "Primary 1",
        "Primary 2",
        "Primary 3",
        "Secondary 1",
        "Secondary 2",
    ]


class _Classifier:
    async def classify(self, question: str) -> QueryClassification:
        del question
        return QueryClassification(
            query_type=QueryType.FACTUAL_LOOKUP,
            confidence=0.95,
            needs_metadata_filters=False,
            rationale="Focused lookup.",
            classifier_name="test",
        )


class _Decomposer:
    async def decompose(self, question: str) -> InformationNeedDecomposition:
        del question
        return InformationNeedDecomposition(
            information_needs=(
                InformationNeed("need_1", "Identify the DAF project document.", "Alex DAF project"),
                InformationNeed("need_2", "Describe the DAF project functionality.", "DAF project functionality"),
            ),
            rationale="Resolve the document before its details.",
            decomposer_name="test",
        )


class _Retriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        self.calls.append((question, top_k))
        if question == "Alex DAF project":
            return [
                _item(1, REALIZATION_ID, "Realization_Draft4.pdf", "Generic project phases."),
                _item(2, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "DAF dashboarding specification."),
            ]
        return [
            _item(1, REALIZATION_ID, "Realization_Draft4.pdf", "Generic phase one."),
            _item(2, REALIZATION_ID, "Realization_Draft4.pdf", "Generic phase two."),
            _item(3, REALIZATION_ID, "Realization_Draft4.pdf", "Generic phase three."),
            _item(4, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "DAF dashboard feature one."),
            _item(5, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "DAF dashboard feature two."),
            _item(6, FUNCTIONAL_ID, "FunctionalSpecDAF_AVP.pdf", "DAF dashboard feature three."),
        ][:top_k]


class _Grader:
    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
        **kwargs,
    ) -> EvidenceGradingReport:
        del question, kwargs
        return _grading(information_needs[0], evidence)


class _AnswerLLM:
    async def generate(self, prompt: str) -> str:
        del prompt
        return "The DAF project was a dashboarding solution [1]."


async def test_later_information_need_inherits_primary_preference_and_balanced_pool() -> None:
    retriever = _Retriever()
    planner = RuleBasedRetrievalPlanner(
        baseline_pipeline_name=BASELINE_RAG_NAME,
        hybrid_pipeline_name="hybrid_rag",
        contextual_pipeline_name="contextual_rag",
        multi_query_pipeline_name="multi_query_rag",
        rerank_pipeline_name="hybrid_cross_encoder_rerank_rag",
        hierarchical_pipeline_name="hierarchical_rag",
    )
    graph = build_agentic_rag_graph(
        query_classifier=_Classifier(),
        information_need_decomposer=_Decomposer(),
        retrieval_planner=planner,
        executions={
            BASELINE_RAG_NAME: RetrievalPlanExecution(
                pipeline_name=BASELINE_RAG_NAME,
                pipeline_version="test",
                strategy=RetrievalStrategy.BASELINE,
                retriever=retriever,
            ),
        },
        evidence_grader=_Grader(),
        retry_policy=RuleBasedRetrievalRetryPolicy(
            pipeline_names={RetrievalStrategy.BASELINE: BASELINE_RAG_NAME},
            max_retries=0,
        ),
        llm_provider=_AnswerLLM(),
        primary_document_detector=RuleBasedPrimaryDocumentDetector(),
        document_candidate_selector=DocumentBalancedCandidateSelector(candidate_multiplier=3),
        max_retries_per_information_need=0,
    )

    state = await graph.run(QueryState(question="What was the project Alex made for DAF?", top_k=5))

    second_plan = state.information_need_executions["need_2"].plan_history[0]
    assert state.primary_document_preference is not None
    assert state.primary_document_preference.document.document_id == FUNCTIONAL_ID
    assert second_plan.preferred_document is not None
    assert second_plan.preferred_document.document.document_id == FUNCTIONAL_ID
    assert "prefer_primary_document" in second_plan.adjustments
    assert retriever.calls == [("Alex DAF project", 15), ("DAF project functionality", 15)]
    balancing = state.metadata["active_information_need_lookup"]["document_balancing"]
    assert balancing["primary_selected_count"] == 3
    assert balancing["selected_document_counts"][f"document:{REALIZATION_ID}"] == 2
    lookup_steps = [step for step in state.trace if step.name == "execute_information_need_plan"]
    assert lookup_steps
    latest_lookup = lookup_steps[-1].metadata["active_information_need_lookup"]
    assert latest_lookup["primary_document"] == "FunctionalSpecDAF_AVP.pdf"
    assert latest_lookup["document_balancing"]["primary_selected_count"] == 3
