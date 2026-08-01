from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.generation import AnswerGenerationRequest, AnswerGenerationService
from packages.rag_core.ports import LLMProvider


class GenerateAnswerNode:
    """Apply the generation service result to query graph state."""

    name = "generate_answer"
    step_type = "generation"

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._service = AnswerGenerationService(llm_provider)

    async def __call__(self, state: QueryState) -> QueryState:
        result = await self._service.generate(
            AnswerGenerationRequest(
                question=state.question,
                candidate_evidence=tuple(state.retrieved_evidence),
                evidence_grading=state.evidence_grading,
                retrieval_constraints=(
                    state.evidence_context.constraints
                    if state.evidence_context is not None
                    else _classification_constraints(state)
                ),
                constraint_validation=state.constraint_validation,
                evidence_context=state.evidence_context,
            ),
        )
        state.answer = result.answer
        state.answer_presentation = result.presentation
        state.retrieved_evidence = list(result.evidence)
        state.citations = list(result.citations)
        state.metadata.update(
            {
                "candidate_evidence_count": result.candidate_evidence_count,
                "evidence_count": len(result.evidence),
                "irrelevant_evidence_filtered_count": result.irrelevant_evidence_filtered_count,
                "unresolved_information": list(result.unresolved_information),
                "supported_information": list(result.supported_information),
                "citation_count": len(result.citations),
                "answer_is_partial": result.is_partial,
                "answer_blocked_by_evidence_grading": result.blocked_by_evidence_grading,
                "answer_presentation": result.presentation.to_metadata(),
            },
        )
        return state


def _classification_constraints(state: QueryState):
    from packages.rag_core.retrieval.models import RetrievalConstraints

    classification = state.query_classification
    if classification is None:
        return RetrievalConstraints(document_scope=state.document_scope)
    return RetrievalConstraints(
        document=classification.document_constraint,
        version=classification.version_constraint,
        dates=classification.date_constraints,
        document_scope=state.document_scope,
    )
