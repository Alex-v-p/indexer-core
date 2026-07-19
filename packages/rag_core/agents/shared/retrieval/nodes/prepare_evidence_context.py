from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.retrieval.constraint_validation import validate_evidence_constraints
from packages.rag_core.retrieval.evidence_context import build_evidence_context_bundle
from packages.rag_core.retrieval.models import RetrievalConstraints


class PrepareEvidenceContextNode:
    """Strictly enforce query constraints and prepare compact metadata for generation."""

    name = "prepare_evidence_context"
    step_type = "evidence_context"

    async def __call__(self, state: QueryState) -> QueryState:
        constraints = _query_constraints(state)
        matched, report = validate_evidence_constraints(state.retrieved_evidence, constraints)
        state.retrieved_evidence = matched
        state.constraint_validation = report
        state.evidence_context = build_evidence_context_bundle(
            matched,
            constraints=constraints,
            validation=report,
        )
        state.metadata["constraint_validation"] = report.to_metadata()
        state.metadata["evidence_context"] = state.evidence_context.to_metadata()
        return state


def _query_constraints(state: QueryState) -> RetrievalConstraints:
    classification = state.query_classification
    if classification is not None:
        return RetrievalConstraints(
            document=classification.document_constraint,
            version=classification.version_constraint,
            dates=classification.date_constraints,
        )
    plan = state.effective_retrieval_plan
    if plan is not None:
        return RetrievalConstraints(
            document=plan.document_constraint,
            version=plan.version_constraint,
            dates=plan.date_constraints,
        )
    return RetrievalConstraints()
