from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.arbitration import EvidenceArbitrator
from packages.rag_core.retrieval.models import RetrievalConstraints


class ArbitrateFinalEvidenceNode:
    """Apply a strict original-question-level filter after per-need aggregation."""

    name = "arbitrate_final_evidence"
    step_type = "evidence_arbitration"

    def __init__(self, arbitrator: EvidenceArbitrator) -> None:
        self._arbitrator = arbitrator

    async def __call__(self, state: QueryState) -> QueryState:
        prior = state.evidence_grading
        if prior is None:
            raise RuntimeError("Final evidence arbitration requires an aggregated grading report.")
        information_needs = _information_needs(state)
        report = await self._arbitrator.arbitrate(
            state.question,
            state.retrieved_evidence,
            information_needs,
            prior,
            constraints=_query_constraints(state),
        )
        grades_by_rank = {grade.evidence_rank: grade for grade in report.grades}
        for item in state.retrieved_evidence:
            grade = grades_by_rank.get(item.rank)
            if grade is not None:
                item.metadata["final_evidence_arbitration"] = grade.to_metadata()

        state.evidence_grading = report
        state.metadata["evidence_arbitration"] = report.to_metadata()
        state.metadata["evidence_grading"] = report.to_metadata()
        state.metadata["unresolved_information"] = list(report.unresolved_information)
        state.metadata["supported_information"] = list(report.supported_information)
        return state


def _information_needs(state: QueryState) -> tuple[InformationNeed, ...]:
    decomposition = state.information_need_decomposition
    if decomposition is not None and decomposition.information_needs:
        return decomposition.information_needs
    return (
        InformationNeed(
            need_id="need_1",
            description=state.question,
            retrieval_query=state.question,
        ),
    )


def _query_constraints(state: QueryState) -> RetrievalConstraints:
    classification = state.query_classification
    if classification is not None:
        return RetrievalConstraints(
            document=classification.document_constraint,
            version=classification.version_constraint,
            dates=classification.date_constraints,
        )
    return RetrievalConstraints()
