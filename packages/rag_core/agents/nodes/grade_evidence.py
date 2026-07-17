from __future__ import annotations

from packages.rag_core.agents.state import QueryState
from packages.rag_core.retrieval.graders import EvidenceGrader


class GradeEvidenceNode:
    """Grade retrieved chunks without owning the grading implementation."""

    name = "grade_evidence"
    step_type = "evidence_grading"

    def __init__(self, evidence_grader: EvidenceGrader) -> None:
        self._evidence_grader = evidence_grader

    async def __call__(self, state: QueryState) -> QueryState:
        report = await self._evidence_grader.grade(state.question, state.retrieved_evidence)
        grades_by_rank = {grade.evidence_rank: grade for grade in report.grades}

        for item in state.retrieved_evidence:
            grade = grades_by_rank.get(item.rank)
            if grade is not None:
                item.metadata["evidence_grade"] = grade.to_metadata()

        state.evidence_grading = report
        state.metadata["evidence_grading"] = report.to_metadata()
        return state
