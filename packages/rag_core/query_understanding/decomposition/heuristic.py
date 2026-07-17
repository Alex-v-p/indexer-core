from __future__ import annotations

import re

from packages.rag_core.query_understanding.decomposition.models import (
    InformationNeed,
    InformationNeedDecomposition,
)

_CLAUSE_BOUNDARY_PATTERN = re.compile(
    r"\s+(?:and|as well as)\s+(?=(?:how|why|what|which|where|when|whether|explain|describe|identify|list)\b)",
    re.IGNORECASE,
)


class HeuristicInformationNeedDecomposer:
    """Conservative deterministic fallback for compound-question decomposition."""

    name = "heuristic_information_need_decomposer"

    def __init__(self, *, max_information_needs: int = 6) -> None:
        if max_information_needs <= 0:
            raise ValueError("max_information_needs must be positive.")
        self._max_information_needs = max_information_needs

    async def decompose(self, question: str) -> InformationNeedDecomposition:
        normalized = " ".join(question.strip().split())
        if not normalized:
            raise ValueError("question must not be empty.")

        clauses = [part.strip(" ,;?.") for part in _CLAUSE_BOUNDARY_PATTERN.split(normalized)]
        clauses = [part for part in clauses if part]
        if len(clauses) <= 1:
            clauses = [normalized]

        needs = tuple(
            InformationNeed(
                need_id=f"need_{index}",
                description=_as_answer_requirement(clause),
                retrieval_query=clause,
            )
            for index, clause in enumerate(clauses[: self._max_information_needs], start=1)
        )
        rationale = (
            "The question was split at explicit compound-question boundaries."
            if len(needs) > 1
            else "The question expresses one cohesive answer requirement."
        )
        return InformationNeedDecomposition(
            information_needs=needs,
            rationale=rationale,
            decomposer_name=self.name,
        )


def _as_answer_requirement(clause: str) -> str:
    normalized = " ".join(clause.strip().split())
    return normalized[0].upper() + normalized[1:] if normalized else normalized
