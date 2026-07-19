from __future__ import annotations

from packages.rag_core.generation import AnswerGenerationRequest, AnswerGenerationService
from packages.rag_core.retrieval.models import EvidenceItem


class CitationAnswerLLM:
    async def generate(self, prompt: str) -> str:
        del prompt
        return "The second source supports the first claim [2]. The others support another claim [1, 3]. Unknown [99]."


async def test_generation_returns_only_citations_explicitly_used_by_the_answer() -> None:
    service = AnswerGenerationService(CitationAnswerLLM())
    evidence = tuple(
        EvidenceItem(rank=rank, text=f"Evidence {rank}.")
        for rank in (1, 2, 3, 4)
    )

    result = await service.generate(
        AnswerGenerationRequest(
            question="What is supported?",
            candidate_evidence=evidence,
            evidence_grading=None,
        ),
    )

    assert [citation.evidence_rank for citation in result.citations] == [2, 1, 3]
    assert [citation.label for citation in result.citations] == ["[2]", "[1]", "[3]"]
    assert len(result.evidence) == 4
