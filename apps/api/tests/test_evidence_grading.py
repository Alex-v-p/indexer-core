from __future__ import annotations

import pytest

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.nodes import GenerateAnswerNode, GradeEvidenceNode
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingError,
    EvidenceGradingReport,
    EvidenceSufficiency,
    LLMEvidenceGrader,
    parse_evidence_grading,
)


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(rank=1, text="The API listens on port 8000."),
        EvidenceItem(rank=2, text="The document upload page uses a navigation header."),
    ]


def test_llm_grading_parser_scores_every_evidence_rank() -> None:
    report = parse_evidence_grading(
        """{
          "grades": [
            {"rank": 1, "relevance_score": 0.95, "rationale": "Directly states the API port."},
            {"rank": 2, "relevance_score": 0.1, "rationale": "Discusses an unrelated UI detail."}
          ],
          "sufficiency": "sufficient",
          "coverage_score": 0.92,
          "rationale": "The first chunk directly answers the question."
        }""",
        evidence=_evidence(),
        relevance_threshold=0.6,
    )

    assert report.status is EvidenceSufficiency.SUFFICIENT
    assert report.sufficient is True
    assert report.relevant_count == 1
    assert report.total_count == 2
    assert report.grades[0].relevant is True
    assert report.grades[1].relevant is False


def test_llm_grading_parser_rejects_incomplete_rank_coverage() -> None:
    with pytest.raises(EvidenceGradingError, match="omitted ranks: 2"):
        parse_evidence_grading(
            """{
              "grades": [
                {"rank": 1, "relevance_score": 0.9, "rationale": "Relevant."}
              ],
              "sufficiency": "sufficient",
              "coverage_score": 0.8,
              "rationale": "Enough evidence."
            }""",
            evidence=_evidence(),
        )


class InvalidGradingLLM:
    async def generate(self, prompt: str) -> str:
        del prompt
        return "not-json"


async def test_llm_grader_falls_back_to_deterministic_grading() -> None:
    grader = LLMEvidenceGrader(llm_provider=InvalidGradingLLM(), fail_open=True)

    report = await grader.grade("Which port does the API use?", _evidence())

    assert report.fallback_used is True
    assert report.grader_name == "heuristic_evidence_grader"
    assert report.status is EvidenceSufficiency.SUFFICIENT
    assert report.grades[0].relevant is True
    assert report.grades[1].relevant is False


async def test_llm_grader_can_fail_closed() -> None:
    grader = LLMEvidenceGrader(llm_provider=InvalidGradingLLM(), fail_open=False)

    with pytest.raises(EvidenceGradingError):
        await grader.grade("Which port does the API use?", _evidence())


class WeakEvidenceGrader:
    async def grade(self, question: str, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
        del question
        return EvidenceGradingReport(
            status=EvidenceSufficiency.WEAK,
            coverage_score=0.45,
            grades=tuple(
                EvidenceGrade(
                    evidence_rank=item.rank,
                    relevance_score=0.45 if item.rank == 1 else 0.1,
                    relevant=item.rank == 1,
                    rationale="Only partial support is present." if item.rank == 1 else "Not relevant.",
                )
                for item in evidence
            ),
            rationale="The evidence does not cover the complete question.",
            grader_name="test_grader",
        )


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "This should not be generated."


async def test_grading_node_stores_scores_and_blocks_generation_when_weak() -> None:
    state = QueryState(
        question="Explain the complete deployment workflow.",
        retrieved_evidence=_evidence(),
    )
    state = await GradeEvidenceNode(WeakEvidenceGrader())(state)

    llm = RecordingLLM()
    state = await GenerateAnswerNode(llm)(state)

    assert state.evidence_grading is not None
    assert state.evidence_grading.status is EvidenceSufficiency.WEAK
    assert state.metadata["evidence_grading"]["weak_evidence"] is True
    assert state.retrieved_evidence[0].metadata["evidence_grade"]["relevance_score"] == 0.45
    assert state.metadata["answer_blocked_by_evidence_grading"] is True
    assert "graded as weak" in (state.answer or "")
    assert state.citations == []
    assert llm.prompts == []
