from __future__ import annotations

import pytest

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.nodes import GenerateAnswerNode, GradeEvidenceNode
from packages.rag_core.query_understanding.decomposition import InformationNeed, InformationNeedDecomposition
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingError,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
    LLMEvidenceGrader,
    parse_evidence_grading,
)


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(rank=1, text="The API listens on port 8000."),
        EvidenceItem(rank=2, text="The document upload page uses a navigation header."),
    ]


def _pipeline_needs() -> tuple[InformationNeed, ...]:
    return (
        InformationNeed(
            need_id="need_1",
            description="Identify the available pipeline flows.",
            retrieval_query="available pipeline flows",
        ),
        InformationNeed(
            need_id="need_2",
            description="Explain how each pipeline flow functions.",
            retrieval_query="pipeline flow functionality execution steps",
        ),
    )


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


def test_claim_level_grading_reports_covered_and_missing_information() -> None:
    evidence = [
        EvidenceItem(rank=1, text="The project registers baseline, hybrid, and multi-query pipeline flows."),
        EvidenceItem(rank=2, text="The UI contains a document upload page."),
    ]
    report = parse_evidence_grading(
        """{
          "grades": [
            {
              "rank": 1,
              "relevance_score": 0.95,
              "supports_information_need_ids": ["need_1"],
              "rationale": "Identifies the available pipeline flows."
            },
            {
              "rank": 2,
              "relevance_score": 0.05,
              "supports_information_need_ids": [],
              "rationale": "Unrelated UI information."
            }
          ],
          "information_need_grades": [
            {
              "information_need_id": "need_1",
              "status": "supported",
              "coverage_score": 0.95,
              "supporting_ranks": [1],
              "rationale": "The flow names are explicitly listed."
            },
            {
              "information_need_id": "need_2",
              "status": "missing",
              "coverage_score": 0.05,
              "supporting_ranks": [],
              "rationale": "No chunk explains how the flows execute."
            }
          ],
          "rationale": "Pipeline identities are covered, but their functionality is absent."
        }""",
        evidence=evidence,
        information_needs=_pipeline_needs(),
    )

    assert report.status is EvidenceSufficiency.WEAK
    assert report.supported_information_need_count == 1
    assert report.supported_required_information_need_count == 1
    assert report.required_information_need_count == 2
    assert report.partial_answer_available is True
    assert report.answerable is True
    assert report.relevant_evidence_ranks == (1,)
    assert report.supported_information == ("Identify the available pipeline flows.",)
    assert report.missing_information_need_count == 1
    assert report.information_need_grades[0].status is InformationNeedSupport.SUPPORTED
    assert report.information_need_grades[1].status is InformationNeedSupport.MISSING
    assert report.unresolved_information == ("Explain how each pipeline flow functions.",)
    assert report.grades[0].supports_information_need_ids == ("need_1",)


def test_claim_level_grading_requires_every_information_need() -> None:
    evidence = [
        EvidenceItem(rank=1, text="The project has baseline and hybrid flows."),
        EvidenceItem(rank=2, text="The baseline flow embeds the query and performs vector retrieval."),
    ]
    report = parse_evidence_grading(
        """{
          "grades": [
            {
              "rank": 1,
              "relevance_score": 0.9,
              "supports_information_need_ids": ["need_1"],
              "rationale": "Lists the flows."
            },
            {
              "rank": 2,
              "relevance_score": 0.9,
              "supports_information_need_ids": ["need_2"],
              "rationale": "Explains execution behavior."
            }
          ],
          "information_need_grades": [
            {
              "information_need_id": "need_1",
              "status": "supported",
              "coverage_score": 0.9,
              "supporting_ranks": [1],
              "rationale": "The flows are identified."
            },
            {
              "information_need_id": "need_2",
              "status": "supported",
              "coverage_score": 0.85,
              "supporting_ranks": [2],
              "rationale": "The flow behavior is explained."
            }
          ],
          "rationale": "All requested aspects are supported."
        }""",
        evidence=evidence,
        information_needs=_pipeline_needs(),
    )

    assert report.status is EvidenceSufficiency.SUFFICIENT
    assert report.sufficient is True
    assert report.unresolved_information == ()


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


def test_llm_grading_parser_rejects_omitted_information_need() -> None:
    with pytest.raises(EvidenceGradingError, match="omitted information need ids: need_2"):
        parse_evidence_grading(
            """{
              "grades": [
                {
                  "rank": 1,
                  "relevance_score": 0.9,
                  "supports_information_need_ids": ["need_1"],
                  "rationale": "Relevant."
                },
                {"rank": 2, "relevance_score": 0.1, "supports_information_need_ids": [], "rationale": "Irrelevant."}
              ],
              "information_need_grades": [
                {
                  "information_need_id": "need_1",
                  "status": "supported",
                  "coverage_score": 0.9,
                  "supporting_ranks": [1],
                  "rationale": "Covered."
                }
              ],
              "rationale": "Incomplete grading."
            }""",
            evidence=_evidence(),
            information_needs=_pipeline_needs(),
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
    assert report.information_need_grades[0].status is InformationNeedSupport.SUPPORTED


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


class ClaimAwareWeakEvidenceGrader:
    async def grade(self, question: str, evidence: list[EvidenceItem]) -> EvidenceGradingReport:
        del question, evidence
        raise AssertionError("The node should use grade_information_needs when decomposition contains needs.")

    async def grade_information_needs(
        self,
        question: str,
        evidence: list[EvidenceItem],
        information_needs: tuple[InformationNeed, ...],
    ) -> EvidenceGradingReport:
        del question
        return EvidenceGradingReport(
            status=EvidenceSufficiency.WEAK,
            coverage_score=0.5,
            grades=(
                EvidenceGrade(
                    evidence_rank=evidence[0].rank,
                    relevance_score=0.9,
                    relevant=True,
                    rationale="Supports only pipeline identification.",
                    supports_information_need_ids=(information_needs[0].need_id,),
                ),
                EvidenceGrade(
                    evidence_rank=evidence[1].rank,
                    relevance_score=0.1,
                    relevant=False,
                    rationale="Does not support either need.",
                ),
            ),
            information_need_grades=(
                InformationNeedGrade(
                    information_need_id=information_needs[0].need_id,
                    description=information_needs[0].description,
                    status=InformationNeedSupport.SUPPORTED,
                    coverage_score=0.9,
                    supporting_evidence_ranks=(1,),
                    rationale="The flows are identified.",
                ),
                InformationNeedGrade(
                    information_need_id=information_needs[1].need_id,
                    description=information_needs[1].description,
                    status=InformationNeedSupport.MISSING,
                    coverage_score=0.1,
                    supporting_evidence_ranks=(),
                    rationale="Their functionality is not explained.",
                ),
            ),
            rationale="One required need is missing.",
            grader_name="claim_aware_test_grader",
        )


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The available pipeline flows are baseline and hybrid [1]."


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
    assert state.metadata["irrelevant_evidence_filtered_count"] == 1
    assert [item.rank for item in state.retrieved_evidence] == [1]
    assert "graded as weak" in (state.answer or "")
    assert state.citations == []
    assert llm.prompts == []


async def test_grading_node_uses_decomposed_information_needs_and_exposes_missing_aspect() -> None:
    needs = _pipeline_needs()
    state = QueryState(
        question="What are the pipeline flows and how do they function?",
        retrieved_evidence=[
            EvidenceItem(rank=1, text="The pipeline names are baseline and hybrid."),
            EvidenceItem(rank=2, text="The UI has an upload page."),
        ],
        information_need_decomposition=InformationNeedDecomposition(
            information_needs=needs,
            rationale="Identification and functionality must be graded independently.",
            decomposer_name="test",
        ),
    )

    state = await GradeEvidenceNode(ClaimAwareWeakEvidenceGrader())(state)
    llm = RecordingLLM()
    state = await GenerateAnswerNode(llm)(state)

    assert state.metadata["unresolved_information"] == ["Explain how each pipeline flow functions."]
    assert state.metadata["supported_information"] == ["Identify the available pipeline flows."]
    assert state.metadata["answer_is_partial"] is True
    assert state.metadata["answer_blocked_by_evidence_grading"] is False
    assert state.metadata["irrelevant_evidence_filtered_count"] == 1
    assert [item.rank for item in state.retrieved_evidence] == [1]
    assert state.retrieved_evidence[0].metadata["evidence_grade"]["supports_information_need_ids"] == ["need_1"]
    assert [citation.label for citation in state.citations] == ["[1]"]
    assert len(llm.prompts) == 1
    assert "Supported required claims:" in llm.prompts[0]
    assert "Unresolved required claims:" in llm.prompts[0]
    assert "The UI has an upload page." not in llm.prompts[0]
    assert "The available pipeline flows are baseline and hybrid [1]." in (state.answer or "")
    assert "The available documents did not provide sufficient evidence for:" in (state.answer or "")
    assert "Explain how each pipeline flow functions." in (state.answer or "")
