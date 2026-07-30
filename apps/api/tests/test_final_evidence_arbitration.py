from __future__ import annotations

import json

import pytest

from packages.rag_core.query_understanding.decomposition import InformationNeed
from packages.rag_core.retrieval.arbitration import (
    LLMQuestionEvidenceArbitrator,
    build_final_evidence_arbitration_prompt,
)
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem


def _need() -> InformationNeed:
    return InformationNeed(
        need_id="need_project",
        description="Explain the purpose and functionality of Alex's DAF project.",
        retrieval_query="Alex DAF project purpose functionality",
    )


def _prior_report(*, status: InformationNeedSupport = InformationNeedSupport.SUPPORTED) -> EvidenceGradingReport:
    coverage = 0.9 if status is InformationNeedSupport.SUPPORTED else 0.55
    return EvidenceGradingReport(
        status=(
            EvidenceSufficiency.SUFFICIENT
            if status is InformationNeedSupport.SUPPORTED
            else EvidenceSufficiency.WEAK
        ),
        coverage_score=coverage,
        grades=(
            EvidenceGrade(
                evidence_rank=1,
                relevance_score=0.9,
                relevant=True,
                rationale="Directly describes the DAF dashboarding project.",
                supports_information_need_ids=("need_project",),
            ),
            EvidenceGrade(
                evidence_rank=2,
                relevance_score=0.7,
                relevant=True,
                rationale="Mentions the project in a reference list.",
                supports_information_need_ids=("need_project",),
            ),
            EvidenceGrade(
                evidence_rank=3,
                relevance_score=0.2,
                relevant=False,
                rationale="Unrelated deployment material.",
            ),
        ),
        information_need_grades=(
            InformationNeedGrade(
                information_need_id="need_project",
                description=_need().description,
                status=status,
                coverage_score=coverage,
                supporting_evidence_ranks=(1, 2),
                rationale="The prior subgraph found direct and indirect project evidence.",
            ),
        ),
        rationale="Prior per-need grading report.",
        grader_name="test_prior",
    )


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(rank=1, text="The DAF dashboarding project specifies a dashboard for operational monitoring."),
        EvidenceItem(rank=2, text="References: DAF project plan, Alex van Poppel, 2026."),
        EvidenceItem(rank=3, text="The LLM Guidance project uses Docker Compose for deployment."),
    ]


class StaticArbitrationLLM:
    def __init__(self, *, requested_status: str = "supported") -> None:
        self.requested_status = requested_status
        self.prompts: list[str] = []
        self.schemas: list[dict[str, object]] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return json.dumps(
            {
                "grades": [
                    {
                        "rank": 1,
                        "relevance_score": 0.95,
                        "supports_information_need_ids": ["need_project"],
                        "rationale": "Directly explains the project functionality.",
                    },
                    {
                        "rank": 2,
                        "relevance_score": 0.15,
                        "supports_information_need_ids": [],
                        "rationale": "A reference-list mention does not answer the question.",
                    },
                    {
                        "rank": 3,
                        "relevance_score": 0.05,
                        "supports_information_need_ids": [],
                        "rationale": "This describes a different project.",
                    },
                ],
                "information_need_grades": [
                    {
                        "information_need_id": "need_project",
                        "status": self.requested_status,
                        "coverage_score": 0.9,
                        "supporting_ranks": [1],
                        "rationale": "Rank 1 directly grounds the answer.",
                    },
                ],
                "rationale": "Only the direct project description should reach generation.",
            },
        )

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        self.schemas.append(response_schema)
        return await self.generate(prompt)


async def test_final_arbiter_rejects_reference_and_unrelated_chunks() -> None:
    llm = StaticArbitrationLLM()
    arbiter = LLMQuestionEvidenceArbitrator(llm_provider=llm, fail_open=False)

    report = await arbiter.arbitrate(
        "What was the project Alex made for DAF?",
        _evidence(),
        (_need(),),
        _prior_report(),
    )

    assert report.relevant_evidence_ranks == (1,)
    assert report.status is EvidenceSufficiency.SUFFICIENT
    assert report.information_need_grades[0].supporting_evidence_ranks == (1,)
    assert "references, bibliographies" in llm.prompts[0]
    assert "different document is allowed" in llm.prompts[0]
    assert len(llm.schemas) == 1
    assert report.structured_output is not None
    assert report.structured_output.outcome == "primary_valid"


async def test_final_arbiter_cannot_upgrade_a_prior_partial_need() -> None:
    arbiter = LLMQuestionEvidenceArbitrator(
        llm_provider=StaticArbitrationLLM(requested_status="supported"),
        fail_open=False,
    )

    report = await arbiter.arbitrate(
        "What was the project Alex made for DAF?",
        _evidence(),
        (_need(),),
        _prior_report(status=InformationNeedSupport.PARTIAL),
    )

    assert report.status is EvidenceSufficiency.WEAK
    assert report.information_need_grades[0].status is InformationNeedSupport.PARTIAL
    assert report.information_need_grades[0].coverage_score == 0.55


def test_arbitration_prompt_exposes_prior_support_without_retrieval_scores() -> None:
    prompt = build_final_evidence_arbitration_prompt(
        "What was the project Alex made for DAF?",
        _evidence(),
        information_needs=(_need(),),
        prior_grading=_prior_report(),
    )

    assert "allowed_support_ids=['need_project']" in prompt
    assert "status=supported" in prompt
    assert "Prefer the smallest sufficient evidence set" in prompt


class InvalidArbitrationLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        del prompt
        return "not valid json"

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        del prompt, response_schema
        self.calls += 1
        return "not valid json"


class RepairingArbitrationLLM(StaticArbitrationLLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        self.schemas.append(response_schema)
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls == 1:
            return "not valid json"
        return await super().generate(prompt)


def _legacy_arbitration_response() -> str:
    return json.dumps(
        {
            "grades": [
                {"rank": 1, "relevance_score": 0.95, "rationale": "Direct evidence."},
                {"rank": 2, "relevance_score": 0.15, "rationale": "Reference only."},
                {"rank": 3, "relevance_score": 0.05, "rationale": "Different project."},
            ],
            "sufficiency": "sufficient",
            "coverage_score": 0.9,
            "rationale": "Only the direct project description should reach generation.",
        },
    )


class LegacyArbitrationLLM(StaticArbitrationLLM):
    def __init__(self, *, repair_with_modern: bool) -> None:
        super().__init__()
        self.repair_with_modern = repair_with_modern
        self.calls = 0

    async def generate_structured(self, prompt: str, *, response_schema) -> str:
        self.schemas.append(response_schema)
        self.calls += 1
        if self.calls == 1 or not self.repair_with_modern:
            self.prompts.append(prompt)
            return _legacy_arbitration_response()
        return await super().generate(prompt)


async def test_final_arbiter_repairs_once_and_preserves_prior_bounds() -> None:
    llm = RepairingArbitrationLLM()
    arbiter = LLMQuestionEvidenceArbitrator(llm_provider=llm, fail_open=False)

    report = await arbiter.arbitrate(
        "What was the project Alex made for DAF?",
        _evidence(),
        (_need(),),
        _prior_report(status=InformationNeedSupport.SUPPORTED),
    )

    assert len(llm.schemas) == 2
    assert report.status is EvidenceSufficiency.SUFFICIENT
    assert report.relevant_evidence_ranks == (1,)
    assert report.grades[0].relevance_score == 0.9
    assert report.structured_output is not None
    assert report.structured_output.outcome == "repair_valid"
    assert report.structured_output.failure_code == "invalid_json"


async def test_final_arbiter_repairs_legacy_response_to_modern_schema() -> None:
    llm = LegacyArbitrationLLM(repair_with_modern=True)
    arbiter = LLMQuestionEvidenceArbitrator(llm_provider=llm, fail_open=False)

    report = await arbiter.arbitrate(
        "What was the project Alex made for DAF?",
        _evidence(),
        (_need(),),
        _prior_report(),
    )

    assert len(llm.schemas) == 2
    assert report.structured_output is not None
    assert report.structured_output.outcome == "repair_valid"
    assert report.structured_output.failure_code == "schema_mismatch"


async def test_final_arbiter_fails_open_when_repair_remains_legacy() -> None:
    prior = _prior_report()
    llm = LegacyArbitrationLLM(repair_with_modern=False)
    arbiter = LLMQuestionEvidenceArbitrator(llm_provider=llm, fail_open=True)

    report = await arbiter.arbitrate(
        "What was the project Alex made for DAF?",
        _evidence(),
        (_need(),),
        prior,
    )

    assert len(llm.schemas) == 2
    assert report.relevant_evidence_ranks == prior.relevant_evidence_ranks
    assert report.fallback_used is True
    assert report.structured_output is not None
    assert report.structured_output.outcome == "fallback"
    assert report.structured_output.failure_code == "repair_invalid"


async def test_final_arbiter_fail_open_preserves_prior_approvals() -> None:
    prior = _prior_report(status=InformationNeedSupport.SUPPORTED)
    llm = InvalidArbitrationLLM()
    arbiter = LLMQuestionEvidenceArbitrator(llm_provider=llm, fail_open=True)

    report = await arbiter.arbitrate(
        "What was the project Alex made for DAF?",
        _evidence(),
        (_need(),),
        prior,
    )

    assert report.relevant_evidence_ranks == prior.relevant_evidence_ranks
    assert report.information_need_grades[0].status is InformationNeedSupport.SUPPORTED
    assert report.fallback_used is True
    assert report.grader_name == "llm_question_level_evidence_arbitrator"
    assert report.rationale == "Final evidence arbitration failed open; preserved the prior grades."
    assert llm.calls == 2
    assert report.structured_output is not None
    assert report.structured_output.outcome == "fallback"
    assert report.structured_output.failure_code == "repair_invalid"


async def test_final_arbiter_can_fail_closed_without_exposing_model_output() -> None:
    arbiter = LLMQuestionEvidenceArbitrator(
        llm_provider=InvalidArbitrationLLM(),
        fail_open=False,
    )

    with pytest.raises(
        RuntimeError,
        match="Final evidence arbitration failed structured validation",
    ) as captured:
        await arbiter.arbitrate(
            "What was the project Alex made for DAF?",
            _evidence(),
            (_need(),),
            _prior_report(),
        )

    assert "not valid json" not in str(captured.value)
