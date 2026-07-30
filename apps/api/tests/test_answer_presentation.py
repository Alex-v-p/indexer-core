from __future__ import annotations

from packages.rag_core.documents import DocumentNameConstraint
from packages.rag_core.generation import (
    AnswerGenerationRequest,
    AnswerGenerationService,
    AnswerPresentationOutcome,
)
from packages.rag_core.retrieval.constraint_validation import (
    ConstraintValidationReport,
    ConstraintValidationStatus,
)
from packages.rag_core.retrieval.graders import (
    EvidenceGrade,
    EvidenceGradingReport,
    EvidenceSufficiency,
    InformationNeedGrade,
    InformationNeedSupport,
)
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints


class StaticAnswerLLM:
    def __init__(self, answer: str = "The API listens on port 8000 [1].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def _evidence() -> tuple[EvidenceItem, ...]:
    return (EvidenceItem(rank=1, text="The API listens on port 8000."),)


def _partial_grading() -> EvidenceGradingReport:
    return EvidenceGradingReport(
        status=EvidenceSufficiency.WEAK,
        coverage_score=0.5,
        grades=(
            EvidenceGrade(
                evidence_rank=1,
                relevance_score=0.95,
                relevant=True,
                rationale="Directly states the port.",
                supports_information_need_ids=("need_port",),
            ),
        ),
        information_need_grades=(
            InformationNeedGrade(
                information_need_id="need_port",
                description="Identify the API port.",
                status=InformationNeedSupport.SUPPORTED,
                coverage_score=0.95,
                supporting_evidence_ranks=(1,),
                rationale="The port is stated.",
            ),
            InformationNeedGrade(
                information_need_id="need_protocol",
                description="Identify the API protocol.",
                status=InformationNeedSupport.MISSING,
                coverage_score=0.0,
                supporting_evidence_ranks=(),
                rationale="The protocol is absent.",
            ),
        ),
        rationale="Only the port is supported.",
        grader_name="test",
    )


def _missing_grading() -> EvidenceGradingReport:
    return EvidenceGradingReport(
        status=EvidenceSufficiency.MISSING,
        coverage_score=0.0,
        grades=(
            EvidenceGrade(
                evidence_rank=1,
                relevance_score=0.0,
                relevant=False,
                rationale="Does not answer the question.",
            ),
        ),
        information_need_grades=(
            InformationNeedGrade(
                information_need_id="need_protocol",
                description="Identify the API protocol.",
                status=InformationNeedSupport.MISSING,
                coverage_score=0.0,
                supporting_evidence_ranks=(),
                rationale="The protocol is absent.",
            ),
        ),
        rationale="No required information is supported.",
        grader_name="test",
    )


async def test_complete_answer_has_versioned_presentation_and_citation_count() -> None:
    llm = StaticAnswerLLM()
    result = await AnswerGenerationService(llm).generate(
        AnswerGenerationRequest(
            question="Which port does the API use?",
            candidate_evidence=_evidence(),
            evidence_grading=None,
        ),
    )

    assert result.answer == "The API listens on port 8000 [1]."
    assert result.presentation.schema_version == "1.0"
    assert result.presentation.outcome is AnswerPresentationOutcome.COMPLETE
    assert result.presentation.title == "Answer"
    assert result.presentation.body == result.answer
    assert result.presentation.supported_information == ()
    assert result.presentation.unresolved_information == ()
    assert result.presentation.citation_count == 1
    assert [citation.evidence_rank for citation in result.citations] == [1]
    assert "Return only the answer body prose with inline citations." in llm.prompts[0]
    assert "Do not add a heading, title, preamble" in llm.prompts[0]
    assert "Do not add or repeat an unresolved-information section" in llm.prompts[0]


async def test_partial_presentation_uses_raw_prose_while_legacy_answer_is_unchanged() -> None:
    generated = "The API listens on port 8000 [1]."
    result = await AnswerGenerationService(StaticAnswerLLM(generated)).generate(
        AnswerGenerationRequest(
            question="Which port and protocol does the API use?",
            candidate_evidence=_evidence(),
            evidence_grading=_partial_grading(),
        ),
    )

    expected_legacy = (
        f"{generated}\n\n"
        "The available documents did not provide sufficient evidence for:\n"
        "- Identify the API protocol."
    )
    assert result.answer == expected_legacy
    assert result.presentation.outcome is AnswerPresentationOutcome.PARTIAL
    assert result.presentation.title == "Partial answer"
    assert result.presentation.body == generated
    assert "did not provide sufficient evidence" not in result.presentation.body
    assert result.presentation.supported_information == ("Identify the API port.",)
    assert result.presentation.unresolved_information == ("Identify the API protocol.",)
    assert result.presentation.citation_count == 1


async def test_constraint_no_match_has_blocked_presentation() -> None:
    constraints = RetrievalConstraints(
        document=DocumentNameConstraint(
            names=("Missing Guide.pdf",),
            confidence=1.0,
            rationale="Explicitly requested.",
            detector_name="test",
        ),
    )
    validation = ConstraintValidationReport(
        status=ConstraintValidationStatus.NO_MATCH,
        constraints=constraints,
        candidate_count=1,
        matched_count=0,
        rejected_count=1,
        rationale="No evidence matched.",
    )
    result = await AnswerGenerationService(StaticAnswerLLM()).generate(
        AnswerGenerationRequest(
            question="What does the guide say?",
            candidate_evidence=_evidence(),
            evidence_grading=None,
            retrieval_constraints=constraints,
            constraint_validation=validation,
        ),
    )

    assert result.presentation.outcome is AnswerPresentationOutcome.BLOCKED_CONSTRAINT_NO_MATCH
    assert result.presentation.title == "No evidence matched the requested scope"
    assert result.presentation.body == result.answer
    assert result.presentation.supported_information == ()
    assert result.presentation.unresolved_information == ()
    assert result.presentation.citation_count == 0


async def test_unanswerable_grading_has_insufficient_evidence_presentation() -> None:
    llm = StaticAnswerLLM()
    result = await AnswerGenerationService(llm).generate(
        AnswerGenerationRequest(
            question="Which protocol does the API use?",
            candidate_evidence=_evidence(),
            evidence_grading=_missing_grading(),
        ),
    )

    assert result.presentation.outcome is AnswerPresentationOutcome.BLOCKED_INSUFFICIENT_EVIDENCE
    assert result.presentation.title == "Insufficient evidence"
    assert result.presentation.body == result.answer
    assert result.presentation.supported_information == ()
    assert result.presentation.unresolved_information == ("Identify the API protocol.",)
    assert result.presentation.citation_count == 0
    assert llm.prompts == []


async def test_no_evidence_has_blocked_no_evidence_presentation() -> None:
    llm = StaticAnswerLLM()
    result = await AnswerGenerationService(llm).generate(
        AnswerGenerationRequest(
            question="Which port does the API use?",
            candidate_evidence=(),
            evidence_grading=None,
        ),
    )

    assert result.answer == (
        "I do not have enough retrieved evidence to answer this question yet. "
        "Upload and index documents first, then ask again."
    )
    assert result.presentation.outcome is AnswerPresentationOutcome.BLOCKED_NO_EVIDENCE
    assert result.presentation.title == "No evidence available"
    assert result.presentation.body == result.answer
    assert result.presentation.supported_information == ()
    assert result.presentation.unresolved_information == ()
    assert result.presentation.citation_count == 0
    assert llm.prompts == []
