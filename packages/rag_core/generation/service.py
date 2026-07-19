from __future__ import annotations

from packages.rag_core.generation.citations import citation_from_evidence, cited_evidence_ranks
from packages.rag_core.generation.evidence_selection import select_answer_evidence
from packages.rag_core.generation.models import AnswerGenerationRequest, AnswerGenerationResult
from packages.rag_core.generation.prompt import build_answer_prompt
from packages.rag_core.retrieval.constraint_validation import ConstraintValidationStatus, describe_constraints
from packages.rag_core.ports import LLMProvider


class AnswerGenerationService:
    """Generate complete or explicit partial answers from approved evidence."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    async def generate(self, request: AnswerGenerationRequest) -> AnswerGenerationResult:
        grading = request.evidence_grading
        evidence = select_answer_evidence(request.candidate_evidence, grading)
        candidate_count = len(request.candidate_evidence)
        unresolved = grading.unresolved_information if grading is not None else ()
        supported = grading.supported_information if grading is not None else ()

        validation = request.constraint_validation
        if validation is not None and validation.status is ConstraintValidationStatus.NO_MATCH:
            scope = describe_constraints(request.retrieval_constraints)
            return AnswerGenerationResult(
                answer=(
                    f"No indexed evidence matched the requested metadata scope ({scope}). "
                    "I did not use documents outside that document/date/version scope as a fallback."
                ),
                evidence=(),
                citations=(),
                candidate_evidence_count=candidate_count,
                irrelevant_evidence_filtered_count=candidate_count,
                unresolved_information=unresolved,
                supported_information=supported,
                is_partial=False,
                blocked_by_evidence_grading=True,
            )

        if grading is not None and not grading.answerable:
            missing_detail = f" Unresolved information: {'; '.join(unresolved)}." if unresolved else ""
            return AnswerGenerationResult(
                answer=(
                    f"The retrieved evidence was graded as {grading.status.value} and is not sufficient to answer "
                    f"any required part of the question reliably.{missing_detail}"
                ),
                evidence=evidence,
                citations=(),
                candidate_evidence_count=candidate_count,
                irrelevant_evidence_filtered_count=candidate_count - len(evidence),
                unresolved_information=unresolved,
                supported_information=supported,
                is_partial=False,
                blocked_by_evidence_grading=True,
            )

        if not evidence:
            return AnswerGenerationResult(
                answer=(
                    "I do not have enough retrieved evidence to answer this question yet. "
                    "Upload and index documents first, then ask again."
                ),
                evidence=(),
                citations=(),
                candidate_evidence_count=candidate_count,
                irrelevant_evidence_filtered_count=candidate_count,
                unresolved_information=unresolved,
                supported_information=supported,
                is_partial=False,
                blocked_by_evidence_grading=grading is not None,
            )

        prompt = build_answer_prompt(
            request.question,
            evidence,
            evidence_grading=grading,
            constraints=request.retrieval_constraints,
        )
        generated_answer = (await self._llm_provider.generate(prompt)).strip()
        is_partial = grading.partial_answer_available if grading is not None else False
        answer = append_unresolved_information(generated_answer, unresolved) if is_partial else generated_answer
        evidence_by_rank = {item.rank: item for item in evidence}
        referenced_ranks = cited_evidence_ranks(generated_answer, allowed_ranks=set(evidence_by_rank))
        citations = tuple(citation_from_evidence(evidence_by_rank[rank]) for rank in referenced_ranks)
        return AnswerGenerationResult(
            answer=answer,
            evidence=evidence,
            citations=citations,
            candidate_evidence_count=candidate_count,
            irrelevant_evidence_filtered_count=candidate_count - len(evidence),
            unresolved_information=unresolved,
            supported_information=supported,
            is_partial=is_partial,
            blocked_by_evidence_grading=False,
        )


def append_unresolved_information(answer: str, unresolved: tuple[str, ...]) -> str:
    if not unresolved:
        return answer
    rendered = "\n".join(f"- {description}" for description in unresolved)
    prefix = f"{answer}\n\n" if answer else ""
    return f"{prefix}The available documents did not provide sufficient evidence for:\n{rendered}"
