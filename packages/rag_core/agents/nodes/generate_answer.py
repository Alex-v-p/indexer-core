from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.agents.state import CitationItem, QueryState
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.graders import EvidenceGradingReport, InformationNeedSupport
from packages.rag_core.retrieval.models import EvidenceItem

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "answer_with_citations.md"


class GenerateAnswerNode:
    """Generate a complete or explicitly partial answer from grader-approved evidence."""

    name = "generate_answer"
    step_type = "generation"

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    async def __call__(self, state: QueryState) -> QueryState:
        grading = state.evidence_grading
        candidate_count = len(state.retrieved_evidence)
        evidence = _select_answer_evidence(state.retrieved_evidence, grading)

        # Only grader-approved evidence crosses the generation/persistence boundary. The
        # complete candidate set remains available in the evidence-grading and retry trace.
        state.retrieved_evidence = evidence
        state.metadata = {
            **state.metadata,
            "candidate_evidence_count": candidate_count,
            "evidence_count": len(evidence),
            "irrelevant_evidence_filtered_count": candidate_count - len(evidence),
            "unresolved_information": list(grading.unresolved_information) if grading is not None else [],
            "supported_information": list(grading.supported_information) if grading is not None else [],
        }

        if grading is not None and not grading.answerable:
            status = grading.status.value
            unresolved = grading.unresolved_information
            missing_detail = (
                f" Unresolved information: {'; '.join(unresolved)}."
                if unresolved
                else ""
            )
            state.answer = (
                f"The retrieved evidence was graded as {status} and is not sufficient to answer "
                f"any required part of the question reliably.{missing_detail}"
            )
            state.citations = []
            state.metadata = {
                **state.metadata,
                "citation_count": 0,
                "answer_is_partial": False,
                "answer_blocked_by_evidence_grading": True,
            }
            return state

        if not evidence:
            state.answer = (
                "I do not have enough retrieved evidence to answer this question yet. "
                "Upload and index documents first, then ask again."
            )
            state.citations = []
            state.metadata = {
                **state.metadata,
                "citation_count": 0,
                "answer_is_partial": False,
                "answer_blocked_by_evidence_grading": grading is not None,
            }
            return state

        prompt = build_answer_prompt(state.question, evidence, evidence_grading=grading)
        generated_answer = (await self._llm_provider.generate(prompt)).strip()
        is_partial = grading.partial_answer_available if grading is not None else False
        state.answer = (
            _append_unresolved_information(generated_answer, grading.unresolved_information)
            if is_partial and grading is not None
            else generated_answer
        )
        state.citations = [_to_citation(item) for item in sorted(evidence, key=lambda item: item.rank)]
        state.metadata = {
            **state.metadata,
            "citation_count": len(state.citations),
            "answer_is_partial": is_partial,
            "answer_blocked_by_evidence_grading": False,
        }
        return state


def build_answer_prompt(
    question: str,
    evidence: list[EvidenceItem],
    *,
    evidence_grading: EvidenceGradingReport | None = None,
) -> str:
    """Build the citation-oriented answer prompt from the markdown template."""

    evidence_block = "\n\n".join(_format_evidence(item) for item in sorted(evidence, key=lambda item: item.rank))
    template = _load_answer_prompt_template()
    return (
        template.replace("{{ question }}", question)
        .replace("{{ claim_coverage }}", _format_claim_coverage(evidence_grading))
        .replace("{{ evidence }}", evidence_block)
        .strip()
    )


@lru_cache(maxsize=1)
def _load_answer_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _select_answer_evidence(
    evidence: list[EvidenceItem],
    grading: EvidenceGradingReport | None,
) -> list[EvidenceItem]:
    if grading is None:
        return [item for item in evidence if item.text.strip()]

    relevant_ranks = set(grading.relevant_evidence_ranks)
    return [item for item in evidence if item.rank in relevant_ranks and item.text.strip()]


def _format_claim_coverage(grading: EvidenceGradingReport | None) -> str:
    if grading is None or not grading.information_need_grades:
        return "No explicit claim-level coverage report is available. Answer only what the evidence directly supports."

    supported = [
        f"- {grade.description}"
        for grade in grading.information_need_grades
        if grade.required and grade.status is InformationNeedSupport.SUPPORTED
    ]
    unresolved = [
        f"- [{grade.status.value}] {grade.description}"
        for grade in grading.information_need_grades
        if grade.required and grade.status is not InformationNeedSupport.SUPPORTED
    ]
    sections: list[str] = []
    if supported:
        sections.append("Supported required claims:\n" + "\n".join(supported))
    if unresolved:
        sections.append("Unresolved required claims:\n" + "\n".join(unresolved))
    return "\n\n".join(sections) or "No required claims were identified."


def _append_unresolved_information(answer: str, unresolved: tuple[str, ...]) -> str:
    if not unresolved:
        return answer
    rendered = "\n".join(f"- {description}" for description in unresolved)
    prefix = f"{answer}\n\n" if answer else ""
    return (
        f"{prefix}The available documents did not provide sufficient evidence for:\n"
        f"{rendered}"
    )


def _format_evidence(item: EvidenceItem) -> str:
    source_bits = _source_bits(item)
    source_line = f"Source metadata: {', '.join(source_bits)}\n" if source_bits else ""
    return f"[{item.rank}]\n{source_line}Text: {item.text.strip()}"


def _source_bits(item: EvidenceItem) -> list[str]:
    bits: list[str] = []
    filename = item.metadata.get("original_filename")
    if isinstance(filename, str) and filename:
        bits.append(f"file={filename}")

    section_title = item.metadata.get("section_title")
    if isinstance(section_title, str) and section_title:
        bits.append(f"section={section_title}")

    page_start = _first_int_metadata(item.metadata, "page_number", "source_page_start")
    page_end = _first_int_metadata(item.metadata, "source_page_end")
    if page_start is not None and page_end is not None and page_end != page_start:
        bits.append(f"pages={page_start}-{page_end}")
    elif page_start is not None:
        bits.append(f"page={page_start}")

    if item.score is not None:
        bits.append(f"score={item.score:.4f}")
    return bits


def _to_citation(item: EvidenceItem) -> CitationItem:
    return CitationItem(
        citation_index=item.rank,
        evidence_rank=item.rank,
        label=f"[{item.rank}]",
        page_number=_first_int_metadata(item.metadata, "page_number", "source_page_start"),
        quote=item.text[:500],
        qdrant_chunk_index_id=item.qdrant_chunk_index_id,
        document_id=item.document_id,
        document_version_id=item.document_version_id,
        metadata={"score": item.score, **item.metadata},
    )


def _first_int_metadata(metadata: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                continue
    return None
