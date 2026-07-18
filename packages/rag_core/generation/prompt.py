from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from packages.rag_core.generation.citations import first_int_metadata
from packages.rag_core.retrieval.graders import EvidenceGradingReport, InformationNeedSupport
from packages.rag_core.retrieval.models import EvidenceItem

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "answer_with_citations.md"


def build_answer_prompt(
    question: str,
    evidence: tuple[EvidenceItem, ...],
    *,
    evidence_grading: EvidenceGradingReport | None = None,
) -> str:
    """Build the citation-oriented answer prompt from the markdown template."""

    evidence_block = "\n\n".join(format_evidence(item) for item in sorted(evidence, key=lambda item: item.rank))
    template = load_answer_prompt_template()
    return (
        template.replace("{{ question }}", question)
        .replace("{{ claim_coverage }}", format_claim_coverage(evidence_grading))
        .replace("{{ evidence }}", evidence_block)
        .strip()
    )


@lru_cache(maxsize=1)
def load_answer_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def format_claim_coverage(grading: EvidenceGradingReport | None) -> str:
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


def format_evidence(item: EvidenceItem) -> str:
    source_bits = source_metadata_bits(item)
    source_line = f"Source metadata: {', '.join(source_bits)}\n" if source_bits else ""
    return f"[{item.rank}]\n{source_line}Text: {item.text.strip()}"


def source_metadata_bits(item: EvidenceItem) -> list[str]:
    bits: list[str] = []
    filename = item.metadata.get("original_filename")
    if isinstance(filename, str) and filename:
        bits.append(f"file={filename}")

    section_title = item.metadata.get("section_title")
    if isinstance(section_title, str) and section_title:
        bits.append(f"section={section_title}")

    page_start = first_int_metadata(item.metadata, "page_number", "source_page_start")
    page_end = first_int_metadata(item.metadata, "source_page_end")
    if page_start is not None and page_end is not None and page_end != page_start:
        bits.append(f"pages={page_start}-{page_end}")
    elif page_start is not None:
        bits.append(f"page={page_start}")

    if item.score is not None:
        bits.append(f"score={item.score:.4f}")
    return bits
