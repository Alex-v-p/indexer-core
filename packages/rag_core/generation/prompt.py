from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from packages.rag_core.generation.citations import first_int_metadata
from packages.rag_core.retrieval.evidence_context import (
    format_constraint_context,
    format_evidence_for_prompt,
)
from packages.rag_core.retrieval.graders import EvidenceGradingReport, InformationNeedSupport
from packages.rag_core.retrieval.models import EvidenceItem, RetrievalConstraints

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "answer_with_citations.md"


def build_answer_prompt(
    question: str,
    evidence: tuple[EvidenceItem, ...],
    *,
    evidence_grading: EvidenceGradingReport | None = None,
    constraints: RetrievalConstraints | None = None,
) -> str:
    """Build the citation-oriented answer prompt with compact source metadata."""

    effective_constraints = constraints or RetrievalConstraints()
    evidence_block = "\n\n".join(
        format_evidence_for_prompt(item, constraints=effective_constraints)
        for item in sorted(evidence, key=lambda item: item.rank)
    )
    template = load_answer_prompt_template()
    return (
        template.replace("{{ question }}", question)
        .replace("{{ constraint_context }}", format_constraint_context(effective_constraints))
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


def format_evidence(item: EvidenceItem, *, constraints: RetrievalConstraints | None = None) -> str:
    """Compatibility helper retained for tests and external callers."""

    return format_evidence_for_prompt(item, constraints=constraints or RetrievalConstraints())


def source_metadata_bits(item: EvidenceItem) -> list[str]:
    """Compatibility helper for existing UI/prompt tests."""

    bits: list[str] = []
    filename = item.metadata.get("original_filename")
    if isinstance(filename, str) and filename:
        bits.append(f"file={filename}")

    version_label = item.metadata.get("document_version_label")
    version_number = first_int_metadata(item.metadata, "document_version_number")
    if isinstance(version_label, str) and version_label:
        bits.append(f"version={version_label}")
    elif version_number is not None:
        bits.append(f"version=v{version_number}")

    section_title = item.metadata.get("section_title")
    if isinstance(section_title, str) and section_title:
        bits.append(f"section={section_title}")

    page_start = first_int_metadata(item.metadata, "page_number", "source_page_start")
    page_end = first_int_metadata(item.metadata, "source_page_end")
    if page_start is not None and page_end is not None and page_end != page_start:
        bits.append(f"pages={page_start}-{page_end}")
    elif page_start is not None:
        bits.append(f"page={page_start}")
    return bits
