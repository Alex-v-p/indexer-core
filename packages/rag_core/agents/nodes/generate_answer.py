from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from packages.rag_core.agents.state import CitationItem, QueryState
from packages.rag_core.ports import LLMProvider
from packages.rag_core.retrieval.models import EvidenceItem

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "answer_with_citations.md"


class GenerateAnswerNode:
    """Graph node that generates the final answer from retrieved evidence."""

    name = "generate_answer"
    step_type = "generation"

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    async def __call__(self, state: QueryState) -> QueryState:
        if state.evidence_grading is not None and not state.evidence_grading.sufficient:
            status = state.evidence_grading.status.value
            state.answer = (
                f"The retrieved evidence was graded as {status} and is not sufficient to answer "
                "the question reliably."
            )
            state.citations = []
            state.metadata = {
                **state.metadata,
                "evidence_count": len(state.retrieved_evidence),
                "citation_count": 0,
                "answer_blocked_by_evidence_grading": True,
            }
            return state

        evidence = [item for item in state.retrieved_evidence if item.text.strip()]
        if not evidence:
            state.answer = (
                "I do not have enough retrieved evidence to answer this question yet. "
                "Upload and index documents first, then ask again."
            )
            state.citations = []
            state.metadata = {**state.metadata, "evidence_count": 0}
            return state

        prompt = build_answer_prompt(state.question, evidence)
        state.answer = await self._llm_provider.generate(prompt)
        state.citations = [_to_citation(item) for item in sorted(evidence, key=lambda evidence: evidence.rank)]
        state.metadata = {**state.metadata, "evidence_count": len(evidence), "citation_count": len(state.citations)}
        return state


def build_answer_prompt(question: str, evidence: list[EvidenceItem]) -> str:
    """Build the citation-oriented answer prompt from the markdown template."""

    evidence_block = "\n\n".join(_format_evidence(item) for item in sorted(evidence, key=lambda item: item.rank))
    template = _load_answer_prompt_template()
    return template.replace("{{ question }}", question).replace("{{ evidence }}", evidence_block).strip()


@lru_cache(maxsize=1)
def _load_answer_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


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
