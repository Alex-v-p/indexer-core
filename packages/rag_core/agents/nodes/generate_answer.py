from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from packages.rag_core.agents.state import CitationItem, QueryState
from packages.rag_core.providers.llms import LLMProvider
from packages.rag_core.retrieval.models import EvidenceItem

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "answer_with_citations.md"


class GenerateAnswerNode:
    """Graph node that generates the final answer from retrieved evidence."""

    name = "generate_answer"
    step_type = "generation"

    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    async def __call__(self, state: QueryState) -> QueryState:
        if not state.retrieved_evidence:
            state.answer = (
                "I do not have enough retrieved evidence to answer this question yet. "
                "Upload and index documents first, then ask again."
            )
            state.citations = []
            return state

        prompt = build_answer_prompt(state.question, state.retrieved_evidence)
        state.answer = await self._llm_provider.generate(prompt)
        state.citations = [
            CitationItem(
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
            for item in sorted(state.retrieved_evidence, key=lambda evidence: evidence.rank)
        ]
        return state


def build_answer_prompt(question: str, evidence: list[EvidenceItem]) -> str:
    """Build the citation-oriented answer prompt from the markdown template."""

    evidence_block = "\n\n".join(
        f"[{item.rank}] {item.text.strip()}" for item in sorted(evidence, key=lambda item: item.rank)
    )
    template = _load_answer_prompt_template()
    return template.replace("{{ question }}", question).replace("{{ evidence }}", evidence_block).strip()


@lru_cache(maxsize=1)
def _load_answer_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _first_int_metadata(metadata: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, int):
            return value
    return None
