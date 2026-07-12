from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from packages.rag_core.documents import DocumentChunk, ParsedDocument
from packages.rag_core.ingestion.context_hierarchy import (
    ContextClusterSummary,
    DocumentContextHierarchy,
    DocumentContextHierarchyBuilder,
)
from packages.rag_core.ports import LLMProvider

_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "contextualize_chunk.md"
_TRUNCATION_MARKER = "\n[... adjacent chunk truncated ...]\n"


class ContextualizationError(RuntimeError):
    """Raised when chunk contextualization cannot produce usable output."""


@dataclass(frozen=True, slots=True)
class ContextualizationConfig:
    """Limits for hierarchy- and neighborhood-aware chunk contextualization."""

    neighbor_chunk_count: int = 2
    max_neighbor_chars: int = 6_000
    max_context_chars: int = 400
    max_concurrency: int = 2

    def __post_init__(self) -> None:
        if self.neighbor_chunk_count < 0:
            raise ValueError("neighbor_chunk_count must not be negative.")
        if self.max_neighbor_chars < 500:
            raise ValueError("max_neighbor_chars must be at least 500.")
        if self.max_context_chars < 100:
            raise ValueError("max_context_chars must be at least 100.")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")


@dataclass(frozen=True, slots=True)
class ContextualizedChunk:
    """Original chunk plus the generated retrieval-only representation."""

    chunk: DocumentChunk
    context: str
    contextualized_text: str
    context_cluster_id: int | None = None


@dataclass(frozen=True, slots=True)
class ContextualizationResult:
    """All contextualized chunks plus the hierarchy used to generate them."""

    chunks: list[ContextualizedChunk]
    hierarchy: DocumentContextHierarchy


@dataclass(frozen=True, slots=True)
class _PromptNeighbor:
    chunk: DocumentChunk
    text: str


class ChunkContextualizer(Protocol):
    """Application-facing capability for contextualizing a complete chunk set."""

    async def contextualize(
        self,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
        chunk_embeddings: list[list[float]],
    ) -> ContextualizationResult:
        """Return contextualized representations and their document hierarchy."""


class LLMChunkContextualizer:
    """Generate concise chunk context from local, thematic, and document context.

    A small hierarchy builder first groups original chunk embeddings, summarizes
    each semantic group, and synthesizes a document summary. Per-chunk generation
    then receives that broad context plus nearby chunks that can repair awkward
    boundaries. Only the generated context is prepended to the target chunk for
    contextual embeddings/BM25; original evidence remains unchanged.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        hierarchy_builder: DocumentContextHierarchyBuilder,
        config: ContextualizationConfig | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._hierarchy_builder = hierarchy_builder
        self._config = config or ContextualizationConfig()

    async def contextualize(
        self,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
        chunk_embeddings: list[list[float]],
    ) -> ContextualizationResult:
        if not chunks:
            empty_hierarchy = DocumentContextHierarchy(document_summary="", clusters=())
            return ContextualizationResult(chunks=[], hierarchy=empty_hierarchy)

        hierarchy = await self._hierarchy_builder.build(parsed_document, chunks, chunk_embeddings)
        semaphore = asyncio.Semaphore(self._config.max_concurrency)

        async def contextualize_one(position: int, chunk: DocumentChunk) -> ContextualizedChunk:
            previous_chunks, next_chunks = _select_bounded_neighbors(
                chunks,
                target_position=position,
                neighbor_chunk_count=self._config.neighbor_chunk_count,
                max_neighbor_chars=self._config.max_neighbor_chars,
            )
            cluster = hierarchy.cluster_for_ordinal(chunk.ordinal)
            async with semaphore:
                prompt = build_contextualization_prompt(
                    document_title=parsed_document.title,
                    document_summary=hierarchy.document_summary,
                    semantic_cluster=cluster,
                    chunk=chunk,
                    previous_chunks=previous_chunks,
                    next_chunks=next_chunks,
                )
                raw_context = await self._llm_provider.generate(prompt)
                context = _normalize_context(raw_context, max_chars=self._config.max_context_chars)
                if not context:
                    raise ContextualizationError(f"Contextualizer returned empty context for chunk {chunk.ordinal}.")
                return ContextualizedChunk(
                    chunk=chunk,
                    context=context,
                    contextualized_text=f"{context}\n\n{chunk.text}".strip(),
                    context_cluster_id=cluster.cluster_id,
                )

        contextualized_chunks = list(
            await asyncio.gather(
                *(contextualize_one(position, chunk) for position, chunk in enumerate(chunks)),
            ),
        )
        return ContextualizationResult(chunks=contextualized_chunks, hierarchy=hierarchy)


def build_contextualization_prompt(
    *,
    document_title: str,
    document_summary: str,
    semantic_cluster: ContextClusterSummary,
    chunk: DocumentChunk,
    previous_chunks: list[_PromptNeighbor] | None = None,
    next_chunks: list[_PromptNeighbor] | None = None,
) -> str:
    """Build the hierarchy-, target-, and neighborhood-aware retrieval prompt."""

    location_parts: list[str] = [f"chunk ordinal {chunk.ordinal}"]
    if chunk.section_title:
        location_parts.append(f"section {chunk.section_title}")
    if chunk.source_page_start is not None:
        if chunk.source_page_end is not None and chunk.source_page_end != chunk.source_page_start:
            location_parts.append(f"pages {chunk.source_page_start}-{chunk.source_page_end}")
        else:
            location_parts.append(f"page {chunk.source_page_start}")

    return (
        _load_prompt_template()
        .replace("{{ document_title }}", document_title.strip() or "Untitled document")
        .replace("{{ document_summary }}", document_summary.strip() or "(unavailable)")
        .replace("{{ semantic_cluster_summary }}", semantic_cluster.summary.strip() or "(unavailable)")
        .replace("{{ chunk_location }}", ", ".join(location_parts))
        .replace("{{ previous_chunks }}", _format_neighbors(previous_chunks or []))
        .replace("{{ chunk_text }}", chunk.text.strip())
        .replace("{{ next_chunks }}", _format_neighbors(next_chunks or []))
        .strip()
    )


@lru_cache(maxsize=1)
def _load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _select_bounded_neighbors(
    chunks: list[DocumentChunk],
    *,
    target_position: int,
    neighbor_chunk_count: int,
    max_neighbor_chars: int,
) -> tuple[list[_PromptNeighbor], list[_PromptNeighbor]]:
    """Select a balanced, nearest-first context window around the target.

    Previous chunks keep their trailing text and following chunks keep their
    leading text when a side exceeds its budget. Those boundary edges are most
    useful for repairing awkward chunk cutoffs.
    """

    if neighbor_chunk_count == 0:
        return [], []

    previous_candidates = [
        chunks[position]
        for position in range(target_position - 1, max(-1, target_position - neighbor_chunk_count - 1), -1)
        if position >= 0
    ]
    next_candidates = [
        chunks[position]
        for position in range(target_position + 1, min(len(chunks), target_position + neighbor_chunk_count + 1))
    ]

    if previous_candidates and next_candidates:
        previous_budget = max_neighbor_chars // 2
        next_budget = max_neighbor_chars - previous_budget
    elif previous_candidates:
        previous_budget, next_budget = max_neighbor_chars, 0
    else:
        previous_budget, next_budget = 0, max_neighbor_chars

    previous = _select_neighbor_side(previous_candidates, max_chars=previous_budget, keep_tail=True)
    following = _select_neighbor_side(next_candidates, max_chars=next_budget, keep_tail=False)

    unused_previous = previous_budget - sum(len(item.text) for item in previous)
    unused_next = next_budget - sum(len(item.text) for item in following)
    if unused_previous > 0 and next_candidates:
        following = _select_neighbor_side(
            next_candidates,
            max_chars=next_budget + unused_previous,
            keep_tail=False,
        )
    if unused_next > 0 and previous_candidates:
        previous = _select_neighbor_side(
            previous_candidates,
            max_chars=previous_budget + unused_next,
            keep_tail=True,
        )

    previous.reverse()
    return previous, following


def _select_neighbor_side(
    candidates: list[DocumentChunk],
    *,
    max_chars: int,
    keep_tail: bool,
) -> list[_PromptNeighbor]:
    selected: list[_PromptNeighbor] = []
    remaining = max_chars
    for candidate in candidates:
        if remaining <= 0:
            break
        text = candidate.text.strip()
        if not text:
            continue
        bounded_text = _take_neighbor_text(text, max_chars=remaining, keep_tail=keep_tail)
        if not bounded_text:
            continue
        selected.append(_PromptNeighbor(chunk=candidate, text=bounded_text))
        remaining -= len(bounded_text)
    return selected


def _take_neighbor_text(text: str, *, max_chars: int, keep_tail: bool) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_MARKER):
        return text[-max_chars:] if keep_tail else text[:max_chars]

    content_chars = max_chars - len(_TRUNCATION_MARKER)
    if keep_tail:
        return f"{_TRUNCATION_MARKER}{text[-content_chars:]}"
    return f"{text[:content_chars]}{_TRUNCATION_MARKER}"


def _format_neighbors(neighbors: list[_PromptNeighbor]) -> str:
    if not neighbors:
        return "(none)"

    parts: list[str] = []
    for neighbor in neighbors:
        location = [f"ordinal={neighbor.chunk.ordinal}"]
        if neighbor.chunk.section_title:
            location.append(f"section={neighbor.chunk.section_title}")
        if neighbor.chunk.source_page_start is not None:
            location.append(f"page={neighbor.chunk.source_page_start}")
        parts.append(f"<adjacent_chunk {' '.join(location)}>\n{neighbor.text}\n</adjacent_chunk>")
    return "\n\n".join(parts)


def _normalize_context(value: str, *, max_chars: int) -> str:
    context = " ".join(value.strip().split())
    prefixes = ("context:", "chunk context:", "contextual description:")
    lowered = context.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            context = context[len(prefix) :].strip()
            break
    context = context.strip('"\'` ')
    return _truncate_at_sentence_boundary(context, max_chars=max_chars)


def _truncate_at_sentence_boundary(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    candidate = value[:max_chars].rstrip()
    boundary = max(candidate.rfind(". "), candidate.rfind("? "), candidate.rfind("! "))
    if boundary >= max_chars // 2:
        return candidate[: boundary + 1].rstrip()
    word_boundary = candidate.rfind(" ")
    if word_boundary >= max_chars // 2:
        candidate = candidate[:word_boundary]
    return candidate.rstrip(" ,;:-")
