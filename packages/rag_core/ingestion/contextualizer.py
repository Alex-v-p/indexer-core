from __future__ import annotations

import asyncio
import re
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
_CONTAINER_STATEMENT_RE = re.compile(
    r"^(?:the|this)\s+(?P<label>(?:(?:[\w-]+)\s+){0,3})"
    r"(?P<container>document|report|paper|file|text|section|chapter|chunk|passage|excerpt)\s+"
    r"(?:mainly\s+)?(?:discusses|describes|covers|explains|outlines|presents|"
    r"focuses\s+on|concerns|details|summarizes|addresses|contains|is\s+about|"
    r"provides(?:\s+an?\s+overview\s+of)?)\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_CONTAINER_LOCATION_RE = re.compile(
    r"^(?:within|in|according\s+to)\s+(?:the|this)\s+"
    r"(?P<label>(?:(?:[\w-]+)\s+){0,3})"
    r"(?P<container>document|report|paper|file|text)\s*[,;:]?\s*(?P<body>.+)$",
    re.IGNORECASE,
)
_CONTAINER_POSSESSIVE_RE = re.compile(
    r"^(?:the|this)\s+(?P<label>(?:(?:[\w-]+)\s+){0,3})"
    r"(?P<container>document|report|paper|file|text)[’']s\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_CHUNK_BELONGS_RE = re.compile(
    r"^(?:the|this)\s+(?:target\s+|source\s+)?(?:chunk|passage|excerpt)\s+"
    r"(?:belongs\s+to|sits\s+within|appears\s+in)\s+(?:the\s+)?"
    r"(?:broader\s+)?(?:subject|topic|context|section)\s+of\s+(?P<body>.+)$",
    re.IGNORECASE,
)
_META_COMMENTARY_RE = re.compile(
    r"(?:^|\s*[\[(]\s*|(?<=[.!?])\s+)(?:note\s*:|output\s+note\s*:|"
    r"this\s+sentence\b|the\s+sentence\b|this\s+output\b|the\s+output\b|"
    r"this\s+response\b|the\s+response\b|word\s+count\b|word\s+limit\b|"
    r"compliance\s+note\b)",
    re.IGNORECASE,
)


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
        hierarchy: DocumentContextHierarchy | None = None,
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
        hierarchy: DocumentContextHierarchy | None = None,
    ) -> ContextualizationResult:
        if not chunks:
            empty_hierarchy = DocumentContextHierarchy(document_summary="", clusters=())
            return ContextualizationResult(chunks=[], hierarchy=empty_hierarchy)

        if hierarchy is None:
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
    prefixes = ("context:", "chunk context:", "contextual description:", "retrieval context:")
    lowered = context.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            context = context[len(prefix) :].strip()
            break

    context = _strip_meta_commentary(context)
    context = _strip_generic_context_lead(context)
    context = context.strip('"\'` ')
    context = _capitalize_first_alpha(context)
    return _truncate_at_sentence_boundary(context, max_chars=max_chars)


def _strip_meta_commentary(value: str) -> str:
    """Remove model commentary about its own answer rather than the source."""

    match = _META_COMMENTARY_RE.search(value)
    if match is None:
        return value
    return value[: match.start()].rstrip(" \t,;:-")


def _strip_generic_context_lead(value: str) -> str:
    """Rewrite container-first prose into a topic-first retrieval prefix."""

    context = value.strip()
    belongs_match = _CHUNK_BELONGS_RE.match(context)
    if belongs_match is not None:
        return _clean_rewritten_body(belongs_match.group("body"))

    for pattern in (_CONTAINER_STATEMENT_RE, _CONTAINER_LOCATION_RE, _CONTAINER_POSSESSIVE_RE):
        match = pattern.match(context)
        if match is None:
            continue
        return _render_topic_first_context(
            label=match.group("label"),
            container=match.group("container"),
            body=match.group("body"),
        )
    return context


def _render_topic_first_context(*, label: str, container: str, body: str) -> str:
    cleaned_body = _clean_rewritten_body(body)
    cleaned_label = " ".join(label.split()).strip()
    generic_labels = {"", "current", "given", "provided", "source", "target"}
    if cleaned_label.casefold() in generic_labels:
        return cleaned_body
    return f"{cleaned_label} {container.casefold()} — {cleaned_body}"


def _clean_rewritten_body(value: str) -> str:
    body = value.lstrip(" \t,;:-")
    body = re.sub(r"^(?:the|a|an)\s+", "", body, count=1, flags=re.IGNORECASE)
    return re.sub(r"^[\"'`](.+?)[\"'`](?=\s|$)", r"\1", body, count=1)


def _capitalize_first_alpha(value: str) -> str:
    for index, character in enumerate(value):
        if character.isalpha():
            return f"{value[:index]}{character.upper()}{value[index + 1:]}"
    return value


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
