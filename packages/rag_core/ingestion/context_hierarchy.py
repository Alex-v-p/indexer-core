from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from packages.rag_core.documents import DocumentChunk, ParsedDocument
from packages.rag_core.ports import LLMProvider

_CLUSTER_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "summarize_context_cluster.md"
_DOCUMENT_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "summarize_document_context.md"
_TRUNCATION_MARKER = "\n[... source text truncated ...]\n"


class ContextHierarchyError(RuntimeError):
    """Raised when a document context hierarchy cannot be constructed."""


@dataclass(frozen=True, slots=True)
class ContextHierarchyConfig:
    """Bounds for the semantic summary scaffold used during contextualization."""

    target_cluster_size: int = 8
    max_clusters: int = 24
    max_cluster_source_chars: int = 8_000
    max_document_source_chars: int = 12_000
    max_cluster_summary_chars: int = 600
    max_document_summary_chars: int = 900
    max_concurrency: int = 2

    def __post_init__(self) -> None:
        if self.target_cluster_size <= 0:
            raise ValueError("target_cluster_size must be positive.")
        if self.max_clusters <= 0:
            raise ValueError("max_clusters must be positive.")
        if self.max_cluster_source_chars < 1_000:
            raise ValueError("max_cluster_source_chars must be at least 1000.")
        if self.max_document_source_chars < 1_000:
            raise ValueError("max_document_source_chars must be at least 1000.")
        if self.max_cluster_summary_chars < 100:
            raise ValueError("max_cluster_summary_chars must be at least 100.")
        if self.max_document_summary_chars < 100:
            raise ValueError("max_document_summary_chars must be at least 100.")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive.")


@dataclass(frozen=True, slots=True)
class ContextClusterSummary:
    """Summary of a semantically related group of source chunks."""

    cluster_id: int
    chunk_ordinals: tuple[int, ...]
    summary: str


@dataclass(frozen=True, slots=True)
class DocumentContextHierarchy:
    """A two-level summary scaffold: semantic groups plus one document root."""

    document_summary: str
    clusters: tuple[ContextClusterSummary, ...]

    def cluster_for_ordinal(self, ordinal: int) -> ContextClusterSummary:
        for cluster in self.clusters:
            if ordinal in cluster.chunk_ordinals:
                return cluster
        raise ContextHierarchyError(f"No semantic context cluster contains chunk ordinal {ordinal}.")


class DocumentContextHierarchyBuilder(Protocol):
    """Build broad document context from source chunks and their embeddings."""

    async def build(
        self,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
        chunk_embeddings: list[list[float]],
    ) -> DocumentContextHierarchy:
        """Create semantic group summaries and a root document summary."""


class LLMDocumentContextHierarchyBuilder:
    """Build a small RAPTOR-inspired scaffold without creating a retrieval tree.

    Original chunk embeddings are used only to group semantically related chunks.
    An LLM summarizes each group, then summarizes those group summaries into one
    document-level description. The generated hierarchy is subsequently supplied
    to chunk contextualization; it is not itself indexed or retrieved.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        config: ContextHierarchyConfig | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._config = config or ContextHierarchyConfig()

    async def build(
        self,
        parsed_document: ParsedDocument,
        chunks: list[DocumentChunk],
        chunk_embeddings: list[list[float]],
    ) -> DocumentContextHierarchy:
        if not chunks:
            raise ContextHierarchyError("Cannot build a context hierarchy for an empty chunk set.")
        _validate_embeddings(chunks, chunk_embeddings)

        grouped_positions = cluster_chunk_embeddings(
            chunk_embeddings,
            target_cluster_size=self._config.target_cluster_size,
            max_clusters=self._config.max_clusters,
        )
        semaphore = asyncio.Semaphore(self._config.max_concurrency)

        async def summarize_cluster(cluster_id: int, positions: list[int]) -> ContextClusterSummary:
            cluster_chunks = [chunks[position] for position in positions]
            prompt = build_cluster_summary_prompt(
                document_title=parsed_document.title,
                cluster_id=cluster_id,
                chunks=cluster_chunks,
                max_source_chars=self._config.max_cluster_source_chars,
            )
            async with semaphore:
                raw_summary = await self._llm_provider.generate(prompt)
            summary = _normalize_summary(raw_summary, max_chars=self._config.max_cluster_summary_chars)
            if not summary:
                raise ContextHierarchyError(f"The model returned an empty summary for semantic cluster {cluster_id}.")
            return ContextClusterSummary(
                cluster_id=cluster_id,
                chunk_ordinals=tuple(sorted(chunk.ordinal for chunk in cluster_chunks)),
                summary=summary,
            )

        cluster_summaries = list(
            await asyncio.gather(
                *(
                    summarize_cluster(cluster_id, positions)
                    for cluster_id, positions in enumerate(grouped_positions, start=1)
                ),
            ),
        )
        cluster_summaries.sort(key=lambda item: item.cluster_id)

        document_prompt = build_document_summary_prompt(
            document_title=parsed_document.title,
            cluster_summaries=cluster_summaries,
            max_source_chars=self._config.max_document_source_chars,
        )
        raw_document_summary = await self._llm_provider.generate(document_prompt)
        document_summary = _normalize_summary(
            raw_document_summary,
            max_chars=self._config.max_document_summary_chars,
        )
        if not document_summary:
            raise ContextHierarchyError("The model returned an empty document summary.")

        return DocumentContextHierarchy(
            document_summary=document_summary,
            clusters=tuple(cluster_summaries),
        )


def cluster_chunk_embeddings(
    embeddings: list[list[float]],
    *,
    target_cluster_size: int,
    max_clusters: int,
) -> list[list[int]]:
    """Group semantically similar embeddings with deterministic bounded batches.

    This is deliberately lighter than RAPTOR's recursive Gaussian-mixture tree.
    It computes pairwise cosine similarities once, chooses semantic outliers as
    seeds, and fills each group with that seed's nearest remaining neighbours.
    The effective group size grows only when needed to respect ``max_clusters``.
    """

    if not embeddings:
        return []
    if target_cluster_size <= 0:
        raise ValueError("target_cluster_size must be positive.")
    if max_clusters <= 0:
        raise ValueError("max_clusters must be positive.")

    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("All chunk embeddings must have the same non-zero dimension.")

    normalized = [_normalize_vector(vector) for vector in embeddings]
    count = len(normalized)
    effective_cluster_size = max(target_cluster_size, math.ceil(count / max_clusters))

    similarities = [[0.0] * count for _ in range(count)]
    for left in range(count):
        similarities[left][left] = 1.0
        for right in range(left + 1, count):
            score = _dot(normalized[left], normalized[right])
            similarities[left][right] = score
            similarities[right][left] = score

    if count == 1:
        return [[0]]

    cohesion = [
        sum(similarities[index][other] for other in range(count) if other != index) / (count - 1)
        for index in range(count)
    ]
    remaining = set(range(count))
    groups: list[list[int]] = []

    while remaining:
        seed = min(remaining, key=lambda index: (cohesion[index], index))
        nearest = sorted(
            remaining,
            key=lambda index: (-similarities[seed][index], index),
        )
        group = nearest[:effective_cluster_size]
        groups.append(sorted(group))
        remaining.difference_update(group)

    groups.sort(key=lambda group: min(group))
    return groups


def build_cluster_summary_prompt(
    *,
    document_title: str,
    cluster_id: int,
    chunks: list[DocumentChunk],
    max_source_chars: int,
) -> str:
    return (
        _load_cluster_prompt_template()
        .replace("{{ document_title }}", document_title.strip() or "Untitled document")
        .replace("{{ cluster_id }}", str(cluster_id))
        .replace("{{ cluster_chunks }}", _format_cluster_chunks(chunks, max_chars=max_source_chars))
        .strip()
    )


def build_document_summary_prompt(
    *,
    document_title: str,
    cluster_summaries: list[ContextClusterSummary],
    max_source_chars: int,
) -> str:
    return (
        _load_document_prompt_template()
        .replace("{{ document_title }}", document_title.strip() or "Untitled document")
        .replace(
            "{{ cluster_summaries }}",
            _format_cluster_summaries(cluster_summaries, max_chars=max_source_chars),
        )
        .strip()
    )


@lru_cache(maxsize=1)
def _load_cluster_prompt_template() -> str:
    return _CLUSTER_PROMPT_PATH.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _load_document_prompt_template() -> str:
    return _DOCUMENT_PROMPT_PATH.read_text(encoding="utf-8")


def _validate_embeddings(chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
    if len(embeddings) != len(chunks):
        raise ContextHierarchyError("Chunk embedding count must match the chunk count.")
    dimensions = {len(vector) for vector in embeddings}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ContextHierarchyError("Chunk embeddings must share one non-zero vector dimension.")


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0:
        return [0.0 for _ in vector]
    return [value / magnitude for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _format_cluster_chunks(chunks: list[DocumentChunk], *, max_chars: int) -> str:
    if not chunks:
        return "(none)"

    ordered = sorted(chunks, key=lambda chunk: chunk.ordinal)
    overhead = 120 * len(ordered)
    per_chunk_budget = max(200, (max_chars - overhead) // len(ordered))
    parts: list[str] = []
    for chunk in ordered:
        location = [f"ordinal={chunk.ordinal}"]
        if chunk.section_title:
            location.append(f"section={chunk.section_title}")
        if chunk.source_page_start is not None:
            location.append(f"page={chunk.source_page_start}")
        text = _truncate_middle(chunk.text.strip(), max_chars=per_chunk_budget)
        parts.append(f"<source_chunk {' '.join(location)}>\n{text}\n</source_chunk>")

    combined = "\n\n".join(parts)
    return _truncate_middle(combined, max_chars=max_chars)


def _format_cluster_summaries(
    summaries: list[ContextClusterSummary],
    *,
    max_chars: int,
) -> str:
    if not summaries:
        return "(none)"

    per_summary_budget = max(150, (max_chars - 100 * len(summaries)) // len(summaries))
    parts = [
        (
            f"<semantic_cluster id={summary.cluster_id} "
            f"chunk_ordinals={','.join(str(value) for value in summary.chunk_ordinals)}>\n"
            f"{_truncate_middle(summary.summary, max_chars=per_summary_budget)}\n"
            "</semantic_cluster>"
        )
        for summary in summaries
    ]
    return _truncate_middle("\n\n".join(parts), max_chars=max_chars)


def _truncate_middle(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= len(_TRUNCATION_MARKER):
        return value[:max_chars]
    remaining = max_chars - len(_TRUNCATION_MARKER)
    left_chars = remaining // 2
    right_chars = remaining - left_chars
    return f"{value[:left_chars].rstrip()}{_TRUNCATION_MARKER}{value[-right_chars:].lstrip()}"


def _normalize_summary(value: str, *, max_chars: int) -> str:
    summary = " ".join(value.strip().split()).strip('"\'` ')
    prefixes = ("summary:", "cluster summary:", "document summary:")
    lowered = summary.casefold()
    for prefix in prefixes:
        if lowered.startswith(prefix):
            summary = summary[len(prefix) :].strip()
            break
    return _truncate_at_sentence_boundary(summary, max_chars=max_chars)


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
