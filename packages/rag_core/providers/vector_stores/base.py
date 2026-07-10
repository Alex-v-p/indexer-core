from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """Point payload sent to a vector store."""

    id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """One vector-store search hit returned during retrieval."""

    id: str
    score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Minimal async vector-store contract used by ingestion/retrieval."""

    async def ensure_collection(self) -> None:
        """Create or verify the backing collection/index."""

    async def upsert_points(self, points: list[VectorPoint], *, batch_size: int = 64) -> None:
        """Persist vector points and metadata."""

    async def search_by_vector(self, vector: list[float], *, top_k: int) -> list[VectorSearchResult]:
        """Return nearest-neighbour vector hits ordered by relevance."""
