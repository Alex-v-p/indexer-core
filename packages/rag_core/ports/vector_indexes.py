from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, TypeAlias

from packages.rag_core.documents import DocumentNameConstraint, DocumentVersionConstraint
from packages.rag_core.query_understanding.temporal import DocumentDateConstraint

VectorPayloadValue: TypeAlias = str | int | float | bool


@dataclass(frozen=True, slots=True)
class VectorPayloadCondition:
    """Exact-match payload condition applied by vector-index adapters.

    Conditions are intentionally limited to equality/``any`` matching so core
    retrieval code does not depend on a provider-specific filter language.
    Multiple conditions are combined with AND semantics by the adapter.
    """

    field: str
    values: tuple[VectorPayloadValue, ...]

    def __post_init__(self) -> None:
        if not self.field.strip():
            raise ValueError("Vector payload condition field must not be empty.")
        if not self.values:
            raise ValueError("Vector payload condition values must not be empty.")


@dataclass(frozen=True, slots=True)
class VectorPoint:
    """One logical point with one or more named vector representations."""

    id: str
    vectors: dict[str, list[float]]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """One vector-search hit ordered by descending relevance."""

    id: str
    score: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class VectorSearcher(Protocol):
    """Read capability required by dense-vector retrievers."""

    async def ensure_collection(self) -> None:
        """Create or verify the backing collection/index."""

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        vector_name: str,
        top_k: int,
        document_constraint: DocumentNameConstraint | None = None,
        version_constraint: DocumentVersionConstraint | None = None,
        date_constraints: tuple[DocumentDateConstraint, ...] = (),
        payload_conditions: tuple[VectorPayloadCondition, ...] = (),
    ) -> list[VectorSearchResult]:
        """Return nearest-neighbour hits from the selected named vector."""


class VectorIndexWriter(Protocol):
    """Write capability required by document ingestion."""

    async def ensure_collection(self) -> None:
        """Create or verify the backing collection/index."""

    async def upsert_points(self, points: list[VectorPoint], *, batch_size: int = 64) -> None:
        """Persist logical points containing one or more named vectors."""

    async def mark_document_version_current(self, *, document_id: str, version_id: str) -> None:
        """Mark older indexed versions as superseded after a successful version upsert."""

    async def delete_points(self, point_ids: list[str]) -> None:
        """Delete concrete points by provider-neutral string IDs."""


class VectorStore(VectorSearcher, VectorIndexWriter, Protocol):
    """Combined read/write vector-index capability for implementations that provide both."""


class VectorStoreError(RuntimeError):
    """Raised when a vector-index implementation fails."""
