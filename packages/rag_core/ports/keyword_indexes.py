from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from packages.rag_core.documents import DocumentVersionConstraint
from packages.rag_core.query_understanding.temporal import DocumentDateConstraint


@dataclass(frozen=True, slots=True)
class KeywordDocument:
    """One searchable text document exposed by a keyword corpus source."""

    id: str
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KeywordSearchResult:
    """One lexical-search hit ordered by descending relevance."""

    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class KeywordCorpusSource(Protocol):
    """Source that exposes the complete corpus used to build a keyword index."""

    async def list_documents(self) -> list[KeywordDocument]:
        """Return all currently searchable documents."""


class KeywordStore(Protocol):
    """Async lexical-search capability required by keyword retrievers."""

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        version_constraint: DocumentVersionConstraint | None = None,
        date_constraints: tuple[DocumentDateConstraint, ...] = (),
    ) -> list[KeywordSearchResult]:
        """Return keyword matches ordered by descending relevance."""


class KeywordStoreError(RuntimeError):
    """Raised when a keyword corpus cannot be loaded or searched."""
