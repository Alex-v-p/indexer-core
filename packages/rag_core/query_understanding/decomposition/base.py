from __future__ import annotations

from typing import Protocol

from packages.rag_core.query_understanding.decomposition.models import InformationNeedDecomposition


class InformationNeedDecomposer(Protocol):
    """Extract independently gradable answer requirements from a question."""

    async def decompose(self, question: str) -> InformationNeedDecomposition:
        """Return one or more information needs for a non-empty question."""
