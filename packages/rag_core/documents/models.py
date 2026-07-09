from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ParsedPage:
    """Single page or logical source unit extracted from a document."""

    page_number: int | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Parser output before chunking."""

    title: str
    pages: list[ParsedPage]
    parser_name: str
    parser_version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Chunk produced by basic ingestion and ready for vector indexing."""

    ordinal: int
    text: str
    content_hash: str
    token_count: int
    source_page_start: int | None = None
    source_page_end: int | None = None
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Character-window chunking settings.

    This intentionally avoids tokenizer-specific dependencies in Phase 1 while
    still keeping chunk size and overlap explicit and configurable.
    """

    max_chars: int = 1200
    overlap_chars: int = 200
