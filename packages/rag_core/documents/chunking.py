from __future__ import annotations

import hashlib
import re

from packages.rag_core.documents.models import ChunkingConfig, DocumentChunk, ParsedDocument, ParsedPage

_WHITESPACE_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(r"^#{1,6}\s+(?P<title>.+)$", flags=re.MULTILINE)


def chunk_document(parsed_document: ParsedDocument, *, config: ChunkingConfig | None = None) -> list[DocumentChunk]:
    """Split a parsed document into stable, metadata-rich chunks.

    Phase 1 uses a simple character window with overlap. It is intentionally
    parser-agnostic and keeps enough metadata to support citations now and more
    advanced retrieval experiments later.
    """

    resolved_config = config or ChunkingConfig()
    _validate_config(resolved_config)

    chunks: list[DocumentChunk] = []
    ordinal = 0
    for page in parsed_document.pages:
        normalized = normalize_text(page.text)
        if not normalized:
            continue

        section_title = _extract_first_heading(page)
        for window in _sliding_windows(normalized, resolved_config):
            ordinal += 1
            metadata = {
                "parser_name": parsed_document.parser_name,
                "parser_version": parsed_document.parser_version,
                "chunking_strategy": "char_window_overlap",
                "chunk_size_chars": resolved_config.max_chars,
                "chunk_overlap_chars": resolved_config.overlap_chars,
                "char_count": len(window),
                **page.metadata,
            }
            if page.page_number is not None:
                metadata["page_number"] = page.page_number
                metadata["source_page_start"] = page.page_number
                metadata["source_page_end"] = page.page_number
            if section_title:
                metadata["section_title"] = section_title

            chunks.append(
                DocumentChunk(
                    ordinal=ordinal,
                    text=window,
                    content_hash=hashlib.sha256(window.encode("utf-8")).hexdigest(),
                    token_count=estimate_token_count(window),
                    source_page_start=page.page_number,
                    source_page_end=page.page_number,
                    section_title=section_title,
                    metadata=metadata,
                ),
            )

    return chunks


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def estimate_token_count(text: str) -> int:
    """Approximate token count without a model-specific tokenizer."""

    word_count = len(text.split())
    return max(1, int(word_count * 1.3)) if text.strip() else 0


def _sliding_windows(text: str, config: ChunkingConfig) -> list[str]:
    if len(text) <= config.max_chars:
        return [text]

    windows: list[str] = []
    start = 0
    step = config.max_chars - config.overlap_chars
    while start < len(text):
        end = min(len(text), start + config.max_chars)
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end == len(text):
            break
        start += step
    return windows


def _extract_first_heading(page: ParsedPage) -> str | None:
    match = _HEADING_RE.search(page.text)
    if not match:
        return None
    return match.group("title").strip()[:512]


def _validate_config(config: ChunkingConfig) -> None:
    if config.max_chars < 200:
        raise ValueError("Chunk max_chars must be at least 200.")
    if config.overlap_chars < 0:
        raise ValueError("Chunk overlap_chars cannot be negative.")
    if config.overlap_chars >= config.max_chars:
        raise ValueError("Chunk overlap_chars must be smaller than max_chars.")
