from pathlib import Path

from packages.rag_core.documents import ChunkingConfig, chunk_document, parse_document
from packages.rag_core.providers import HashingEmbeddingProvider


def test_text_document_parser_and_chunker_keep_metadata(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Intro\n\nIndexer Core stores chunks with metadata. " * 80, encoding="utf-8")

    parsed = parse_document(source, filename="notes.md", content_type="text/markdown")
    chunks = chunk_document(parsed, config=ChunkingConfig(max_chars=400, overlap_chars=50))

    assert parsed.parser_name == "markdown"
    assert len(chunks) > 1
    assert chunks[0].ordinal == 1
    assert chunks[0].content_hash
    assert chunks[0].metadata["chunking_strategy"] == "char_window_overlap"
    assert chunks[0].metadata["parser_name"] == "markdown"


async def test_hashing_embedding_provider_is_deterministic() -> None:
    provider = HashingEmbeddingProvider(vector_size=16)

    first = await provider.embed_texts(["same text"])
    second = await provider.embed_texts(["same text"])

    assert first == second
    assert len(first[0]) == 16
