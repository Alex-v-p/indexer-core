from __future__ import annotations

from packages.rag_core.documents import DocumentChunk, ParsedDocument, ParsedPage
from packages.rag_core.ingestion import ContextualizationConfig, LLMChunkContextualizer


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[len(self.prompts) - 1]


def _chunk(ordinal: int, text: str, *, section: str | None = None) -> DocumentChunk:
    return DocumentChunk(
        ordinal=ordinal,
        text=text,
        content_hash=f"hash-{ordinal}",
        token_count=max(1, len(text.split())),
        source_page_start=ordinal,
        source_page_end=ordinal,
        section_title=section,
    )


async def test_contextualizer_uses_adjacent_chunks_without_repeating_full_document() -> None:
    llm = FakeLLM(
        [
            "Opening context",
            "Context: This chunk describes the rollback action for the Apollo deployment.",
            "Closing context",
        ],
    )
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        config=ContextualizationConfig(
            neighbor_chunk_count=1,
            max_neighbor_chars=2_000,
            max_context_chars=300,
            max_concurrency=1,
        ),
    )
    parsed_document = ParsedDocument(
        title="Apollo Runbook",
        pages=[ParsedPage(page_number=1, text="A full document body that must not be repeated in every prompt.")],
        parser_name="text",
        parser_version="1.0",
    )
    chunks = [
        _chunk(1, "The Apollo release is validated through a post-deployment health check."),
        _chunk(2, "If it fails, run rollback.sh immediately.", section="Rollback"),
        _chunk(3, "After rollback, verify that the previous service version is healthy."),
    ]

    result = await contextualizer.contextualize(parsed_document, chunks)

    target_prompt = llm.prompts[1]
    assert "Apollo Runbook" in target_prompt
    assert chunks[0].text in target_prompt
    assert chunks[1].text in target_prompt
    assert chunks[2].text in target_prompt
    assert "full document body" not in target_prompt.casefold()
    assert "<document>" not in target_prompt
    assert "section Rollback" in target_prompt
    assert result[1].chunk is chunks[1]
    assert result[1].context == "This chunk describes the rollback action for the Apollo deployment."
    assert result[1].contextualized_text == (
        "This chunk describes the rollback action for the Apollo deployment.\n\n"
        "If it fails, run rollback.sh immediately."
    )


async def test_contextualizer_preserves_chunk_order() -> None:
    llm = FakeLLM(["First context", "Second context"])
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        config=ContextualizationConfig(max_concurrency=2),
    )
    parsed_document = ParsedDocument(
        title="Notes",
        pages=[ParsedPage(page_number=None, text="A complete document.")],
        parser_name="text",
        parser_version="1.0",
    )
    chunks = [
        _chunk(1, "First chunk"),
        _chunk(2, "Second chunk"),
    ]

    result = await contextualizer.contextualize(parsed_document, chunks)

    assert [item.chunk.ordinal for item in result] == [1, 2]
    assert [item.context for item in result] == ["First context", "Second context"]


async def test_contextualizer_bounds_both_sides_and_prioritizes_boundary_text() -> None:
    llm = FakeLLM(["one", "two", "three"])
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        config=ContextualizationConfig(
            neighbor_chunk_count=1,
            max_neighbor_chars=500,
            max_context_chars=200,
            max_concurrency=1,
        ),
    )
    previous = "PREVIOUS_START " + ("p" * 500) + " PREVIOUS_BOUNDARY"
    following = "FOLLOWING_BOUNDARY " + ("n" * 500) + " FOLLOWING_END"
    chunks = [
        _chunk(1, previous),
        _chunk(2, "Target chunk"),
        _chunk(3, following),
    ]
    parsed_document = ParsedDocument(
        title="Boundary Test",
        pages=[ParsedPage(page_number=1, text="Unused full document")],
        parser_name="text",
        parser_version="1.0",
    )

    await contextualizer.contextualize(parsed_document, chunks)

    target_prompt = llm.prompts[1]
    assert "PREVIOUS_BOUNDARY" in target_prompt
    assert "PREVIOUS_START" not in target_prompt
    assert "FOLLOWING_BOUNDARY" in target_prompt
    assert "FOLLOWING_END" not in target_prompt
    assert "adjacent chunk truncated" in target_prompt
