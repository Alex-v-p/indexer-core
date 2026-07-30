from __future__ import annotations

from packages.rag_core.documents import DocumentChunk, ParsedDocument, ParsedPage
from packages.rag_core.ingestion import (
    ContextClusterSummary,
    ContextualizationConfig,
    DocumentContextHierarchy,
    LLMChunkContextualizer,
)


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[len(self.prompts) - 1]


class FakeHierarchyBuilder:
    def __init__(self, *, document_summary: str = "Document-wide summary") -> None:
        self.document_summary = document_summary
        self.received_embeddings: list[list[float]] | None = None

    async def build(self, parsed_document, chunks, chunk_embeddings):
        self.received_embeddings = chunk_embeddings
        return DocumentContextHierarchy(
            document_summary=self.document_summary,
            clusters=(
                ContextClusterSummary(
                    cluster_id=1,
                    chunk_ordinals=tuple(chunk.ordinal for chunk in chunks),
                    summary="Semantic group summary for the Apollo deployment workflow.",
                ),
            ),
        )


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


async def test_contextualizer_uses_hierarchy_and_adjacent_chunks_without_full_document() -> None:
    llm = FakeLLM(
        [
            "Opening context",
            "Context: Apollo production rollback instruction after a failed post-deployment health check.",
            "Closing context",
        ],
    )
    hierarchy_builder = FakeHierarchyBuilder(document_summary="Apollo deployment and recovery runbook.")
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=hierarchy_builder,
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
    embeddings = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]

    result = await contextualizer.contextualize(parsed_document, chunks, embeddings)

    target_prompt = llm.prompts[1]
    assert "Apollo Runbook" in target_prompt
    assert "Apollo deployment and recovery runbook." in target_prompt
    assert "Semantic group summary for the Apollo deployment workflow." in target_prompt
    assert chunks[0].text in target_prompt
    assert chunks[1].text in target_prompt
    assert chunks[2].text in target_prompt
    assert "full document body" not in target_prompt.casefold()
    assert "section Rollback" in target_prompt
    assert "Write topic-first" in target_prompt
    assert "Never start with" in target_prompt
    assert hierarchy_builder.received_embeddings is embeddings
    assert result.chunks[1].chunk is chunks[1]
    assert result.chunks[1].context == (
        "Apollo production rollback instruction after a failed post-deployment health check."
    )
    assert result.chunks[1].contextualized_text == (
        "Apollo production rollback instruction after a failed post-deployment health check.\n\n"
        "If it fails, run rollback.sh immediately."
    )
    assert result.chunks[1].context_cluster_id == 1
    assert result.hierarchy.document_summary == "Apollo deployment and recovery runbook."


async def test_contextualizer_preserves_chunk_order() -> None:
    llm = FakeLLM(["First context", "Second context"])
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=FakeHierarchyBuilder(),
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

    result = await contextualizer.contextualize(parsed_document, chunks, [[1.0], [0.5]])

    assert [item.chunk.ordinal for item in result.chunks] == [1, 2]
    assert [item.context for item in result.chunks] == ["First context", "Second context"]


async def test_contextualizer_bounds_both_sides_and_prioritizes_boundary_text() -> None:
    llm = FakeLLM(["one", "two", "three"])
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=FakeHierarchyBuilder(),
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

    await contextualizer.contextualize(parsed_document, chunks, [[1.0], [0.5], [0.1]])

    target_prompt = llm.prompts[1]
    assert "PREVIOUS_BOUNDARY" in target_prompt
    assert "PREVIOUS_START" not in target_prompt
    assert "FOLLOWING_BOUNDARY" in target_prompt
    assert "FOLLOWING_END" not in target_prompt
    assert "adjacent chunk truncated" in target_prompt


async def test_contextualizer_truncates_overlong_output_at_a_clean_boundary() -> None:
    llm = FakeLLM(
        [
            "The first sentence adds the missing document context. "
            "The second sentence merely repeats a large amount of unnecessary detail that should be removed."
        ],
    )
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=FakeHierarchyBuilder(),
        config=ContextualizationConfig(max_context_chars=100, max_concurrency=1),
    )
    parsed_document = ParsedDocument(
        title="Notes",
        pages=[ParsedPage(page_number=1, text="Body")],
        parser_name="text",
        parser_version="1.0",
    )

    result = await contextualizer.contextualize(
        parsed_document,
        [_chunk(1, "Target")],
        [[1.0, 0.0]],
    )

    assert result.chunks[0].context == "The first sentence adds the missing document context."


async def test_contextualizer_rewrites_container_first_output_and_removes_meta_commentary() -> None:
    llm = FakeLLM(
        [
            (
                "The document discusses the realization of a layered system architecture, "
                "focusing on deployment and operational readiness. "
                "(This sentence resolves the boundary cutoff in the target chunk.)"
            ),
            (
                "The realization document discusses the development of an LLM Guidance System "
                "for medical professionals. (Note: This response follows the requested word limit."
            ),
        ],
    )
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=FakeHierarchyBuilder(),
        config=ContextualizationConfig(max_context_chars=300, max_concurrency=1),
    )
    parsed_document = ParsedDocument(
        title="Realization Document",
        pages=[ParsedPage(page_number=1, text="Body")],
        parser_name="text",
        parser_version="1.0",
    )
    chunks = [
        _chunk(1, "Deployment source text"),
        _chunk(2, "Guidance-system source text"),
    ]

    result = await contextualizer.contextualize(parsed_document, chunks, [[1.0], [0.5]])

    assert result.chunks[0].context == (
        "Realization of a layered system architecture, focusing on deployment and operational readiness."
    )
    assert result.chunks[1].context == (
        "Realization document — development of an LLM Guidance System for medical professionals."
    )
    assert all(not item.context.casefold().startswith("the document") for item in result.chunks)
    assert all("note:" not in item.context.casefold() for item in result.chunks)
    assert all("this sentence" not in item.context.casefold() for item in result.chunks)


async def test_contextualizer_rewrites_chunk_first_boilerplate_into_a_topic_first_line() -> None:
    llm = FakeLLM(
        [
            'This chunk belongs to the broader subject of "Security" within the Realization Document, '
            "covering authentication and internal service isolation."
        ],
    )
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=FakeHierarchyBuilder(),
        config=ContextualizationConfig(max_context_chars=300, max_concurrency=1),
    )
    parsed_document = ParsedDocument(
        title="Realization Document",
        pages=[ParsedPage(page_number=1, text="Body")],
        parser_name="text",
        parser_version="1.0",
    )

    result = await contextualizer.contextualize(
        parsed_document,
        [_chunk(1, "Authentication source text")],
        [[1.0]],
    )

    assert result.chunks[0].context == (
        "Security within the Realization Document, covering authentication and internal service isolation."
    )


async def test_contextualizer_preserves_domain_language_containing_note() -> None:
    llm = FakeLLM(["Release note for version 2.4 — authentication migration requirements."])
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=FakeHierarchyBuilder(),
        config=ContextualizationConfig(max_context_chars=300, max_concurrency=1),
    )
    parsed_document = ParsedDocument(
        title="Release Guide",
        pages=[ParsedPage(page_number=1, text="Body")],
        parser_name="text",
        parser_version="1.0",
    )

    result = await contextualizer.contextualize(
        parsed_document,
        [_chunk(1, "Migration source text")],
        [[1.0]],
    )

    assert result.chunks[0].context == (
        "Release note for version 2.4 — authentication migration requirements."
    )



async def test_contextualizer_reuses_prebuilt_hierarchy_without_rebuilding_summaries() -> None:
    llm = FakeLLM(["Focused context"])
    hierarchy_builder = FakeHierarchyBuilder(document_summary="This builder should not run.")
    contextualizer = LLMChunkContextualizer(
        llm_provider=llm,
        hierarchy_builder=hierarchy_builder,
        config=ContextualizationConfig(max_concurrency=1),
    )
    chunk = _chunk(1, "Target source chunk.")
    parsed_document = ParsedDocument(
        title="Shared Hierarchy",
        pages=[ParsedPage(page_number=1, text=chunk.text)],
        parser_name="text",
        parser_version="1.0",
    )
    hierarchy = DocumentContextHierarchy(
        document_summary="Prebuilt document summary.",
        clusters=(
            ContextClusterSummary(
                cluster_id=7,
                chunk_ordinals=(1,),
                summary="Prebuilt semantic section summary.",
            ),
        ),
    )

    result = await contextualizer.contextualize(
        parsed_document,
        [chunk],
        [[1.0, 0.0]],
        hierarchy=hierarchy,
    )

    assert hierarchy_builder.received_embeddings is None
    assert result.hierarchy is hierarchy
    assert result.chunks[0].context_cluster_id == 7
    assert "Prebuilt document summary." in llm.prompts[0]
    assert "Prebuilt semantic section summary." in llm.prompts[0]
