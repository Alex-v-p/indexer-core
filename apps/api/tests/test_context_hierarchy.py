from __future__ import annotations

from packages.rag_core.documents import DocumentChunk, ParsedDocument, ParsedPage
from packages.rag_core.ingestion import (
    ContextHierarchyConfig,
    LLMDocumentContextHierarchyBuilder,
    cluster_chunk_embeddings,
)


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.responses[len(self.prompts) - 1]


def _chunk(ordinal: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        ordinal=ordinal,
        text=text,
        content_hash=f"hash-{ordinal}",
        token_count=len(text.split()),
        source_page_start=ordinal,
        source_page_end=ordinal,
    )


def test_semantic_clustering_groups_nearby_embeddings() -> None:
    embeddings = [
        [1.0, 0.0],
        [0.95, 0.05],
        [0.0, 1.0],
        [0.05, 0.95],
    ]

    groups = cluster_chunk_embeddings(
        embeddings,
        target_cluster_size=2,
        max_clusters=4,
    )

    assert {frozenset(group) for group in groups} == {frozenset({0, 1}), frozenset({2, 3})}


def test_semantic_clustering_respects_max_cluster_count() -> None:
    embeddings = [[float(index), 1.0] for index in range(10)]

    groups = cluster_chunk_embeddings(
        embeddings,
        target_cluster_size=2,
        max_clusters=3,
    )

    assert len(groups) <= 3
    assert sorted(position for group in groups for position in group) == list(range(10))


async def test_hierarchy_builder_summarizes_clusters_then_document() -> None:
    llm = FakeLLM(
        [
            "Authentication and gateway security controls.",
            "Local deployment and container isolation.",
            "The document explains a locally deployed system and its layered security architecture.",
        ],
    )
    builder = LLMDocumentContextHierarchyBuilder(
        llm_provider=llm,
        config=ContextHierarchyConfig(
            target_cluster_size=2,
            max_clusters=4,
            max_cluster_source_chars=2_000,
            max_document_source_chars=2_000,
            max_cluster_summary_chars=300,
            max_document_summary_chars=400,
            max_concurrency=1,
        ),
    )
    chunks = [
        _chunk(1, "Auth0 protects the user-facing gateway."),
        _chunk(2, "Internal inference calls use a service token."),
        _chunk(3, "The deployment runs fully locally."),
        _chunk(4, "Containers isolate the system components."),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.95, 0.05],
        [0.0, 1.0],
        [0.05, 0.95],
    ]
    parsed_document = ParsedDocument(
        title="Security Architecture",
        pages=[ParsedPage(page_number=1, text="Full body is not directly summarized.")],
        parser_name="text",
        parser_version="1.0",
    )

    hierarchy = await builder.build(parsed_document, chunks, embeddings)

    assert len(hierarchy.clusters) == 2
    assert hierarchy.document_summary.startswith("The document explains")
    assert "Auth0 protects" in llm.prompts[0]
    assert "Containers isolate" in llm.prompts[1]
    assert "Authentication and gateway security controls." in llm.prompts[2]
    assert "Local deployment and container isolation." in llm.prompts[2]
    assert "Full body is not directly summarized" not in "\n".join(llm.prompts)
    assert hierarchy.cluster_for_ordinal(1).cluster_id != hierarchy.cluster_for_ordinal(3).cluster_id
