from __future__ import annotations

import uuid
from dataclasses import dataclass

from packages.rag_core.documents import DocumentNameConstraint
from packages.rag_core.ports import VectorPayloadCondition, VectorSearchResult
from packages.rag_core.document_scope import DocumentScope
from packages.rag_core.retrieval.models import RetrievalConstraints
from packages.rag_core.retrieval.retrievers import HierarchicalRetriever, HierarchicalRetrieverConfig


class FakeEmbeddingProvider:
    vector_size = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


@dataclass(frozen=True)
class SearchCall:
    vector_name: str
    top_k: int
    document_constraint: object
    version_constraint: object
    date_constraints: tuple[object, ...]
    document_scope: DocumentScope
    payload_conditions: tuple[VectorPayloadCondition, ...]


class FakeVectorStore:
    def __init__(self, responses: list[list[VectorSearchResult]]) -> None:
        self.responses = list(responses)
        self.searches: list[SearchCall] = []
        self.ensure_calls = 0

    async def ensure_collection(self) -> None:
        self.ensure_calls += 1

    async def search_by_vector(
        self,
        vector: list[float],
        *,
        vector_name: str,
        top_k: int,
        document_constraint=None,
        version_constraint=None,
        date_constraints=(),
        document_scope=DocumentScope(),
        payload_conditions: tuple[VectorPayloadCondition, ...] = (),
    ) -> list[VectorSearchResult]:
        assert vector == [0.1, 0.2, 0.3]
        self.searches.append(
            SearchCall(
                vector_name=vector_name,
                top_k=top_k,
                document_constraint=document_constraint,
                version_constraint=version_constraint,
                date_constraints=date_constraints,
                document_scope=document_scope,
                payload_conditions=payload_conditions,
            ),
        )
        return self.responses.pop(0)


def _hit(*, point_id: str, score: float, **payload) -> VectorSearchResult:
    return VectorSearchResult(id=point_id, score=score, payload=payload)


async def test_hierarchical_retriever_routes_document_to_section_to_source_chunk() -> None:
    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    chunk_index_id = uuid.uuid4()
    section_id = f"{version_id}:section:2"
    embeddings = FakeEmbeddingProvider()
    store = FakeVectorStore(
        responses=[
            [
                _hit(
                    point_id="document-summary",
                    score=0.93,
                    document_id=str(document_id),
                    document_version_id=str(version_id),
                    hierarchy_level="document",
                    summary_text="The project implements an agent-ready retrieval system.",
                ),
            ],
            [
                _hit(
                    point_id="section-summary",
                    score=0.90,
                    document_id=str(document_id),
                    document_version_id=str(version_id),
                    hierarchy_level="section",
                    hierarchy_section_id=section_id,
                    context_cluster_id=2,
                    summary_text="This section explains hierarchical retrieval and routing.",
                ),
            ],
            [
                _hit(
                    point_id="source-chunk",
                    score=0.87,
                    text="Hierarchical retrieval first narrows documents, then sections, then chunks.",
                    qdrant_chunk_index_id=str(chunk_index_id),
                    document_id=str(document_id),
                    document_version_id=str(version_id),
                    document_version_number=3,
                    hierarchy_section_id=section_id,
                    context_cluster_id=2,
                    source_page_start=14,
                ),
            ],
        ],
    )
    retriever = HierarchicalRetriever(
        embedding_provider=embeddings,
        vector_store=store,
        hierarchy_vector_name="hierarchy",
        chunk_vector_name="contextual",
        fallback_chunk_vector_name="original",
        config=HierarchicalRetrieverConfig(
            document_candidates=4,
            section_candidates=6,
            chunk_candidate_multiplier=3,
            max_chunk_candidates=20,
        ),
    )
    document_constraint = DocumentNameConstraint(
        names=("Indexer Core",),
        confidence=1.0,
        rationale="Explicit title.",
        detector_name="test",
    )

    strict_scope = DocumentScope.strict_scope((document_id,))
    batch = await retriever.retrieve_with_metadata(
        "How does hierarchical retrieval work?",
        top_k=2,
        constraints=RetrievalConstraints(
            document=document_constraint,
            document_scope=strict_scope,
        ),
    )

    assert embeddings.calls == [["How does hierarchical retrieval work?"]]
    assert store.ensure_calls == 1
    assert [call.vector_name for call in store.searches] == ["hierarchy", "hierarchy", "contextual"]
    assert store.searches[0].payload_conditions == (
        VectorPayloadCondition("point_type", ("hierarchy_summary",)),
        VectorPayloadCondition("hierarchy_level", ("document",)),
    )
    assert store.searches[1].payload_conditions[-1] == VectorPayloadCondition(
        "document_version_id",
        (str(version_id),),
    )
    assert store.searches[2].payload_conditions == (
        VectorPayloadCondition("retrieval_level", ("chunk",)),
        VectorPayloadCondition("hierarchy_section_id", (section_id,)),
    )
    assert all(call.document_constraint == document_constraint for call in store.searches)
    assert all(call.document_scope == strict_scope for call in store.searches)

    assert len(batch.evidence) == 1
    evidence = batch.evidence[0]
    assert evidence.text.startswith("Hierarchical retrieval first narrows")
    assert evidence.qdrant_chunk_index_id == chunk_index_id
    assert evidence.document_id == document_id
    assert evidence.document_version_id == version_id
    assert evidence.metadata["retrieval_source"] == "hierarchical"
    path = evidence.metadata["hierarchical_retrieval"]
    assert path["document_summary_score"] == 0.93
    assert path["section_summary_score"] == 0.90
    assert path["hierarchy_section_id"] == section_id
    assert path["chunk_vector_name"] == "contextual"
    assert path["chunk_vector_fallback_used"] is False

    assert batch.metadata["strategy"] == "hierarchical_document_section_chunk"
    assert batch.metadata["document_summary_candidate_count"] == 1
    assert batch.metadata["section_summary_candidate_count"] == 1
    assert batch.metadata["chunk_candidate_count"] == 1
    assert batch.metadata["selected_document_version_ids"] == [str(version_id)]
    assert batch.metadata["selected_hierarchy_section_ids"] == [section_id]
    assert batch.metadata["stop_reason"] is None


async def test_hierarchical_retriever_falls_back_to_original_chunk_vector() -> None:
    version_id = uuid.uuid4()
    section_id = f"{version_id}:section:1"
    store = FakeVectorStore(
        responses=[
            [_hit(point_id="document", score=0.8, document_version_id=str(version_id))],
            [
                _hit(
                    point_id="section",
                    score=0.78,
                    document_version_id=str(version_id),
                    hierarchy_section_id=section_id,
                ),
            ],
            [],
            [
                _hit(
                    point_id="chunk",
                    score=0.7,
                    text="Original-vector evidence.",
                    document_version_id=str(version_id),
                    hierarchy_section_id=section_id,
                ),
            ],
        ],
    )
    retriever = HierarchicalRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        hierarchy_vector_name="hierarchy",
        chunk_vector_name="contextual",
        fallback_chunk_vector_name="original",
    )

    batch = await retriever.retrieve_with_metadata("question", top_k=1)

    assert [call.vector_name for call in store.searches] == [
        "hierarchy",
        "hierarchy",
        "contextual",
        "original",
    ]
    assert all(call.document_scope.is_global for call in store.searches)
    assert batch.evidence[0].text == "Original-vector evidence."
    assert batch.evidence[0].metadata["hierarchical_retrieval"]["chunk_vector_fallback_used"] is True
    assert batch.metadata["chunk_vector_name"] == "original"
    assert batch.metadata["chunk_vector_fallback_used"] is True


async def test_hierarchical_retriever_stops_when_no_document_summary_matches() -> None:
    store = FakeVectorStore(responses=[[]])
    retriever = HierarchicalRetriever(
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
        hierarchy_vector_name="hierarchy",
        chunk_vector_name="original",
    )

    batch = await retriever.retrieve_with_metadata("unmatched question", top_k=3)

    assert batch.evidence == []
    assert len(store.searches) == 1
    assert store.searches[0].document_scope.is_global
    assert batch.metadata["stop_reason"] == "no_document_summaries"
