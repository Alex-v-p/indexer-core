from __future__ import annotations

import uuid

from packages.rag_core.agents import QueryState
from packages.rag_core.agents.retrieval.nodes import RetrieveNode
from packages.rag_core.retrieval import EvidenceItem
from packages.rag_core.retrieval.retrievers import MultiQueryRetriever


class StaticQueryVariantGenerator:
    def __init__(self, variants: list[str] | None = None, error: Exception | None = None) -> None:
        self.variants = variants or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def generate(self, question: str, *, count: int) -> list[str]:
        self.calls.append((question, count))
        if self.error is not None:
            raise self.error
        return self.variants[:count]


class QueryAwareRetriever:
    def __init__(self, results: dict[str, list[EvidenceItem]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        self.calls.append((question, top_k))
        return self.results.get(question, [])[:top_k]


async def test_multi_query_retriever_expands_retrieves_and_fuses_results() -> None:
    shared_id = uuid.uuid4()
    generator = StaticQueryVariantGenerator(
        [
            "reciprocal rank fusion across query variants",
            "RAG retrieval using query paraphrases",
        ],
    )
    base_retriever = QueryAwareRetriever(
        {
            "How does multi-query retrieval work?": [
                EvidenceItem(rank=1, text="original-only evidence", score=0.92),
                EvidenceItem(rank=2, text="shared evidence", score=0.84, qdrant_chunk_index_id=shared_id),
            ],
            "reciprocal rank fusion across query variants": [
                EvidenceItem(rank=1, text="shared evidence", score=0.89, qdrant_chunk_index_id=shared_id),
                EvidenceItem(rank=2, text="variant-only evidence", score=0.80),
            ],
            "RAG retrieval using query paraphrases": [
                EvidenceItem(rank=1, text="shared evidence", score=0.87, qdrant_chunk_index_id=shared_id),
            ],
        },
    )
    retriever = MultiQueryRetriever(
        query_variant_generator=generator,
        retriever=base_retriever,
        variant_count=2,
        candidate_multiplier=2,
        max_candidates_per_query=10,
        rrf_k=60,
    )

    batch = await retriever.retrieve_with_metadata("How does multi-query retrieval work?", top_k=2)

    assert generator.calls == [("How does multi-query retrieval work?", 2)]
    assert base_retriever.calls == [
        ("How does multi-query retrieval work?", 4),
        ("reciprocal rank fusion across query variants", 4),
        ("RAG retrieval using query paraphrases", 4),
    ]
    assert len(batch.evidence) == 2
    assert batch.evidence[0].qdrant_chunk_index_id == shared_id
    assert batch.evidence[0].metadata["retrieval_source"] == "multi_query"
    assert len(batch.evidence[0].metadata["multi_query_fusion"]["matches"]) == 3
    assert batch.metadata["generated_variant_count"] == 2
    assert batch.metadata["query_count"] == 3
    assert batch.metadata["fusion_method"] == "weighted_reciprocal_rank_fusion"


async def test_multi_query_retriever_falls_back_to_original_query_when_expansion_fails() -> None:
    generator = StaticQueryVariantGenerator(error=RuntimeError("model unavailable"))
    base_retriever = QueryAwareRetriever(
        {
            "What is indexed?": [EvidenceItem(rank=1, text="Indexed evidence", score=0.9)],
        },
    )
    retriever = MultiQueryRetriever(
        query_variant_generator=generator,
        retriever=base_retriever,
        variant_count=3,
        fail_open=True,
    )

    batch = await retriever.retrieve_with_metadata("What is indexed?", top_k=1)

    assert [item.text for item in batch.evidence] == ["Indexed evidence"]
    assert base_retriever.calls == [("What is indexed?", 2)]
    assert batch.metadata["generation_fallback_used"] is True
    assert batch.metadata["generation_error"] == "model unavailable"
    assert batch.metadata["queries"][0]["kind"] == "original"


async def test_retrieve_node_copies_multi_query_details_into_query_state_trace_metadata() -> None:
    retriever = MultiQueryRetriever(
        query_variant_generator=StaticQueryVariantGenerator(["expanded search query"]),
        retriever=QueryAwareRetriever(
            {
                "original question": [EvidenceItem(rank=1, text="original evidence")],
                "expanded search query": [EvidenceItem(rank=1, text="expanded evidence")],
            },
        ),
        variant_count=1,
    )
    state = QueryState(question="original question", top_k=2)

    state = await RetrieveNode(retriever)(state)

    assert state.metadata["retrieval"]["strategy"] == "multi_query"
    assert state.metadata["retrieval"]["query_count"] == 2
    assert [query["text"] for query in state.metadata["retrieval"]["queries"]] == [
        "original question",
        "expanded search query",
    ]
    assert state.metadata["retrieval"]["retrieved_count"] == 2


class RecordingAnswerLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Multi-query retrieval expands and fuses search results [1]."


async def test_multi_query_pipeline_trace_summarizes_expansion() -> None:
    from packages.rag_core.pipelines import build_multi_query_rag_graph

    retriever = MultiQueryRetriever(
        query_variant_generator=StaticQueryVariantGenerator(["expanded search query"]),
        retriever=QueryAwareRetriever(
            {
                "original question": [EvidenceItem(rank=1, text="original evidence")],
                "expanded search query": [EvidenceItem(rank=1, text="expanded evidence")],
            },
        ),
        base_retrieval_strategy="hybrid",
        variant_count=1,
    )
    llm = RecordingAnswerLLM()
    graph = build_multi_query_rag_graph(retriever=retriever, llm_provider=llm)

    state = await graph.run(QueryState(question="original question", top_k=2))

    assert [step.name for step in state.trace] == ["select_pipeline", "classify_query", "retrieve", "generate_answer"]
    assert state.trace[2].output_summary == (
        "evidence_count=2; query_count=2; generated_variants=1; generation_fallback=False"
    )
    assert state.metadata["retrieval"]["base_retrieval_strategy"] == "hybrid"
    assert state.answer == "Multi-query retrieval expands and fuses search results [1]."


class PartiallyFailingRetriever:
    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        if question == "failing variant":
            raise RuntimeError("variant lookup failed")
        return [EvidenceItem(rank=1, text=f"evidence for {question}")]


async def test_multi_query_retriever_can_ignore_one_failed_variant_lookup() -> None:
    retriever = MultiQueryRetriever(
        query_variant_generator=StaticQueryVariantGenerator(["failing variant", "working variant"]),
        retriever=PartiallyFailingRetriever(),
        variant_count=2,
        fail_open=True,
    )

    batch = await retriever.retrieve_with_metadata("original question", top_k=3)

    assert {item.text for item in batch.evidence} == {
        "evidence for original question",
        "evidence for working variant",
    }
    assert batch.metadata["retrieval_fail_open_used"] is True
    assert batch.metadata["failed_queries"] == [
        {
            "query_index": 1,
            "query": "failing variant",
            "error": "variant lookup failed",
        },
    ]
