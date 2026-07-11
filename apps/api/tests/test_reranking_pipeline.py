from __future__ import annotations

from collections.abc import Sequence

from packages.rag_core.agents import QueryState
from packages.rag_core.pipelines import build_hybrid_rerank_rag_graph
from packages.rag_core.retrieval import EvidenceItem


class RecordingRetriever:
    def __init__(self) -> None:
        self.top_k_calls: list[int] = []

    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        del question
        self.top_k_calls.append(top_k)
        return [
            EvidenceItem(rank=1, text="lower relevance", score=0.8),
            EvidenceItem(rank=2, text="direct answer", score=0.7),
            EvidenceItem(rank=3, text="supporting detail", score=0.6),
        ][:top_k]


class StaticReranker:
    def __init__(self) -> None:
        self.candidate_counts: list[int] = []

    async def rerank(
        self,
        question: str,
        evidence: Sequence[EvidenceItem],
        *,
        top_k: int,
    ) -> list[EvidenceItem]:
        del question
        self.candidate_counts.append(len(evidence))
        ordered = [evidence[1], evidence[2], evidence[0]][:top_k]
        return [
            EvidenceItem(
                rank=rank,
                text=item.text,
                score=1.0 - (rank * 0.1),
                metadata={**item.metadata, "rerank": {"original_rank": item.rank}},
            )
            for rank, item in enumerate(ordered, start=1)
        ]


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "The direct answer is supported [1]."


async def test_hybrid_rerank_graph_retrieves_candidates_then_reranks_to_requested_top_k() -> None:
    retriever = RecordingRetriever()
    reranker = StaticReranker()
    llm = RecordingLLM()
    graph = build_hybrid_rerank_rag_graph(
        retriever=retriever,
        reranker=reranker,
        llm_provider=llm,
        candidate_multiplier=3,
        max_candidates=10,
    )

    state = await graph.run(QueryState(question="What is the answer?", top_k=2))

    assert retriever.top_k_calls == [6]
    assert reranker.candidate_counts == [3]
    assert [item.text for item in state.retrieved_evidence] == ["direct answer", "supporting detail"]
    assert [step.name for step in state.trace] == ["select_pipeline", "retrieve", "rerank", "generate_answer"]
    assert state.metadata["retrieval"]["candidate_top_k"] == 6
    assert state.metadata["reranking"] == {"candidate_count": 3, "result_count": 2, "top_k": 2}
    assert [citation.label for citation in state.citations] == ["[1]", "[2]"]
    assert "direct answer" in llm.prompts[0]
    assert "lower relevance" not in llm.prompts[0]
