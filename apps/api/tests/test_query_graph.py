from __future__ import annotations

from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.query import EvidenceItem, QueryState
from packages.rag_core.retrieval.retrievers import EmptyRetriever


class RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Graph runners use retrieved evidence [1]."


class StaticRetriever:
    async def retrieve(self, question: str, *, top_k: int) -> list[EvidenceItem]:
        return [
            EvidenceItem(rank=1, text="Graph runners execute RAG as nodes.", score=0.91, metadata={"page_number": 3}),
            EvidenceItem(rank=2, text="QueryState is shared between nodes.", score=0.82),
        ][:top_k]


async def test_baseline_graph_runs_retrieve_then_generate_answer() -> None:
    llm = RecordingLLM()
    graph = build_baseline_rag_graph(retriever=StaticRetriever(), llm_provider=llm)

    state = await graph.run(QueryState(question="How does the query graph work?", top_k=2))

    assert state.answer == "Graph runners use retrieved evidence [1]."
    assert [step.name for step in state.trace] == ["retrieve", "generate_answer"]
    assert [step.status for step in state.trace] == ["succeeded", "succeeded"]
    assert len(state.retrieved_evidence) == 2
    assert [citation.label for citation in state.citations] == ["[1]", "[2]"]
    assert "[1] Graph runners execute RAG as nodes." in llm.prompts[0]


async def test_empty_retrieval_returns_safe_no_evidence_answer_without_llm_call() -> None:
    llm = RecordingLLM()
    graph = build_baseline_rag_graph(retriever=EmptyRetriever(), llm_provider=llm)

    state = await graph.run(QueryState(question="What is in the documents?"))

    assert state.answer is not None
    assert "not have enough retrieved evidence" in state.answer
    assert state.citations == []
    assert llm.prompts == []
    assert [step.name for step in state.trace] == ["retrieve", "generate_answer"]
