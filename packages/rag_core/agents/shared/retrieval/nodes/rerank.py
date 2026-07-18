from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.retrieval.rerankers import Reranker


class RerankNode:
    """Graph node that reranks retrieved candidates before answer generation."""

    name = "rerank"
    step_type = "reranking"

    def __init__(self, reranker: Reranker) -> None:
        self._reranker = reranker

    async def __call__(self, state: QueryState) -> QueryState:
        candidate_count = len(state.retrieved_evidence)
        retrieval_query = state.effective_retrieval_query
        retrieval_top_k = state.effective_retrieval_top_k
        state.retrieved_evidence = await self._reranker.rerank(
            retrieval_query,
            state.retrieved_evidence,
            top_k=retrieval_top_k,
        )
        rerank_metadata = [
            item.metadata.get("rerank")
            for item in state.retrieved_evidence
            if isinstance(item.metadata.get("rerank"), dict)
        ]
        providers = sorted(
            {
                str(metadata["provider"])
                for metadata in rerank_metadata
                if metadata.get("provider") is not None
            },
        )
        state.metadata["reranking"] = {
            "candidate_count": candidate_count,
            "result_count": len(state.retrieved_evidence),
            "top_k": retrieval_top_k,
            "providers": providers,
            "fallback_count": sum(bool(metadata.get("fallback_used")) for metadata in rerank_metadata),
        }
        return state
