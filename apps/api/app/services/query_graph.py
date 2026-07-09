from __future__ import annotations

from app.core.config import Settings
from packages.rag_core.agents import GraphRunner
from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.providers import OllamaLLMProvider
from packages.rag_core.retrieval.retrievers import EmptyRetriever


def build_query_graph(settings: Settings) -> GraphRunner:
    """Wire the current query graph dependencies.

    Retrieval is intentionally an EmptyRetriever until the ingestion/vector-store
    phase lands. The generation dependency is already Ollama-based so the graph
    stays local/container friendly.
    """

    return build_baseline_rag_graph(
        retriever=EmptyRetriever(),
        llm_provider=OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        ),
    )
