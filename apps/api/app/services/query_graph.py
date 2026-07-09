from __future__ import annotations

from app.adapters.embeddings import build_embedding_provider
from app.adapters.vector_store import build_vector_store
from app.core.config import Settings
from packages.rag_core.agents import GraphRunner
from packages.rag_core.pipelines import build_baseline_rag_graph
from packages.rag_core.providers.llms import OllamaLLMProvider
from packages.rag_core.retrieval.retrievers import VectorRetriever


def build_query_graph(settings: Settings) -> GraphRunner:
    """Wire the current baseline query graph dependencies.

    The API still depends only on the graph runner boundary. The graph now uses
    the real dense-vector retriever, which embeds the question, searches Qdrant,
    and passes normalized evidence into answer generation.
    """

    return build_baseline_rag_graph(
        retriever=VectorRetriever(
            embedding_provider=build_embedding_provider(settings),
            vector_store=build_vector_store(settings),
        ),
        llm_provider=OllamaLLMProvider(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        ),
    )
