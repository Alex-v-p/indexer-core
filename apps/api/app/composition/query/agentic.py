"""API compatibility facade for agentic query composition."""

from packages.indexer_bootstrap.composition.query.agentic import (
    build_agentic_retrieval_executions,
    register_agentic_query_pipeline,
)

__all__ = ["build_agentic_retrieval_executions", "register_agentic_query_pipeline"]
