"""API compatibility facade for query composition."""

from packages.indexer_bootstrap.composition.query import (
    build_query_graph,
    build_query_pipeline_registry,
    build_query_tool_registry,
)

__all__ = [
    "build_query_graph",
    "build_query_pipeline_registry",
    "build_query_tool_registry",
]
