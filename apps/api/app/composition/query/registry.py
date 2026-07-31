"""API compatibility facade for the query pipeline registry."""

from packages.indexer_bootstrap.composition.query.registry import (
    build_query_graph,
    build_query_pipeline_registry,
)

__all__ = ["build_query_graph", "build_query_pipeline_registry"]
