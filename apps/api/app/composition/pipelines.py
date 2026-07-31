"""Backward-compatible facade for query composition.

The implementation is shared with the worker through
``packages.indexer_bootstrap.composition``.
"""

from packages.indexer_bootstrap.composition.query.registry import (
    build_query_graph,
    build_query_pipeline_registry,
)
from packages.indexer_bootstrap.composition.query.tools import build_query_tool_registry

__all__ = [
    "build_query_graph",
    "build_query_pipeline_registry",
    "build_query_tool_registry",
]
