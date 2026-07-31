"""Backward-compatible query composition facade.

Query tool construction and pipeline-family registration now live under
``packages.indexer_bootstrap.composition.query``. Deployable apps may expose their own thin facades.
"""

from packages.indexer_bootstrap.composition.query.registry import build_query_graph, build_query_pipeline_registry
from packages.indexer_bootstrap.composition.query.tools import build_query_tool_registry

__all__ = [
    "build_query_graph",
    "build_query_pipeline_registry",
    "build_query_tool_registry",
]
