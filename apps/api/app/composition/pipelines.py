"""Backward-compatible query composition facade.

Query tool construction and pipeline-family registration now live under
``app.composition.query``. Existing imports remain valid through this module.
"""

from app.composition.query.registry import build_query_graph, build_query_pipeline_registry
from app.composition.query.tools import build_query_tool_registry

__all__ = [
    "build_query_graph",
    "build_query_pipeline_registry",
    "build_query_tool_registry",
]
