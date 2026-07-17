from __future__ import annotations

from packages.rag_core.query_understanding.classification import HeuristicQueryClassifier, QueryClassifier
from packages.rag_core.agents.graph import (
    NodeSpec,
    classification_summary,
    classification_trace_metadata,
    question_summary,
)
from packages.rag_core.agents.nodes import ClassifyQueryNode


def build_query_classification_node(classifier: QueryClassifier | None = None) -> NodeSpec:
    """Build the shared first node used by every query pipeline."""

    return NodeSpec(
        node=ClassifyQueryNode(classifier or HeuristicQueryClassifier()),
        input_summary=question_summary,
        output_summary=classification_summary,
        trace_metadata=classification_trace_metadata,
    )
