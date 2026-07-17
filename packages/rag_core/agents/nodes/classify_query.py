from __future__ import annotations

from packages.rag_core.query_understanding.classification import QueryClassifier
from packages.rag_core.agents.state import QueryState


class ClassifyQueryNode:
    """Graph node that classifies the user's information need before retrieval."""

    name = "classify_query"
    step_type = "classification"

    def __init__(self, classifier: QueryClassifier) -> None:
        self._classifier = classifier

    async def __call__(self, state: QueryState) -> QueryState:
        classification = await self._classifier.classify(state.question)
        state.query_classification = classification
        state.metadata["query_classification"] = classification.to_metadata()
        return state
