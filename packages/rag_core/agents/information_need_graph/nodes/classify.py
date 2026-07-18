from __future__ import annotations

from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.query_understanding.classification import QueryClassifier


class ClassifyInformationNeedNode:
    """Classify the active information item independently from the original query."""

    name = "classify_information_need"
    step_type = "information_need_classification"

    def __init__(self, classifier: QueryClassifier) -> None:
        self._classifier = classifier

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None:
            raise RuntimeError("Information-need classification requires an active work item.")
        classification = await self._classifier.classify(execution.information_need.retrieval_query)
        if execution.classification is not None:
            execution.reclassifications_used += 1
        execution.classification = classification
        execution.classification_history.append(classification)
        execution.next_route = None
        state.metadata["active_information_need_classification"] = {
            "information_need_id": execution.information_need.need_id,
            "classification": classification.to_metadata(),
            "classification_count": len(execution.classification_history),
        }
        return state
