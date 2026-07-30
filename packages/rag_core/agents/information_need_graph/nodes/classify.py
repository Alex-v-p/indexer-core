from __future__ import annotations

from packages.rag_core.agents.information_need_graph.models import ClassificationSource
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.query_understanding.classification import QueryClassifier


class ClassifyInformationNeedNode:
    """Classify the active item, reusing an equivalent top-level result once."""

    name = "classify_information_need"
    step_type = "information_need_classification"

    def __init__(self, classifier: QueryClassifier) -> None:
        self._classifier = classifier

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None:
            raise RuntimeError("Information-need classification requires an active work item.")
        first_classification = execution.classification is None and not execution.classification_history
        source: ClassificationSource
        if first_classification and _can_reuse_top_level_classification(state):
            classification = state.query_classification
            if classification is None:
                raise RuntimeError("Top-level classification reuse requires a classification.")
            source = "top_level_reuse"
        else:
            classification = await self._classifier.classify(execution.information_need.retrieval_query)
            source = "model"
        if not first_classification:
            execution.reclassifications_used += 1
        execution.classification = classification
        execution.classification_history.append(classification)
        execution.classification_source_history.append(source)
        execution.next_route = None
        state.metadata["active_information_need_classification"] = {
            "information_need_id": execution.information_need.need_id,
            "classification": classification.to_metadata(),
            "classification_source": source,
            "classification_count": len(execution.classification_history),
        }
        return state


def _can_reuse_top_level_classification(state: QueryState) -> bool:
    decomposition = state.information_need_decomposition
    execution = state.active_information_need_execution
    return (
        execution is not None
        and decomposition is not None
        and len(decomposition.information_needs) == 1
        and state.query_classification is not None
        and _normalize_query(execution.information_need.retrieval_query)
        == _normalize_query(state.question)
    )


def _normalize_query(value: str) -> str:
    return " ".join(value.split()).casefold()
