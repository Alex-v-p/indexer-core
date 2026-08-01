from __future__ import annotations

from packages.rag_core.agents.information_need_graph.evidence import evidence_for_information_need
from packages.rag_core.agents.query_graph.state import QueryState
from packages.rag_core.retrieval.document_selection import PrimaryDocumentDetector


class DetectPrimaryDocumentNode:
    """Update the query-wide soft document preference after a completed work item."""

    name = "detect_primary_document"
    step_type = "document_preference"

    def __init__(self, detector: PrimaryDocumentDetector) -> None:
        self._detector = detector

    async def __call__(self, state: QueryState) -> QueryState:
        execution = state.active_information_need_execution
        if execution is None or execution.last_grading is None:
            raise RuntimeError("Primary-document detection requires a graded active information need.")

        previous = state.primary_document_preference
        if previous is not None and execution.information_need.document_scope.strict:
            document_id = previous.document.document_id
            if document_id is None or not execution.information_need.document_scope.allows(document_id):
                previous = None
        evidence = evidence_for_information_need(state, execution.information_need.need_id)
        detected = self._detector.detect(
            question=state.question,
            information_need=execution.information_need,
            evidence=evidence,
            grading=execution.last_grading,
            existing_preference=previous,
        )
        state.primary_document_preference = detected
        state.metadata["primary_document_preference"] = (
            detected.to_metadata() if detected is not None else None
        )
        state.metadata["primary_document_detection"] = {
            "information_need_id": execution.information_need.need_id,
            "detector_name": getattr(self._detector, "name", type(self._detector).__name__),
            "candidate_evidence_count": len(evidence),
            "previous_document_key": previous.document.key if previous is not None else None,
            "selected_document_key": detected.document.key if detected is not None else None,
            "preference_changed": (
                (previous.document.key if previous is not None else None)
                != (detected.document.key if detected is not None else None)
            ),
        }
        return state
