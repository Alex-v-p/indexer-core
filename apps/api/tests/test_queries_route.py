from fastapi.testclient import TestClient

from app.api.presenters.query import (
    _to_answer_presentation_response,
    _to_primary_document_preference_response,
)
from app.dependencies.database import get_session
from app.main import create_app
from app.schemas.queries import (
    EvidenceGradingResponse,
    InformationNeedDecompositionResponse,
    QueryClassificationResponse,
)


def test_queries_route_is_registered() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    body = response.json()
    assert "/api/v1/queries" in body["paths"]
    query_schema = body["components"]["schemas"]["QueryRequest"]
    assert "pipeline_name" in query_schema["properties"]
    response_schema = body["components"]["schemas"]["QueryResponse"]
    assert "classification" in response_schema["properties"]
    assert "information_need_decomposition" in response_schema["properties"]
    assert "retrieval_plan" in response_schema["properties"]
    assert "evidence_grading" in response_schema["properties"]
    assert "retrieval_retry" in response_schema["properties"]
    assert "primary_document_preference" in response_schema["properties"]
    assert "information_need_resolution" in response_schema["properties"]
    assert "answer_presentation" in response_schema["properties"]
    assert "constraint_validation" in response_schema["properties"]
    assert "evidence_context" in response_schema["properties"]
    decomposition_schema = body["components"]["schemas"]["InformationNeedDecompositionResponse"]
    assert "information_needs" in decomposition_schema["properties"]
    assert "structured_output" in decomposition_schema["properties"]
    classification_schema = body["components"]["schemas"]["QueryClassificationResponse"]
    assert "structured_output" in classification_schema["properties"]
    retrieval_plan_schema = body["components"]["schemas"]["RetrievalPlanResponse"]
    assert "target_information_need_ids" in retrieval_plan_schema["properties"]
    assert "preferred_document" in retrieval_plan_schema["properties"]
    assert "information_needs" not in retrieval_plan_schema["properties"]
    grading_schema = body["components"]["schemas"]["EvidenceGradingResponse"]
    assert "information_need_grades" in grading_schema["properties"]
    assert "unresolved_information" in grading_schema["properties"]
    assert "answerable" in grading_schema["properties"]
    assert "partial_answer_available" in grading_schema["properties"]
    assert "relevant_evidence_ranks" in grading_schema["properties"]
    assert "supported_information" in grading_schema["properties"]
    assert "structured_output" in grading_schema["properties"]
    retry_schema = body["components"]["schemas"]["RetrievalRetryResponse"]
    assert "attempts" in retry_schema["properties"]
    assert "claim_plan_count" in retry_schema["properties"]
    assert "claim_lookup_count" in retry_schema["properties"]
    assert "stop_reason" in retry_schema["properties"]
    attempt_schema = body["components"]["schemas"]["RetrievalAttemptResponse"]
    assert "claim_retrieval_plan" in attempt_schema["properties"]
    assert "claim_lookups" in attempt_schema["properties"]
    assert "accumulated_evidence_count" in attempt_schema["properties"]
    claim_plan_schema = body["components"]["schemas"]["ClaimRetrievalPlanResponse"]
    assert "tasks" in claim_plan_schema["properties"]
    assert "deferred_information_need_ids" in claim_plan_schema["properties"]
    resolution_schema = body["components"]["schemas"]["InformationNeedResolutionResponse"]
    assert "executions" in resolution_schema["properties"]
    assert "total_retrieval_attempts" in resolution_schema["properties"]
    assert "supported_information_need_ids" in resolution_schema["properties"]
    assert "unresolved_information_need_ids" in resolution_schema["properties"]
    execution_schema = body["components"]["schemas"]["InformationNeedExecutionResponse"]
    assert "classification_history" in execution_schema["properties"]
    assert "classification_source_history" in execution_schema["properties"]
    assert "plan_history" in execution_schema["properties"]
    assert "attempts" in execution_schema["properties"]
    assert "constraint_validation_history" in execution_schema["properties"]
    information_need_attempt_schema = body["components"]["schemas"]["InformationNeedAttemptResponse"]
    assert "constraint_validation" in information_need_attempt_schema["properties"]
    assert "evidence" in information_need_attempt_schema["properties"]
    assert "retrieval_metadata" in information_need_attempt_schema["properties"]
    assert "reranking_metadata" in information_need_attempt_schema["properties"]
    assert "document_balancing" in information_need_attempt_schema["properties"]
    attempt_evidence_schema = body["components"]["schemas"]["InformationNeedAttemptEvidenceResponse"]
    assert "retained_after_need_grading" in attempt_evidence_schema["properties"]
    retrieval_metadata_schema = body["components"]["schemas"]["RetrievalExecutionMetadataResponse"]
    assert "query_variants" in retrieval_metadata_schema["properties"]
    assert "hierarchical_selected_section_ids" in retrieval_metadata_schema["properties"]
    reranking_metadata_schema = body["components"]["schemas"]["RerankingMetadataResponse"]
    assert "candidate_count_before" in reranking_metadata_schema["properties"]
    balancing_schema = body["components"]["schemas"]["AttemptDocumentBalancingResponse"]
    assert "selected_chunks_per_document" in balancing_schema["properties"]
    assert "primary_document_quota" in balancing_schema["properties"]
    information_need_plan_schema = body["components"]["schemas"]["InformationNeedRetrievalPlanResponse"]
    assert "preferred_document" in information_need_plan_schema["properties"]
    preference_schema = body["components"]["schemas"]["DocumentPreferenceResponse"]
    assert "semantics" in preference_schema["properties"]
    assert "final_grade" in execution_schema["properties"]
    presentation_schema = body["components"]["schemas"]["AnswerPresentationResponse"]
    assert presentation_schema["properties"]["schema_version"]["const"] == "1.0"
    assert presentation_schema["properties"]["outcome"]["enum"] == [
        "complete",
        "partial",
        "blocked_constraint_no_match",
        "blocked_insufficient_evidence",
        "blocked_no_evidence",
    ]


def test_answer_presentation_metadata_maps_additively_and_historical_absence_is_none() -> None:
    assert _to_answer_presentation_response({}) is None

    presentation = _to_answer_presentation_response(
        {
            "answer_presentation": {
                "schema_version": "1.0",
                "outcome": "partial",
                "title": "Partial answer",
                "body": "The supported fact [1].",
                "supported_information": ["Supported fact."],
                "unresolved_information": ["Missing fact."],
                "citation_count": 1,
            },
        },
    )

    assert presentation is not None
    assert presentation.outcome == "partial"
    assert presentation.body == "The supported fact [1]."
    assert presentation.unresolved_information == ["Missing fact."]


def test_historical_query_control_metadata_defaults_structured_diagnostics_to_none() -> None:
    classification = QueryClassificationResponse.model_validate(
        {
            "query_type": "factual_lookup",
            "confidence": 0.9,
            "needs_metadata_filters": False,
            "rationale": "Historical classification.",
            "classifier_name": "legacy",
        },
    )
    decomposition = InformationNeedDecompositionResponse.model_validate(
        {
            "information_needs": [],
            "information_need_count": 0,
            "rationale": "Historical decomposition.",
            "decomposer_name": "legacy",
        },
    )
    grading = EvidenceGradingResponse.model_validate(
        {
            "status": "missing",
            "coverage_score": 0.0,
            "sufficient": False,
            "missing_evidence": True,
            "weak_evidence": False,
            "relevant_count": 0,
            "total_count": 0,
            "rationale": "Historical grading.",
            "grader_name": "legacy",
        },
    )

    assert classification.structured_output is None
    assert decomposition.structured_output is None
    assert grading.structured_output is None


def test_query_request_rejects_unregistered_pipeline_before_database_use() -> None:
    app = create_app()

    async def session_override():
        yield object()

    app.dependency_overrides[get_session] = session_override
    client = TestClient(app)

    response = client.post(
        "/api/v1/queries",
        json={"question": "What is indexed?", "top_k": 5, "pipeline_name": "missing_pipeline"},
    )

    assert response.status_code == 422
    assert "Unknown pipeline 'missing_pipeline'" in response.json()["detail"]


def test_primary_document_preference_metadata_is_exposed() -> None:
    response = _to_primary_document_preference_response(
        {
            "primary_document_preference": {
                "document": {
                    "key": "document:11111111-1111-1111-1111-111111111111",
                    "display_name": "FunctionalSpecDAF_AVP.pdf",
                    "document_id": "11111111-1111-1111-1111-111111111111",
                    "document_version_ids": [],
                    "normalized_names": ["functionalspecdaf avp"],
                },
                "score": 0.91,
                "confidence": 0.86,
                "margin": 0.22,
                "supporting_information_need_ids": ["need_1"],
                "supporting_evidence_ranks": [1, 2],
                "rationale": "This document had the strongest directly graded evidence.",
                "detector_name": "rule_based_soft_primary_document_detector",
                "semantics": "soft_preference_not_filter",
            },
        },
    )

    assert response is not None
    assert response.document.display_name == "FunctionalSpecDAF_AVP.pdf"
    assert response.semantics == "soft_preference_not_filter"


def test_query_route_presents_typed_application_result_without_metadata_key_discovery() -> None:
    import uuid
    from datetime import UTC, datetime

    from app.dependencies.application import get_execute_query_handler
    from packages.indexer_application.dto import QueryExecutionResult, QueryRunRecord, QueryRunStatus

    now = datetime.now(UTC)
    record = QueryRunRecord(
        id=uuid.uuid4(),
        question="What is indexed?",
        answer="A grounded answer.",
        status=QueryRunStatus.SUCCEEDED,
        pipeline_name="stub",
        pipeline_version="1.0.0",
        top_k=5,
        started_at=now,
        completed_at=now,
        error_message=None,
        metadata={
            "query_classification": {
                "query_type": "factual_lookup",
                "confidence": 0.9,
                "needs_metadata_filters": False,
                "rationale": "Direct lookup.",
                "classifier_name": "stub",
            },
        },
    )

    class FakeExecuteQueryHandler:
        async def __call__(self, command):
            assert command.question == "What is indexed?"
            assert command.top_k == 5
            assert command.pipeline_name == "stub"
            return QueryExecutionResult.from_record(record)

    app = create_app()
    app.dependency_overrides[get_execute_query_handler] = lambda: FakeExecuteQueryHandler()
    client = TestClient(app)

    response = client.post(
        "/api/v1/queries",
        json={"question": "What is indexed?", "pipeline_name": "stub"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["answer"] == "A grounded answer."
    assert body["classification"]["query_type"] == "factual_lookup"
    assert body["pipeline_name"] == "stub"
