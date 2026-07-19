from fastapi.testclient import TestClient

from app.dependencies.database import get_session
from app.main import create_app


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
    assert "constraint_validation" in response_schema["properties"]
    assert "evidence_context" in response_schema["properties"]
    decomposition_schema = body["components"]["schemas"]["InformationNeedDecompositionResponse"]
    assert "information_needs" in decomposition_schema["properties"]
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
    assert "plan_history" in execution_schema["properties"]
    assert "attempts" in execution_schema["properties"]
    assert "constraint_validation_history" in execution_schema["properties"]
    information_need_attempt_schema = body["components"]["schemas"]["InformationNeedAttemptResponse"]
    assert "constraint_validation" in information_need_attempt_schema["properties"]
    information_need_plan_schema = body["components"]["schemas"]["InformationNeedRetrievalPlanResponse"]
    assert "preferred_document" in information_need_plan_schema["properties"]
    preference_schema = body["components"]["schemas"]["DocumentPreferenceResponse"]
    assert "semantics" in preference_schema["properties"]
    assert "final_grade" in execution_schema["properties"]


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
    from app.api.routes.queries import _to_primary_document_preference_response

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
