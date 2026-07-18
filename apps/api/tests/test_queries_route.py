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
    decomposition_schema = body["components"]["schemas"]["InformationNeedDecompositionResponse"]
    assert "information_needs" in decomposition_schema["properties"]
    retrieval_plan_schema = body["components"]["schemas"]["RetrievalPlanResponse"]
    assert "target_information_need_ids" in retrieval_plan_schema["properties"]
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
