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
