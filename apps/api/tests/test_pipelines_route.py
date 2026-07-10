from fastapi.testclient import TestClient

from app.main import create_app


def test_pipeline_catalog_lists_default_pipeline_and_tools() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    body = response.json()
    assert body["default_pipeline_name"] == "baseline_rag"
    assert body["pipelines"][0]["name"] == "baseline_rag"
    assert body["pipelines"][0]["is_default"] is True
    assert {tool["kind"] for tool in body["pipelines"][0]["tools"]} == {"retriever", "generator"}
