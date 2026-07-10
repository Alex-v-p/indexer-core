from fastapi.testclient import TestClient

from app.main import create_app


def test_documents_route_is_registered() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/documents" in response.json()["paths"]
