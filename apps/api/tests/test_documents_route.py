from fastapi.testclient import TestClient

from app.composition import build_chunk_contextualizer, build_document_ingestion_config
from app.core.config import Settings
from app.main import create_app


def test_documents_route_is_registered() -> None:
    client = TestClient(create_app())

    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/documents" in response.json()["paths"]


def test_default_document_ingestion_composition_enables_contextualization(monkeypatch) -> None:
    monkeypatch.delenv("CONTEXTUALIZATION_ENABLED", raising=False)
    monkeypatch.delenv("CONTEXTUALIZATION_FAIL_OPEN", raising=False)
    settings = Settings(_env_file=None)

    config = build_document_ingestion_config(settings)

    assert config.contextualization_enabled is True
    assert config.contextualization_fail_open is False
    assert build_chunk_contextualizer(settings) is not None
