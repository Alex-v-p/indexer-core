from fastapi.testclient import TestClient

from app.main import create_app


def test_pipeline_catalog_lists_default_pipeline_and_tools() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    body = response.json()
    assert body["default_pipeline_name"] == "baseline_rag"
    pipelines = {pipeline["name"]: pipeline for pipeline in body["pipelines"]}
    assert set(pipelines) == {"baseline_rag", "hybrid_rag", "hybrid_rerank_rag"}
    assert pipelines["baseline_rag"]["is_default"] is True
    assert pipelines["hybrid_rag"]["is_default"] is False
    assert {tool["kind"] for tool in pipelines["baseline_rag"]["tools"]} == {"retriever", "generator"}
    assert pipelines["hybrid_rag"]["metadata"]["fusion_method"] == "weighted_reciprocal_rank_fusion"
    assert {tool["name"] for tool in pipelines["hybrid_rag"]["tools"]} >= {
        "retriever.vector",
        "retriever.keyword_bm25",
        "retriever.hybrid_rrf",
    }
    assert pipelines["hybrid_rerank_rag"]["metadata"]["reranking_strategy"] == "ollama_pointwise_relevance"
    assert {tool["name"] for tool in pipelines["hybrid_rerank_rag"]["tools"]} >= {
        "retriever.hybrid_rrf",
        "reranker.ollama",
    }
    assert {tool["kind"] for tool in pipelines["hybrid_rerank_rag"]["tools"]} >= {
        "retriever",
        "reranker",
        "generator",
    }
