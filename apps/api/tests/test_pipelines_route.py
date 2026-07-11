from fastapi.testclient import TestClient

from app.main import create_app


def test_pipeline_catalog_lists_default_pipeline_and_tools() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    body = response.json()
    assert body["default_pipeline_name"] == "baseline_rag"
    pipelines = {pipeline["name"]: pipeline for pipeline in body["pipelines"]}
    assert set(pipelines) == {
        "baseline_rag",
        "hybrid_rag",
        "hybrid_llm_rerank_rag",
        "hybrid_cross_encoder_rerank_rag",
    }
    assert pipelines["baseline_rag"]["is_default"] is True
    assert pipelines["hybrid_rag"]["is_default"] is False
    assert {tool["kind"] for tool in pipelines["baseline_rag"]["tools"]} == {"retriever", "generator"}
    assert pipelines["hybrid_rag"]["metadata"]["fusion_method"] == "weighted_reciprocal_rank_fusion"
    assert {tool["name"] for tool in pipelines["hybrid_rag"]["tools"]} >= {
        "retriever.vector",
        "retriever.keyword_bm25",
        "retriever.hybrid_rrf",
    }
    assert pipelines["hybrid_llm_rerank_rag"]["metadata"]["reranking_strategy"] == "ollama_pointwise_relevance"
    assert {tool["name"] for tool in pipelines["hybrid_llm_rerank_rag"]["tools"]} >= {
        "retriever.hybrid_rrf",
        "reranker.ollama",
    }
    assert {tool["kind"] for tool in pipelines["hybrid_llm_rerank_rag"]["tools"]} >= {
        "retriever",
        "reranker",
        "generator",
    }


def test_pipeline_catalog_exposes_cross_encoder_as_a_separate_reranking_option() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    pipelines = {pipeline["name"]: pipeline for pipeline in response.json()["pipelines"]}
    cross_encoder = pipelines["hybrid_cross_encoder_rerank_rag"]
    assert cross_encoder["metadata"]["reranking_strategy"] == "cross_encoder_pairwise_relevance"
    assert {tool["name"] for tool in cross_encoder["tools"]} >= {
        "retriever.hybrid_rrf",
        "reranker.cross_encoder",
    }
    ollama = pipelines["hybrid_llm_rerank_rag"]
    assert ollama["metadata"]["reranking_strategy"] == "ollama_pointwise_relevance"
    assert ollama["metadata"]["incomplete_response_policy"] == "retry_then_preserve_original_rank"
