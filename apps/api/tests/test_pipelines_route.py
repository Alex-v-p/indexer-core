from fastapi.testclient import TestClient

from app.main import create_app


def test_pipeline_catalog_lists_default_pipeline_and_tools() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    body = response.json()
    assert body["default_pipeline_name"] == "agentic_rag"
    pipelines = {pipeline["name"]: pipeline for pipeline in body["pipelines"]}
    assert set(pipelines) == {
        "baseline_rag",
        "hybrid_rag",
        "hybrid_llm_rerank_rag",
        "hybrid_cross_encoder_rerank_rag",
        "contextual_rag",
        "multi_query_rag",
        "agentic_rag",
    }
    assert pipelines["baseline_rag"]["is_default"] is False
    assert pipelines["hybrid_rag"]["is_default"] is False
    assert pipelines["agentic_rag"]["is_default"] is True
    assert {tool["kind"] for tool in pipelines["baseline_rag"]["tools"]} == {"classifier", "retriever", "generator"}
    classifier = next(
        tool for tool in pipelines["baseline_rag"]["tools"] if tool["name"] == "classifier.query"
    )
    assert classifier["kind"] == "classifier"
    assert classifier["metadata"]["query_types"] == [
        "factual_lookup",
        "broad_explanation",
        "comparison",
        "version_specific",
    ]
    assert pipelines["baseline_rag"]["metadata"]["stages"][0] == "classify_query"
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
        "classifier",
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
    cross_encoder_tool = next(
        tool for tool in cross_encoder["tools"] if tool["name"] == "reranker.cross_encoder"
    )
    assert cross_encoder_tool["metadata"]["local_files_only"] is True
    assert cross_encoder_tool["metadata"]["model_path"] == "/root/.cache/huggingface/indexer/cross-encoder"
    assert cross_encoder_tool["metadata"]["revision"] == "c5ee24cb16019beea0893ab7796b1df96625c6b8"
    ollama = pipelines["hybrid_llm_rerank_rag"]
    assert ollama["metadata"]["reranking_strategy"] == "ollama_pointwise_relevance"
    assert ollama["metadata"]["incomplete_response_policy"] == "retry_then_preserve_original_rank"


def test_pipeline_registries_are_application_scoped() -> None:
    app = create_app()
    tool_registry = app.state.query_tool_registry
    pipeline_registry = app.state.query_pipeline_registry
    cross_encoder = tool_registry.resolve("reranker.cross_encoder")

    client = TestClient(app)
    assert client.get("/api/v1/pipelines").status_code == 200
    assert client.get("/api/v1/pipelines").status_code == 200

    assert app.state.query_tool_registry is tool_registry
    assert app.state.query_pipeline_registry is pipeline_registry
    assert app.state.query_tool_registry.resolve("reranker.cross_encoder") is cross_encoder


def test_pipeline_catalog_exposes_contextual_retrieval_with_original_evidence() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    pipelines = {pipeline["name"]: pipeline for pipeline in response.json()["pipelines"]}
    contextual = pipelines["contextual_rag"]
    assert contextual["metadata"]["contextualization_strategy"] == "adjacent_chunk_window"
    assert contextual["metadata"]["evidence_text"] == "original_chunk"
    assert {tool["name"] for tool in contextual["tools"]} >= {
        "retriever.contextual_vector",
        "retriever.contextual_keyword_bm25",
        "retriever.contextual_hybrid_rrf",
    }
    vector_tool = next(tool for tool in contextual["tools"] if tool["name"] == "retriever.contextual_vector")
    assert vector_tool["metadata"]["collection"] == "indexer_chunks"
    assert vector_tool["metadata"]["vector_name"] == "contextual"


def test_pipeline_catalog_exposes_multi_query_expansion_and_fusion() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    pipelines = {pipeline["name"]: pipeline for pipeline in response.json()["pipelines"]}
    multi_query = pipelines["multi_query_rag"]
    assert multi_query["metadata"]["retrieval_strategy"] == "multi_query_hybrid"
    assert multi_query["metadata"]["base_retrieval_strategy"] == "hybrid"
    assert multi_query["metadata"]["internal_retrieval_stages"] == [
        "generate_query_variants",
        "retrieve_each",
        "fuse",
    ]
    assert {tool["name"] for tool in multi_query["tools"]} >= {
        "retriever.hybrid_rrf",
        "query_generator.llm_variants",
        "retriever.multi_query_rrf",
    }
    generator = next(
        tool for tool in multi_query["tools"] if tool["name"] == "query_generator.llm_variants"
    )
    assert generator["kind"] == "query_generator"
    assert generator["metadata"]["variant_count"] == 3


def test_pipeline_catalog_exposes_agentic_retrieval_planning() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    pipelines = {pipeline["name"]: pipeline for pipeline in response.json()["pipelines"]}
    agentic = pipelines["agentic_rag"]
    assert agentic["metadata"]["selection_mode"] == "classification_driven"
    assert agentic["metadata"]["selectable_strategies"] == [
        "baseline",
        "hybrid",
        "contextual",
        "multi_query",
        "rerank",
    ]
    assert agentic["metadata"]["stages"] == [
        "classify_query",
        "plan_retrieval",
        "execute_retrieval_plan",
        "grade_evidence",
        "generate_answer",
    ]
    tools = {tool["name"]: tool for tool in agentic["tools"]}
    assert tools["planner.retrieval"]["kind"] == "planner"
    assert tools["planner.retrieval"]["metadata"]["rerank_pipeline"] == "hybrid_cross_encoder_rerank_rag"
