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
        "hierarchical_rag",
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


def test_pipeline_catalog_exposes_hierarchical_document_section_chunk_routing() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    pipelines = {pipeline["name"]: pipeline for pipeline in response.json()["pipelines"]}
    hierarchical = pipelines["hierarchical_rag"]
    assert hierarchical["metadata"]["retrieval_strategy"] == "hierarchical_document_section_chunk"
    assert hierarchical["metadata"]["routing_levels"] == [
        "document_summary",
        "section_summary",
        "source_chunk",
    ]
    assert hierarchical["metadata"]["answer_evidence_level"] == "source_chunk"
    tool = next(tool for tool in hierarchical["tools"] if tool["name"] == "retriever.hierarchical")
    assert tool["metadata"]["hierarchy_vector_name"] == "hierarchy"
    assert tool["metadata"]["chunk_vector_name"] == "contextual"
    assert tool["metadata"]["document_candidates"] == 8
    assert tool["metadata"]["section_candidates"] == 24


def test_pipeline_catalog_exposes_agentic_retrieval_planning() -> None:
    client = TestClient(create_app())

    response = client.get("/api/v1/pipelines")

    assert response.status_code == 200
    pipelines = {pipeline["name"]: pipeline for pipeline in response.json()["pipelines"]}
    agentic = pipelines["agentic_rag"]
    assert agentic["metadata"]["graph_mode"] == "hierarchical_top_level_with_cyclic_information_need_subgraph"
    assert agentic["metadata"]["selection_mode"] == "per_information_need_classification_and_planning"
    assert agentic["metadata"]["selectable_strategies"] == [
        "baseline",
        "hybrid",
        "contextual",
        "hierarchical",
        "multi_query",
        "rerank",
    ]
    assert agentic["metadata"]["top_level_stages"] == [
        "classify_query",
        "decompose_information_needs",
        "initialize_information_need_work",
        "resolve_information_needs",
        "aggregate_information_needs",
        "prepare_evidence_context",
        "generate_answer",
    ]
    assert agentic["metadata"]["information_need_subgraph_stages"] == [
        "select_information_need",
        "classify_information_need",
        "plan_information_need",
        "execute_information_need_plan",
        "validate_information_need_constraints",
        "grade_information_need",
        "decide_information_need",
        "complete_information_need",
    ]
    tools = {tool["name"]: tool for tool in agentic["tools"]}
    assert tools["query.information_need_decomposer"]["kind"] == "decomposer"
    assert tools["query.information_need_decomposer"]["metadata"]["decomposer"] == "llm_information_need_decomposer"
    assert tools["planner.retrieval"]["kind"] == "planner"
    assert tools["planner.retrieval"]["metadata"]["rerank_pipeline"] == "hybrid_cross_encoder_rerank_rag"
    assert tools["planner.retrieval"]["metadata"]["hierarchical_pipeline"] == "hierarchical_rag"
    assert tools["planner.retrieval"]["metadata"]["inputs"] == [
        "information_need",
        "information_need_classification",
        "previous_grade",
        "attempt_history",
    ]
    assert tools["planner.retrieval"]["metadata"]["planning_scope"] == "per_information_need"
    assert "planner.claim_retry" not in tools
    assert tools["grader.evidence_relevance"]["metadata"]["information_need_support_threshold"] == 0.75
    assert tools["policy.retrieval_retry"]["kind"] == "retry_policy"
    assert tools["policy.retrieval_retry"]["metadata"]["max_retries"] == 2
    assert tools["policy.retrieval_retry"]["metadata"]["max_total_attempts"] == 20
    assert tools["policy.retrieval_retry"]["metadata"]["max_reclassifications"] == 1
    assert tools["policy.retrieval_retry"]["metadata"]["max_accumulated_evidence"] == 40
    assert tools["policy.retrieval_retry"]["metadata"]["scope"] == "per_information_need"
    assert agentic["metadata"]["retry_mode"] == "per_information_need_bounded_cycles"
    assert agentic["metadata"]["retry_budget_mode"] == "per_information_need_and_query_global_limits"
    assert agentic["metadata"]["constraint_enforcement_mode"] == "strict_subgraph_and_pre_generation_validation"
    assert agentic["metadata"]["evidence_metadata_context_mode"] == "constraint_relevant_compact_source_metadata"
    assert agentic["metadata"]["answer_evidence_mode"] == "union_of_per_information_need_grader_approved_evidence"
    assert agentic["metadata"]["partial_answer_mode"] == "explicit_unresolved_information_need_disclosure"
