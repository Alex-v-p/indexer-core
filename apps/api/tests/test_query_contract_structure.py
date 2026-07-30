from __future__ import annotations

import ast
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.api.presenters.query import to_query_response as presenter_to_query_response
from app.api.routes.queries import to_query_response as route_to_query_response
from app.schemas.queries import (
    AnswerPresentationResponse,
    QueryRequest,
    QueryResponse,
    RetrievalPlanResponse,
    TraceStepResponse,
)
from app.schemas.queries.answer import AnswerPresentationResponse as SplitAnswerPresentationResponse
from app.schemas.queries.request import QueryRequest as SplitQueryRequest
from app.schemas.queries.response import QueryResponse as SplitQueryResponse
from app.schemas.queries.retrieval import RetrievalPlanResponse as SplitRetrievalPlanResponse
from app.schemas.queries.trace import TraceStepResponse as SplitTraceStepResponse
from packages.indexer_application.dto import QueryRunRecord, QueryRunStatus

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_query_schema_facade_preserves_public_model_identity() -> None:
    assert QueryRequest is SplitQueryRequest
    assert AnswerPresentationResponse is SplitAnswerPresentationResponse
    assert RetrievalPlanResponse is SplitRetrievalPlanResponse
    assert TraceStepResponse is SplitTraceStepResponse
    assert QueryResponse is SplitQueryResponse


def test_query_route_uses_presenter_compatibility_facade() -> None:
    assert route_to_query_response is presenter_to_query_response

    route_file = REPOSITORY_ROOT / "apps" / "api" / "app" / "api" / "routes" / "queries.py"
    tree = ast.parse(route_file.read_text(encoding="utf-8"), filename=str(route_file))
    defined_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert defined_functions == {"create_query_run", "read_query_run"}


def test_answer_presentation_is_additive_and_legacy_answer_remains_unchanged() -> None:
    now = datetime.now(UTC)
    legacy_answer = (
        "Supported information [1].\n\n"
        "The available documents did not provide sufficient evidence for:\n"
        "- Missing information."
    )
    record = QueryRunRecord(
        id=uuid.uuid4(),
        question="What is supported?",
        answer=legacy_answer,
        status=QueryRunStatus.SUCCEEDED,
        pipeline_name="agentic",
        pipeline_version="1.0.0",
        top_k=5,
        started_at=now,
        completed_at=now,
        error_message=None,
        metadata={
            "answer_presentation": {
                "schema_version": "1.0",
                "outcome": "partial",
                "title": "Partial answer",
                "body": "Supported information [1].",
                "supported_information": ["Supported information."],
                "unresolved_information": ["Missing information."],
                "citation_count": 1,
            },
        },
    )

    response = presenter_to_query_response(record)

    assert response.answer == legacy_answer
    assert response.answer_presentation is not None
    assert response.answer_presentation.body == "Supported information [1]."
    assert response.answer_presentation.unresolved_information == ["Missing information."]


def test_historical_answer_without_presentation_preserves_exact_fallback() -> None:
    now = datetime.now(UTC)
    record = QueryRunRecord(
        id=uuid.uuid4(),
        question="Historical question",
        answer="Historical answer.\nSecond line.",
        status=QueryRunStatus.SUCCEEDED,
        pipeline_name="baseline",
        pipeline_version="1.0.0",
        top_k=5,
        started_at=now,
        completed_at=now,
        error_message=None,
        metadata={},
    )

    response = presenter_to_query_response(record)

    assert response.answer == "Historical answer.\nSecond line."
    assert response.answer_presentation is None


def test_angular_query_contracts_separate_transport_and_view_models() -> None:
    query_root = REPOSITORY_ROOT / "apps" / "web" / "src" / "app" / "features" / "queries"
    models_root = query_root / "models"
    view_models_root = query_root / "view-models"

    assert {
        "query-contract.models.ts",
        "answer.models.ts",
        "retrieval.models.ts",
        "trace.models.ts",
    }.issubset({path.name for path in models_root.glob("*.ts")})
    assert {
        "agent-trace-view.models.ts",
        "trace-evidence-view.models.ts",
        "answer.view-model.ts",
    }.issubset({path.name for path in view_models_root.glob("*.ts")})

    compatibility_facade = (models_root / "query.models.ts").read_text(encoding="utf-8")
    assert compatibility_facade.splitlines() == [
        "export * from './query-contract.models';",
        "export * from './answer.models';",
        "export * from './retrieval.models';",
        "export * from './trace.models';",
    ]

    transport_source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            models_root / "query-contract.models.ts",
            models_root / "answer.models.ts",
            models_root / "retrieval.models.ts",
            models_root / "trace.models.ts",
        )
    )
    assert "AgentTraceViewModel" not in transport_source
    assert "TraceEvidenceCardViewModel" not in transport_source
    assert "QueryAnswerViewModel" not in transport_source
