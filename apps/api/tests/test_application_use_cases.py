from __future__ import annotations

import ast
import uuid
from datetime import UTC, datetime
from pathlib import Path

from packages.indexer_application.commands import ExecuteQueryCommand, ExecuteQueryHandler
from packages.indexer_application.dto import (
    QueryExecutionResult,
    QueryRunRecord,
    QueryRunStatus,
)
from packages.indexer_application.queries import GetQueryRunHandler, GetQueryRunQuery
from packages.rag_core.agents import QueryState


class StubPipeline:
    name = "stub"
    version = "1.0.0"

    async def run(self, state: QueryState) -> QueryState:
        state.answer = "Grounded answer."
        state.metadata["query_classification"] = {
            "query_type": "factual_lookup",
            "confidence": 0.9,
            "needs_metadata_filters": False,
            "rationale": "Direct lookup.",
            "classifier_name": "stub",
        }
        state.metadata["internal_only"] = {"may_change": True}
        return state


class StubPipelineRegistry:
    def __init__(self, pipeline: StubPipeline) -> None:
        self.pipeline = pipeline
        self.requested_name = None

    def build(self, pipeline_name: str | None = None) -> StubPipeline:
        self.requested_name = pipeline_name
        return self.pipeline


class FakeQueryRunRepository:
    def __init__(self) -> None:
        self.query_run_id = uuid.uuid4()
        self.record: QueryRunRecord | None = None

    async def create_running(self, **kwargs) -> uuid.UUID:
        self.created = kwargs
        return self.query_run_id

    async def mark_succeeded(self, *, query_run_id: uuid.UUID, state: QueryState) -> None:
        now = datetime.now(UTC)
        self.record = QueryRunRecord(
            id=query_run_id,
            question=state.question,
            answer=state.answer,
            status=QueryRunStatus.SUCCEEDED,
            pipeline_name=state.pipeline_name,
            pipeline_version=state.pipeline_version,
            top_k=state.top_k,
            started_at=now,
            completed_at=now,
            error_message=None,
            metadata=dict(state.metadata),
        )

    async def mark_failed(self, **kwargs) -> None:
        raise AssertionError("The successful path must not be marked failed.")

    async def get(self, query_run_id: uuid.UUID) -> QueryRunRecord | None:
        assert query_run_id == self.query_run_id
        return self.record


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.query_runs = FakeQueryRunRepository()
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


class FailingPipeline:
    name = "failing"
    version = "1.0.0"

    async def run(self, state: QueryState) -> QueryState:
        raise RuntimeError("provider unavailable")


class FailureQueryRunRepository(FakeQueryRunRepository):
    async def mark_failed(
        self,
        *,
        query_run_id: uuid.UUID,
        error_message: str,
        trace,
    ) -> None:
        del trace
        now = datetime.now(UTC)
        self.record = QueryRunRecord(
            id=query_run_id,
            question=self.created["question"],
            answer=None,
            status=QueryRunStatus.FAILED,
            pipeline_name=self.created["pipeline_name"],
            pipeline_version=self.created["pipeline_version"],
            top_k=self.created["top_k"],
            started_at=now,
            completed_at=now,
            error_message=error_message,
            metadata={},
        )


class FailureUnitOfWork(FakeUnitOfWork):
    def __init__(self) -> None:
        super().__init__()
        self.query_runs = FailureQueryRunRepository()


async def test_execute_query_handler_owns_selection_execution_and_typed_result() -> None:
    uow = FakeUnitOfWork()
    registry = StubPipelineRegistry(StubPipeline())
    handler = ExecuteQueryHandler(uow=uow, pipeline_registry=registry)

    result = await handler(
        ExecuteQueryCommand(
            question="What is indexed?",
            top_k=4,
            pipeline_name="stub",
        ),
    )

    assert isinstance(result, QueryExecutionResult)
    assert result.query_run.answer == "Grounded answer."
    assert result.metadata.classification is not None
    assert result.metadata.classification["query_type"] == "factual_lookup"
    assert not hasattr(result.metadata, "internal_only")
    assert registry.requested_name == "stub"
    assert uow.query_runs.created["requested_pipeline_name"] == "stub"
    assert uow.commit_calls == 1


async def test_execute_query_handler_preserves_failed_run_behavior() -> None:
    uow = FailureUnitOfWork()
    handler = ExecuteQueryHandler(
        uow=uow,
        pipeline_registry=StubPipelineRegistry(FailingPipeline()),
    )

    result = await handler(
        ExecuteQueryCommand(
            question="Why did retrieval fail?",
            top_k=5,
            pipeline_name="failing",
        ),
    )

    assert result.query_run.status is QueryRunStatus.FAILED
    assert result.query_run.error_message == "provider unavailable"
    assert result.query_run.answer is None
    assert uow.commit_calls == 1


async def test_get_query_run_handler_returns_same_typed_contract() -> None:
    uow = FakeUnitOfWork()
    now = datetime.now(UTC)
    uow.query_runs.record = QueryRunRecord(
        id=uow.query_runs.query_run_id,
        question="Historical question",
        answer="Historical answer",
        status=QueryRunStatus.SUCCEEDED,
        pipeline_name="stub",
        pipeline_version="1.0.0",
        top_k=3,
        started_at=now,
        completed_at=now,
        error_message=None,
        metadata={"answer_presentation": {"schema_version": "1.0"}},
    )
    handler = GetQueryRunHandler(uow=uow)

    result = await handler(GetQueryRunQuery(query_run_id=uow.query_runs.query_run_id))

    assert result is not None
    assert result.query_run.answer == "Historical answer"
    assert result.metadata.answer_presentation == {"schema_version": "1.0"}


def test_application_handlers_are_framework_and_adapter_independent() -> None:
    repository_root = Path(__file__).resolve().parents[3]
    application_entrypoints = (
        repository_root / "packages" / "indexer_application" / "commands",
        repository_root / "packages" / "indexer_application" / "queries",
    )
    forbidden_prefixes = (
        "fastapi",
        "sqlalchemy",
        "minio",
        "qdrant_client",
        "packages.indexer_infrastructure",
    )
    violations: list[str] = []

    for entrypoint_root in application_entrypoints:
        for source_file in entrypoint_root.rglob("*.py"):
            tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
            for node in ast.walk(tree):
                module_names: list[str] = []
                if isinstance(node, ast.Import):
                    module_names.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module_names.append(node.module)
                for module_name in module_names:
                    if any(
                        module_name == prefix or module_name.startswith(f"{prefix}.")
                        for prefix in forbidden_prefixes
                    ):
                        relative = source_file.relative_to(repository_root)
                        violations.append(f"{relative}: imports {module_name}")

    assert not violations, "Application handler dependency violations:\n" + "\n".join(violations)
