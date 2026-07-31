from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from packages.indexer_application.commands import SubmitQueryCommand, SubmitQueryHandler
from packages.indexer_application.dto import (
    BackgroundJobRecord,
    BackgroundJobStatus,
    BackgroundJobSubmission,
    BackgroundJobType,
    QueryRunRecord,
    QueryRunStatus,
)
from packages.indexer_application.services.background_jobs import (
    ProcessQueryJobHandler,
    QueryJobProgressTracker,
)
from packages.rag_core.agents import QueryState
from packages.rag_core.agents.runtime import GraphProgressEvent, GraphRunner, NodeSpec


NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


class StubPipeline:
    name = "stub"
    version = "1.0.0"

    def __init__(self) -> None:
        self.run_calls = 0

    async def run(self, state: QueryState) -> QueryState:
        self.run_calls += 1
        if state.progress_observer is not None:
            await state.progress_observer(
                GraphProgressEvent(
                    node_name="classify_query",
                    step_type="classification",
                    status="started",
                    graph_name=self.name,
                    graph_version=self.version,
                    graph_depth=0,
                    step_order=1,
                )
            )
            await state.progress_observer(
                GraphProgressEvent(
                    node_name="generate_answer",
                    step_type="generation",
                    status="started",
                    graph_name=self.name,
                    graph_version=self.version,
                    graph_depth=0,
                    step_order=2,
                )
            )
        state.answer = "A queued grounded answer."
        return state


class StubPipelineRegistry:
    def __init__(self) -> None:
        self.pipeline = StubPipeline()
        self.requested: list[str | None] = []

    def build(self, pipeline_name: str | None = None) -> StubPipeline:
        self.requested.append(pipeline_name)
        return self.pipeline


class FakeBackgroundJobs:
    def __init__(self) -> None:
        self.submission: BackgroundJobSubmission | None = None

    async def enqueue(self, submission: BackgroundJobSubmission) -> BackgroundJobRecord:
        self.submission = submission
        scheduled_at = submission.scheduled_at or NOW
        return BackgroundJobRecord(
            id=uuid.uuid4(),
            job_type=submission.job_type,
            status=BackgroundJobStatus.QUEUED,
            priority=submission.priority,
            payload=dict(submission.payload),
            result={},
            progress=0.0,
            current_stage="queued",
            attempts=0,
            max_attempts=submission.max_attempts,
            dedupe_key=submission.dedupe_key,
            scheduled_at=scheduled_at,
            locked_at=None,
            locked_by=None,
            heartbeat_at=None,
            started_at=None,
            completed_at=None,
            error_message=None,
            created_at=NOW,
            updated_at=NOW,
        )


class FakeQueryRuns:
    def __init__(self) -> None:
        self.query_run_id = uuid.uuid4()
        self.record: QueryRunRecord | None = None
        self.created: dict[str, object] = {}
        self.running: dict[str, object] = {}
        self.succeeded_state: QueryState | None = None

    async def create_pending(self, **kwargs) -> uuid.UUID:
        self.created = kwargs
        self.record = QueryRunRecord(
            id=self.query_run_id,
            question=str(kwargs["question"]),
            answer=None,
            status=QueryRunStatus.PENDING,
            pipeline_name=str(kwargs["pipeline_name"]),
            pipeline_version=str(kwargs["pipeline_version"]),
            top_k=int(kwargs["top_k"]),
            started_at=NOW,
            completed_at=None,
            error_message=None,
            metadata={"requested_pipeline_name": kwargs["requested_pipeline_name"]},
        )
        return self.query_run_id

    async def get(self, query_run_id: uuid.UUID) -> QueryRunRecord | None:
        assert query_run_id == self.query_run_id
        return self.record

    async def mark_running(self, **kwargs) -> None:
        self.running = kwargs
        assert self.record is not None
        self.record = QueryRunRecord(
            id=self.record.id,
            question=self.record.question,
            answer=None,
            status=QueryRunStatus.RUNNING,
            pipeline_name=str(kwargs["pipeline_name"]),
            pipeline_version=str(kwargs["pipeline_version"]),
            top_k=self.record.top_k,
            started_at=NOW,
            completed_at=None,
            error_message=None,
            metadata=self.record.metadata,
        )

    async def mark_succeeded(self, *, query_run_id: uuid.UUID, state: QueryState) -> None:
        assert query_run_id == self.query_run_id
        self.succeeded_state = state
        self.record = QueryRunRecord(
            id=query_run_id,
            question=state.question,
            answer=state.answer,
            status=QueryRunStatus.SUCCEEDED,
            pipeline_name=state.pipeline_name,
            pipeline_version=state.pipeline_version,
            top_k=state.top_k,
            started_at=NOW,
            completed_at=NOW,
            error_message=None,
            metadata=state.metadata,
        )


class FakeUnitOfWork:
    def __init__(self) -> None:
        self.query_runs = FakeQueryRuns()
        self.background_jobs = FakeBackgroundJobs()
        self.commit_calls = 0

    async def commit(self) -> None:
        self.commit_calls += 1


async def test_submit_query_creates_pending_run_and_schedulable_job() -> None:
    uow = FakeUnitOfWork()
    registry = StubPipelineRegistry()
    scheduled_at = NOW + timedelta(minutes=10)
    handler = SubmitQueryHandler(
        uow=uow,  # type: ignore[arg-type]
        pipeline_registry=registry,  # type: ignore[arg-type]
        max_attempts=3,
        priority=20,
    )

    result = await handler(
        SubmitQueryCommand(
            question="  What is indexed?  ",
            top_k=5,
            pipeline_name="stub",
            scheduled_at=scheduled_at,
        )
    )

    submission = uow.background_jobs.submission
    assert submission is not None
    assert submission.job_type is BackgroundJobType.RUN_QUERY
    assert submission.priority == 20
    assert submission.max_attempts == 3
    assert submission.scheduled_at == scheduled_at
    assert submission.payload == {
        "query_run_id": str(uow.query_runs.query_run_id),
        "pipeline_name": "stub",
        "requested_pipeline_name": "stub",
        "top_k": 5,
    }
    assert result.query.query_run.status is QueryRunStatus.PENDING
    assert result.job.status is BackgroundJobStatus.QUEUED
    assert uow.query_runs.created["question"] == "What is indexed?"
    assert uow.commit_calls == 1


async def test_submit_query_rejects_naive_delayed_schedule() -> None:
    handler = SubmitQueryHandler(
        uow=FakeUnitOfWork(),  # type: ignore[arg-type]
        pipeline_registry=StubPipelineRegistry(),  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError, match="timezone offset"):
        await handler(
            SubmitQueryCommand(
                question="Run later",
                top_k=5,
                pipeline_name="stub",
                scheduled_at=datetime(2026, 8, 1, 9, 30),
            )
        )


async def test_query_job_executes_one_complete_graph_and_reports_generation_progress() -> None:
    uow = FakeUnitOfWork()
    await uow.query_runs.create_pending(
        question="What is indexed?",
        pipeline_name="stub",
        pipeline_version="1.0.0",
        top_k=5,
        requested_pipeline_name="stub",
    )
    reports: list[tuple[float, str]] = []

    async def report(progress: float, stage: str) -> None:
        reports.append((progress, stage))

    job_id = uuid.uuid4()
    result = await ProcessQueryJobHandler(
        uow=uow,  # type: ignore[arg-type]
        pipeline_registry=StubPipelineRegistry(),  # type: ignore[arg-type]
    )(
        job_id=job_id,
        attempt=1,
        payload={
            "query_run_id": str(uow.query_runs.query_run_id),
            "pipeline_name": "stub",
            "requested_pipeline_name": "stub",
            "top_k": 5,
        },
        report=report,
    )

    assert result["query_run_id"] == str(uow.query_runs.query_run_id)
    assert result["query_status"] == "succeeded"
    assert uow.query_runs.running["background_job_id"] == job_id
    assert uow.query_runs.succeeded_state is not None
    assert uow.query_runs.succeeded_state.answer == "A queued grounded answer."
    assert [stage for _, stage in reports] == [
        "preparing_query_execution",
        "classifying_query",
        "generating_answer",
        "saving_query_result",
    ]
    assert [progress for progress, _ in reports] == sorted(progress for progress, _ in reports)
    assert uow.commit_calls == 2


async def test_reclaimed_query_job_does_not_regenerate_an_already_persisted_answer() -> None:
    uow = FakeUnitOfWork()
    uow.query_runs.record = QueryRunRecord(
        id=uow.query_runs.query_run_id,
        question="What is indexed?",
        answer="Already persisted.",
        status=QueryRunStatus.SUCCEEDED,
        pipeline_name="stub",
        pipeline_version="1.0.0",
        top_k=5,
        started_at=NOW,
        completed_at=NOW,
        error_message=None,
        metadata={},
    )
    registry = StubPipelineRegistry()
    reports: list[tuple[float, str]] = []

    async def report(progress: float, stage: str) -> None:
        reports.append((progress, stage))

    result = await ProcessQueryJobHandler(
        uow=uow,  # type: ignore[arg-type]
        pipeline_registry=registry,  # type: ignore[arg-type]
    )(
        job_id=uuid.uuid4(),
        attempt=2,
        payload={
            "query_run_id": str(uow.query_runs.query_run_id),
            "pipeline_name": "stub",
            "top_k": 5,
        },
        report=report,
    )

    assert result["query_status"] == "succeeded"
    assert registry.pipeline.run_calls == 0
    assert reports == [(0.98, "query_result_already_available")]
    assert uow.commit_calls == 0


async def test_query_progress_advances_across_multiple_information_needs() -> None:
    reports: list[tuple[float, str]] = []

    async def report(progress: float, stage: str) -> None:
        reports.append((progress, stage))

    tracker = QueryJobProgressTracker(report)
    common = {
        "information_need_count": 2,
        "active_information_need_index": 1,
    }
    await tracker(
        GraphProgressEvent(
            node_name="retrieve",
            step_type="retrieval",
            status="started",
            graph_name="information_need_graph",
            graph_version="1.0.0",
            graph_depth=1,
            step_order=1,
            metadata={**common, "completed_information_need_count": 0},
        )
    )
    await tracker(
        GraphProgressEvent(
            node_name="complete_information_need",
            step_type="orchestration",
            status="started",
            graph_name="information_need_graph",
            graph_version="1.0.0",
            graph_depth=1,
            step_order=2,
            metadata={**common, "completed_information_need_count": 0},
        )
    )
    await tracker(
        GraphProgressEvent(
            node_name="retrieve",
            step_type="retrieval",
            status="started",
            graph_name="information_need_graph",
            graph_version="1.0.0",
            graph_depth=1,
            step_order=3,
            metadata={
                "information_need_count": 2,
                "completed_information_need_count": 1,
                "active_information_need_index": 2,
            },
        )
    )

    assert [stage for _, stage in reports] == [
        "retrieving_evidence",
        "completing_information_need",
        "retrieving_evidence",
    ]
    assert reports[0][0] < reports[1][0] < reports[2][0]
    assert reports[-1][0] < 0.80


class ProgressNode:
    name = "retrieve"
    step_type = "retrieval"

    async def __call__(self, state: QueryState) -> QueryState:
        return state


async def test_graph_runner_emits_progress_without_splitting_nodes_into_jobs() -> None:
    events: list[GraphProgressEvent] = []

    async def observe(event: GraphProgressEvent) -> None:
        events.append(event)

    state = QueryState(question="What is indexed?", progress_observer=observe)
    graph = GraphRunner(
        name="test_graph",
        version="1.0.0",
        nodes=[NodeSpec(node=ProgressNode())],
    )

    await graph.run(state)

    assert [(event.node_name, event.status) for event in events] == [
        ("retrieve", "started"),
        ("retrieve", "succeeded"),
    ]
    assert len(state.trace) == 2  # select_pipeline plus the completed node trace

async def test_progress_observer_failure_does_not_change_graph_execution() -> None:
    async def broken_observer(event: GraphProgressEvent) -> None:
        del event
        raise RuntimeError("status store unavailable")

    state = QueryState(question="What is indexed?", progress_observer=broken_observer)
    graph = GraphRunner(
        name="test_graph",
        version="1.0.0",
        nodes=[NodeSpec(node=ProgressNode())],
    )

    result = await graph.run(state)

    assert result.error_message is None
    assert result.trace[-1].name == "retrieve"
    assert result.trace[-1].status == "succeeded"

