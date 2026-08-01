from __future__ import annotations

import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.indexer_bootstrap.composition import (
    build_chunk_contextualizer,
    build_document_context_hierarchy_builder,
    build_document_ingestion_config,
    build_document_object_store,
    build_embedding_provider,
    build_keyword_cache_invalidator,
    build_query_graph,
    build_query_pipeline_registry,
    build_language_model,
    build_subject_classification_policy,
    build_vector_store,
)
from packages.indexer_bootstrap.config import Settings
from packages.indexer_application.dto import BackgroundJobRecord, BackgroundJobType
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.services.background_jobs import (
    DeleteDocumentVersionsJobHandler,
    ClassifyDocumentSubjectsJobHandler,
    ProcessDocumentIngestionJobHandler,
    ProcessQueryJobHandler,
    ProgressReporter,
    ReindexDocumentJobHandler,
    SubjectClassificationJobConfig,
)
from packages.indexer_application.services.query_subject_scope import QuerySubjectScopeConfig
from packages.rag_core.subjects import (
    StructuredSubjectDiscoveryProvider,
    StructuredSubjectModelEvidenceProvider,
)
from packages.rag_core.evaluation import (
    EvaluationRunner,
    StabilityEvaluationReport,
    StabilityEvaluationRunner,
    load_evaluation_dataset,
    write_evaluation_report,
    write_stability_evaluation_report,
)


class BackgroundJobDispatcher:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._object_store = build_document_object_store(settings)
        self._embedding_provider = build_embedding_provider(settings)
        self._vector_store = build_vector_store(settings)
        self._keyword_cache = build_keyword_cache_invalidator(settings)
        self._hierarchy_builder = build_document_context_hierarchy_builder(settings)
        self._contextualizer = build_chunk_contextualizer(
            settings,
            hierarchy_builder=self._hierarchy_builder,
        )
        self._ingestion_config = build_document_ingestion_config(settings)
        self._query_pipeline_registry = build_query_pipeline_registry(settings)
        self._subject_classification_config = SubjectClassificationJobConfig(
            policy=build_subject_classification_policy(settings),
            max_summary_chars=settings.subject_classification_max_summary_chars,
            discovery_enabled=settings.subject_classification_discovery_enabled,
        )
        self._subject_model_evidence = (
            StructuredSubjectModelEvidenceProvider(
                provider=build_language_model(settings),
                max_summary_chars=settings.subject_classification_max_summary_chars,
                max_repair_attempts=settings.structured_output_max_repair_attempts,
            )
            if settings.subject_classification_model_enabled
            else None
        )
        self._subject_discovery = (
            StructuredSubjectDiscoveryProvider(
                provider=build_language_model(settings),
                max_summary_chars=settings.subject_classification_max_summary_chars,
                max_repair_attempts=settings.structured_output_max_repair_attempts,
            )
            if settings.subject_classification_model_enabled
            and settings.subject_classification_discovery_enabled
            else None
        )

    async def dispatch(
        self,
        *,
        job: BackgroundJobRecord,
        uow: UnitOfWork,
        report: ProgressReporter,
    ) -> dict[str, Any]:
        if job.job_type is BackgroundJobType.INGEST_DOCUMENT:
            return await self._ingestion_handler(uow)(job.payload, report)
        if job.job_type is BackgroundJobType.REBUILD_DOCUMENT_INDEX:
            return await self._reindex_handler(uow)(job.payload, report)
        if job.job_type is BackgroundJobType.CONTEXTUALIZE_DOCUMENT:
            return await self._reindex_handler(uow)(
                job.payload,
                report,
                require_contextualization=True,
            )
        if job.job_type is BackgroundJobType.RUN_EVALUATION:
            return await self._run_evaluation(job=job, report=report)
        if job.job_type is BackgroundJobType.RUN_QUERY:
            return await ProcessQueryJobHandler(
                uow=uow,
                pipeline_registry=self._query_pipeline_registry,
                subject_scope_config=QuerySubjectScopeConfig(
                    max_document_ids=self._settings.query_subject_scope_max_document_ids,
                    max_project_lanes=self._settings.query_subject_scope_max_project_lanes,
                    policy_revision=self._settings.query_subject_scope_policy_revision,
                ),
            )(
                job_id=job.id,
                attempt=job.attempts,
                payload=job.payload,
                report=report,
            )
        if job.job_type is BackgroundJobType.DELETE_DOCUMENT_VERSIONS:
            return await self._delete_versions_handler(uow)(job.payload, report)
        if job.job_type is BackgroundJobType.CLASSIFY_DOCUMENT_SUBJECTS:
            if not self._settings.subject_classification_enabled:
                raise ValueError("Automatic subject classification is disabled.")
            return await ClassifyDocumentSubjectsJobHandler(
                uow=uow,
                config=self._subject_classification_config,
                model_evidence=self._subject_model_evidence,
                subject_discovery=self._subject_discovery,
            )(job.payload, report, job_id=job.id)
        if job.job_type is BackgroundJobType.DELETE_DOCUMENT:
            raise ValueError(
                "Legacy whole-document deletion jobs are no longer supported; "
                "submit explicit document-version deletions instead."
            )
        raise ValueError(f"Unsupported background job type: {job.job_type.value}.")

    def _delete_versions_handler(self, uow: UnitOfWork) -> DeleteDocumentVersionsJobHandler:
        return DeleteDocumentVersionsJobHandler(
            uow=uow,
            object_store=self._object_store,
            document_index=self._vector_store,
            version_index=self._vector_store,
            keyword_cache=self._keyword_cache,
        )

    def _ingestion_handler(self, uow: UnitOfWork) -> ProcessDocumentIngestionJobHandler:
        return ProcessDocumentIngestionJobHandler(
            uow=uow,
            config=self._ingestion_config,
            object_store=self._object_store,
            embedding_provider=self._embedding_provider,
            vector_index=self._vector_store,
            version_index=self._vector_store,
            keyword_cache=self._keyword_cache,
            contextualizer=self._contextualizer,
            hierarchy_builder=self._hierarchy_builder,
            subject_classification_enabled=self._settings.subject_classification_enabled,
            subject_classification_policy_version=(
                self._settings.subject_classification_policy_version
            ),
            subject_classification_max_attempts=(
                self._settings.background_job_subject_classification_max_attempts
            ),
        )

    def _reindex_handler(self, uow: UnitOfWork) -> ReindexDocumentJobHandler:
        return ReindexDocumentJobHandler(
            uow=uow,
            config=self._ingestion_config,
            object_store=self._object_store,
            embedding_provider=self._embedding_provider,
            vector_index=self._vector_store,
            version_index=self._vector_store,
            keyword_cache=self._keyword_cache,
            contextualizer=self._contextualizer,
            hierarchy_builder=self._hierarchy_builder,
            subject_classification_enabled=self._settings.subject_classification_enabled,
            subject_classification_policy_version=(
                self._settings.subject_classification_policy_version
            ),
            subject_classification_max_attempts=(
                self._settings.background_job_subject_classification_max_attempts
            ),
        )

    async def _run_evaluation(
        self,
        *,
        job: BackgroundJobRecord,
        report: ProgressReporter,
    ) -> dict[str, Any]:
        await report(0.05, "loading_evaluation_dataset")
        dataset_path = _resolve_dataset_path(
            str(job.payload.get("dataset_path") or ""),
            configured_root=_resolve_configured_directory(
                self._settings.evaluation_dataset_dir,
                create=False,
            ),
        )
        dataset = load_evaluation_dataset(dataset_path)
        pipeline_name = _optional_string(job.payload.get("pipeline_name"))
        top_k = _optional_int(job.payload.get("top_k"))
        repetitions = _optional_int(job.payload.get("repetitions")) or 1

        await report(0.12, "building_evaluation_pipeline")
        graph = build_query_graph(self._settings, pipeline_name=pipeline_name)
        await report(0.18, "running_evaluation")
        if repetitions >= 2:
            evaluation_report = await StabilityEvaluationRunner(
                graph=graph,
                requested_pipeline_name=pipeline_name,
            ).run(
                dataset,
                repetitions=repetitions,
                top_k_override=top_k,
            )
        else:
            evaluation_report = await EvaluationRunner(
                graph=graph,
                requested_pipeline_name=pipeline_name,
            ).run(dataset, top_k_override=top_k)

        await report(0.92, "writing_evaluation_report")
        report_dir = _resolve_configured_directory(
            self._settings.evaluation_report_dir,
            create=True,
        )
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", dataset.name).strip("-") or "evaluation"
        suffix = "stability" if isinstance(evaluation_report, StabilityEvaluationReport) else "evaluation"
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = report_dir / f"{safe_name}-{suffix}-{job.id}-{timestamp}.json"
        if isinstance(evaluation_report, StabilityEvaluationReport):
            write_stability_evaluation_report(evaluation_report, output_path)
        else:
            write_evaluation_report(evaluation_report, output_path)

        result: dict[str, Any] = {
            "dataset_name": evaluation_report.dataset_name,
            "dataset_version": evaluation_report.dataset_version,
            "pipeline_name": evaluation_report.pipeline_name,
            "pipeline_version": evaluation_report.pipeline_version,
            "duration_ms": evaluation_report.duration_ms,
            "report_path": str(output_path),
            "metrics": asdict(evaluation_report.metrics),
        }
        if isinstance(evaluation_report, StabilityEvaluationReport):
            result.update(
                {
                    "repetitions": evaluation_report.repetitions,
                    "total_attempts": evaluation_report.total_attempts,
                    "failed_attempts": evaluation_report.failed_attempts,
                },
            )
        else:
            result.update(
                {
                    "total_cases": evaluation_report.total_cases,
                    "failed_cases": evaluation_report.failed_cases,
                },
            )
        return result


def _resolve_configured_directory(raw_path: str, *, create: bool) -> Path:
    configured = Path(raw_path).expanduser()
    if configured.is_absolute():
        resolved = configured.resolve()
    else:
        working_directory = Path.cwd().resolve()
        repository_root = next(
            (
                candidate
                for candidate in (working_directory, *working_directory.parents)
                if (candidate / "compose.yaml").is_file()
            ),
            working_directory,
        )
        resolved = (repository_root / configured).resolve()
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _resolve_dataset_path(raw_path: str, *, configured_root: Path) -> Path:
    if not raw_path.strip():
        raise ValueError("Evaluation job dataset_path must not be empty.")
    root = configured_root.resolve()
    raw = Path(raw_path)
    candidates = [raw.resolve()] if raw.is_absolute() else [(Path.cwd() / raw).resolve(), (root / raw).resolve()]
    for candidate in candidates:
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.is_file():
            return candidate
    raise ValueError(f"Evaluation dataset must be a JSON file under {root}: {raw_path}")


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Expected an integer job payload value.")
    return value
