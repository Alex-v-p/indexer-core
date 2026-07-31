from __future__ import annotations

from fastapi import Depends

from app.composition import build_document_object_store
from app.core.config import Settings, get_settings
from app.dependencies.database import get_unit_of_work
from app.dependencies.query_runtime import get_query_pipeline_registry
from packages.indexer_application.commands import (
    EnqueueDocumentDeletionHandler,
    EnqueueDocumentMaintenanceHandler,
    EnqueueEvaluationHandler,
    ExecuteQueryHandler,
    SubmitDocumentIngestionHandler,
)
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.queries import (
    GetBackgroundJobHandler,
    GetDocumentHandler,
    GetQueryRunHandler,
    ListBackgroundJobsHandler,
    ListDocumentsHandler,
)
from packages.rag_core.pipelines import PipelineRegistry


def get_submit_document_ingestion_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> SubmitDocumentIngestionHandler:
    return SubmitDocumentIngestionHandler(
        uow=uow,
        object_store=build_document_object_store(settings),
        max_attempts=settings.background_job_ingestion_max_attempts,
    )


def get_enqueue_document_deletion_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueDocumentDeletionHandler:
    return EnqueueDocumentDeletionHandler(
        uow=uow,
        max_attempts=settings.background_job_deletion_max_attempts,
    )


def get_enqueue_document_maintenance_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueDocumentMaintenanceHandler:
    return EnqueueDocumentMaintenanceHandler(
        uow=uow,
        max_attempts=settings.background_job_maintenance_max_attempts,
    )


def get_enqueue_evaluation_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> EnqueueEvaluationHandler:
    return EnqueueEvaluationHandler(
        uow=uow,
        max_attempts=settings.background_job_evaluation_max_attempts,
    )


def get_background_job_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetBackgroundJobHandler:
    return GetBackgroundJobHandler(uow=uow)


def get_list_background_jobs_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListBackgroundJobsHandler:
    return ListBackgroundJobsHandler(uow=uow)


def get_execute_query_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    pipeline_registry: PipelineRegistry = Depends(get_query_pipeline_registry),
) -> ExecuteQueryHandler:
    return ExecuteQueryHandler(uow=uow, pipeline_registry=pipeline_registry)


def get_document_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetDocumentHandler:
    return GetDocumentHandler(uow=uow)


def get_list_documents_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> ListDocumentsHandler:
    return ListDocumentsHandler(uow=uow)


def get_query_run_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
) -> GetQueryRunHandler:
    return GetQueryRunHandler(uow=uow)
