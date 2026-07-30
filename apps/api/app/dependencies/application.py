from __future__ import annotations

from fastapi import Depends

from app.composition import (
    build_chunk_contextualizer,
    build_document_context_hierarchy_builder,
    build_document_ingestion_config,
    build_document_object_store,
    build_embedding_provider,
    build_keyword_cache_invalidator,
    build_vector_store,
)
from app.core.config import Settings, get_settings
from app.dependencies.database import get_unit_of_work
from app.dependencies.query_runtime import get_query_pipeline_registry
from packages.indexer_application.commands import ExecuteQueryHandler, IngestDocumentHandler
from packages.indexer_application.ports import UnitOfWork
from packages.indexer_application.queries import (
    GetDocumentHandler,
    GetQueryRunHandler,
    ListDocumentsHandler,
)
from packages.rag_core.pipelines import PipelineRegistry


def get_ingest_document_handler(
    uow: UnitOfWork = Depends(get_unit_of_work),
    settings: Settings = Depends(get_settings),
) -> IngestDocumentHandler:
    hierarchy_builder = build_document_context_hierarchy_builder(settings)
    contextualizer = build_chunk_contextualizer(
        settings,
        hierarchy_builder=hierarchy_builder,
    )
    vector_store = build_vector_store(settings)
    return IngestDocumentHandler(
        uow=uow,
        config=build_document_ingestion_config(settings),
        object_store=build_document_object_store(settings),
        embedding_provider=build_embedding_provider(settings),
        vector_index=vector_store,
        version_index=vector_store,
        keyword_cache=build_keyword_cache_invalidator(settings),
        contextualizer=contextualizer,
        hierarchy_builder=hierarchy_builder,
    )


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
