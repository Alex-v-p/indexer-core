from sqlalchemy.orm import configure_mappers

from packages.indexer_infrastructure.postgres import Base
from packages.indexer_application.dto import DocumentStatus, QueryRunStatus, TraceStepStatus
from packages.indexer_infrastructure.postgres.models import QdrantChunkIndex, QueryRun


def test_core_domain_tables_are_registered() -> None:
    expected_tables = {
        "documents",
        "document_versions",
        "qdrant_chunk_indexes",
        "query_runs",
        "evidence",
        "citations",
        "trace_steps",
        "subjects",
        "subject_aliases",
        "document_subject_decisions",
    }

    assert expected_tables.issubset(Base.metadata.tables.keys())
    assert "chunks" not in Base.metadata.tables
    assert "queries" not in Base.metadata.tables


def test_core_domain_relationships_are_configurable() -> None:
    configure_mappers()


def test_domain_status_enums_use_storage_values() -> None:
    assert DocumentStatus.UPLOADED.value == "uploaded"
    assert QueryRunStatus.PENDING.value == "pending"
    assert TraceStepStatus.SKIPPED.value == "skipped"


def test_query_run_stores_question_without_separate_query_table() -> None:
    columns = QueryRun.__table__.columns

    assert "question" in columns
    assert "answer" in columns
    assert "query_id" not in columns


def test_qdrant_chunk_index_is_lightweight_registry() -> None:
    columns = QdrantChunkIndex.__table__.columns

    assert "text" not in columns
    assert "ordinal" in columns
    assert "content_hash" in columns
    assert "qdrant_collection" in columns
    assert "qdrant_point_id" in columns
