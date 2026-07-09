from sqlalchemy.orm import configure_mappers

from app.adapters.database import Base
from app.adapters.database.models import Chunk, DocumentStatus, QueryRun, QueryRunStatus, TraceStepStatus


def test_core_domain_tables_are_registered() -> None:
    expected_tables = {
        "documents",
        "document_versions",
        "chunks",
        "query_runs",
        "evidence",
        "citations",
        "trace_steps",
    }

    assert expected_tables.issubset(Base.metadata.tables.keys())
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


def test_chunk_model_is_lightweight_qdrant_registry() -> None:
    columns = Chunk.__table__.columns

    assert "text" not in columns
    assert "ordinal" in columns
    assert "content_hash" in columns
    assert "qdrant_collection" in columns
    assert "qdrant_point_id" in columns
