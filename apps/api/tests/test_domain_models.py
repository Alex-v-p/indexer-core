from sqlalchemy.orm import configure_mappers

from app.adapters.database import Base
from app.adapters.database.models import DocumentStatus, QueryRunStatus, TraceStepStatus


def test_core_domain_tables_are_registered() -> None:
    expected_tables = {
        "documents",
        "document_versions",
        "chunks",
        "queries",
        "query_runs",
        "evidence",
        "citations",
        "trace_steps",
    }

    assert expected_tables.issubset(Base.metadata.tables.keys())


def test_core_domain_relationships_are_configurable() -> None:
    configure_mappers()


def test_domain_status_enums_use_storage_values() -> None:
    assert DocumentStatus.UPLOADED.value == "uploaded"
    assert QueryRunStatus.PENDING.value == "pending"
    assert TraceStepStatus.SKIPPED.value == "skipped"
