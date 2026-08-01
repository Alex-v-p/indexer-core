from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_subject_migration_is_the_single_additive_head() -> None:
    config = Config(str(REPOSITORY_ROOT / "alembic.ini"))
    config.set_main_option(
        "script_location",
        str(REPOSITORY_ROOT / "infra" / "migrations"),
    )
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("0007_subjects")

    classification_revision = scripts.get_revision(
        "0008_subject_classification_jobs"
    )

    assert scripts.get_current_head() == "0008_subject_classification_jobs"
    assert revision is not None
    assert revision.down_revision == "0006_query_jobs"
    assert classification_revision is not None
    assert classification_revision.down_revision == "0007_subjects"


def test_subject_migration_is_additive_and_preserves_existing_documents() -> None:
    source = (
        REPOSITORY_ROOT
        / "infra"
        / "migrations"
        / "versions"
        / "0007_add_subjects.py"
    ).read_text(encoding="utf-8")

    assert 'op.create_table(\n        "subjects"' in source
    assert 'op.create_table(\n        "subject_aliases"' in source
    assert 'op.create_table(\n        "document_subject_decisions"' in source
    assert "UPDATE documents" not in source
    assert "ALTER TABLE documents" not in source
    assert '["documents.id"],\n            ondelete="CASCADE"' in source


def test_subject_classification_job_migration_is_additive() -> None:
    source = (
        REPOSITORY_ROOT
        / "infra"
        / "migrations"
        / "versions"
        / "0008_add_subject_classification_jobs.py"
    ).read_text(encoding="utf-8")

    assert "ALTER TYPE background_job_type ADD VALUE IF NOT EXISTS" in source
    assert "'classify_document_subjects'" in source
    assert "CREATE TABLE" not in source.upper()
