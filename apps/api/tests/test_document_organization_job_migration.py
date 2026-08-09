from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[3]


def test_document_organization_job_migration_is_single_additive_head() -> None:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "infra" / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("0010_document_organization_jobs")

    assert scripts.get_current_head() == "0010_document_organization_jobs"
    assert revision is not None
    assert revision.down_revision == "0009_document_organization"


def test_job_migration_adds_new_type_and_payload_constraint_without_removing_legacy() -> None:
    source = (ROOT / "infra" / "migrations" / "versions" / "0010_add_document_organization_jobs.py").read_text(encoding="utf-8")
    assert "ALTER TYPE background_job_type ADD VALUE IF NOT EXISTS" in source
    assert "'classify_document_organization'" in source
    assert "ck_background_jobs_organization_payload" in source
    assert '"classify_document_subjects"' in source
