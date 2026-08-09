from __future__ import annotations

import re
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[3]
EXPLICIT_IDENTIFIER = re.compile(r"[\"']((?:ck|uq|ix|fk)_[a-z0-9_]+)[\"']")


def test_document_organization_migration_is_single_additive_head() -> None:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "infra" / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("0009_document_organization")

    assert scripts.get_current_head() == "0010_document_organization_jobs"
    assert revision is not None
    assert revision.down_revision == "0008_subject_classification_jobs"


def test_migration_seeds_extensible_catalogue_without_touching_legacy_subjects() -> None:
    source = (ROOT / "infra" / "migrations" / "versions" / "0009_add_document_organization.py").read_text(encoding="utf-8")
    for key in ("plan", "report", "specification", "research", "presentation", "notes", "reference", "other"):
        assert f'"{key}"' in source
    assert "ON CONFLICT (key) DO UPDATE" in source
    assert 'op.create_table(\n        "document_types"' in source
    assert 'op.create_table(\n        "content_groups"' in source
    assert 'op.create_table(\n        "document_content_group_assignments"' in source
    assert "ALTER TABLE subjects" not in source
    assert "document_subject_decisions" in source  # compatibility/rollback comment only
    assert "op.drop_table(\"subjects\")" not in source


def test_new_migration_and_orm_identifiers_fit_postgresql_limit() -> None:
    paths = (
        ROOT / "infra" / "migrations" / "versions" / "0009_add_document_organization.py",
        ROOT / "infra" / "migrations" / "versions" / "0010_add_document_organization_jobs.py",
        ROOT / "packages" / "indexer_infrastructure" / "postgres" / "models" / "document_organization.py",
        ROOT / "packages" / "indexer_infrastructure" / "postgres" / "models" / "background_jobs.py",
    )
    identifiers = {
        identifier
        for path in paths
        for identifier in EXPLICIT_IDENTIFIER.findall(path.read_text(encoding="utf-8"))
    }

    assert identifiers
    assert {
        identifier: len(identifier)
        for identifier in sorted(identifiers)
        if len(identifier) > 63
    } == {}
