"""Add complete query graph background jobs.

Revision ID: 0006_query_jobs
Revises: 0005_version_deletion_jobs
"""

from __future__ import annotations

from alembic import op

revision = "0006_query_jobs"
down_revision = "0005_version_deletion_jobs"
branch_labels = None
depends_on = None


_PREVIOUS_JOB_TYPES = (
    "ingest_document",
    "rebuild_document_index",
    "contextualize_document",
    "run_evaluation",
    "delete_document",
    "delete_document_versions",
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE background_job_type ADD VALUE IF NOT EXISTS 'run_query'")


def downgrade() -> None:
    op.execute("DELETE FROM background_jobs WHERE job_type::text = 'run_query'")
    op.execute(
        "ALTER TABLE background_jobs ALTER COLUMN job_type TYPE VARCHAR "
        "USING job_type::text"
    )
    op.execute("DROP TYPE background_job_type")
    values = ", ".join(f"'{value}'" for value in _PREVIOUS_JOB_TYPES)
    op.execute(f"CREATE TYPE background_job_type AS ENUM ({values})")
    op.execute(
        "ALTER TABLE background_jobs ALTER COLUMN job_type "
        "TYPE background_job_type USING job_type::background_job_type"
    )
