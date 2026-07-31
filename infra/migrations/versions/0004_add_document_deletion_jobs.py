"""Add document deletion background jobs.

Revision ID: 0004_document_deletion_jobs
Revises: 0003_background_jobs
"""

from __future__ import annotations

from alembic import op

revision = "0004_document_deletion_jobs"
down_revision = "0003_background_jobs"
branch_labels = None
depends_on = None


_OLD_JOB_TYPES = (
    "ingest_document",
    "rebuild_document_index",
    "contextualize_document",
    "run_evaluation",
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE background_job_type ADD VALUE IF NOT EXISTS 'delete_document'")


def downgrade() -> None:
    op.execute("DELETE FROM background_jobs WHERE job_type::text = 'delete_document'")
    op.execute(
        "ALTER TABLE background_jobs ALTER COLUMN job_type TYPE VARCHAR "
        "USING job_type::text"
    )
    op.execute("DROP TYPE background_job_type")
    values = ", ".join(f"'{value}'" for value in _OLD_JOB_TYPES)
    op.execute(f"CREATE TYPE background_job_type AS ENUM ({values})")
    op.execute(
        "ALTER TABLE background_jobs ALTER COLUMN job_type "
        "TYPE background_job_type USING job_type::background_job_type"
    )
