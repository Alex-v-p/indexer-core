"""Add durable document subject classification jobs.

Revision ID: 0008_subject_classification_jobs
Revises: 0007_subjects
"""

from __future__ import annotations

from alembic import op

revision = "0008_subject_classification_jobs"
down_revision = "0007_subjects"
branch_labels = None
depends_on = None

_PREVIOUS_JOB_TYPES = (
    "ingest_document",
    "rebuild_document_index",
    "contextualize_document",
    "run_evaluation",
    "delete_document",
    "delete_document_versions",
    "run_query",
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE background_job_type ADD VALUE IF NOT EXISTS "
            "'classify_document_subjects'"
        )


def downgrade() -> None:
    op.execute(
        "DELETE FROM background_jobs "
        "WHERE job_type::text = 'classify_document_subjects'"
    )
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
