"""Add asynchronous document-organization classification jobs.

Revision ID: 0010_document_organization_jobs
Revises: 0009_document_organization
"""

from __future__ import annotations

from alembic import op

revision = "0010_document_organization_jobs"
down_revision = "0009_document_organization"
branch_labels = None
depends_on = None

_PREVIOUS_JOB_TYPES = (
    "ingest_document",
    "rebuild_document_index",
    "contextualize_document",
    "run_evaluation",
    "run_query",
    "delete_document",
    "delete_document_versions",
    "classify_document_subjects",
)


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute(
            "ALTER TYPE background_job_type ADD VALUE IF NOT EXISTS "
            "'classify_document_organization'"
        )
    op.create_check_constraint(
        "ck_background_jobs_organization_payload",
        "background_jobs",
        "job_type::text <> 'classify_document_organization' OR "
        "(payload ? 'document_id' AND payload ? 'document_version_id' AND "
        "payload ? 'policy_version' AND "
        "jsonb_typeof(payload->'document_id') = 'string' AND "
        "jsonb_typeof(payload->'document_version_id') = 'string' AND "
        "jsonb_typeof(payload->'policy_version') = 'string')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_background_jobs_organization_payload",
        "background_jobs",
        type_="check",
    )
    op.execute(
        "DELETE FROM background_jobs "
        "WHERE job_type::text = 'classify_document_organization'"
    )
    op.execute("ALTER TABLE background_jobs ALTER COLUMN job_type TYPE VARCHAR USING job_type::text")
    op.execute("DROP TYPE background_job_type")
    values = ", ".join(f"'{value}'" for value in _PREVIOUS_JOB_TYPES)
    op.execute(f"CREATE TYPE background_job_type AS ENUM ({values})")
    op.execute(
        "ALTER TABLE background_jobs ALTER COLUMN job_type TYPE background_job_type "
        "USING job_type::background_job_type"
    )
