"""add durable background jobs

Revision ID: 0003_background_jobs
Revises: 0002_version_publication_date
Create Date: 2026-07-31
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_background_jobs"
down_revision: str | None = "0002_version_publication_date"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

background_job_type = sa.Enum(
    "ingest_document",
    "rebuild_document_index",
    "contextualize_document",
    "run_evaluation",
    name="background_job_type",
)
background_job_status = sa.Enum(
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    name="background_job_status",
)


def upgrade() -> None:
    op.create_table(
        "background_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_type", background_job_type, nullable=False),
        sa.Column("status", background_job_status, server_default="queued", nullable=False),
        sa.Column("priority", sa.Integer(), server_default="100", nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "result",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("progress", sa.Float(), server_default="0", nullable=False),
        sa.Column("current_stage", sa.String(length=255), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default="3", nullable=False),
        sa.Column("dedupe_key", sa.String(length=512), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(length=255), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("progress >= 0 AND progress <= 1", name="ck_background_jobs_progress_range"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_background_jobs_status_scheduled",
        "background_jobs",
        ["status", "scheduled_at", "priority"],
    )
    op.create_index(
        "ix_background_jobs_type_created",
        "background_jobs",
        ["job_type", "created_at"],
    )
    op.create_index(
        "ix_background_jobs_heartbeat",
        "background_jobs",
        ["status", "heartbeat_at"],
    )
    op.create_index(
        "uq_background_jobs_active_dedupe_key",
        "background_jobs",
        ["dedupe_key"],
        unique=True,
        postgresql_where=sa.text(
            "dedupe_key IS NOT NULL AND status IN ('queued', 'running')"
        ),
    )


def downgrade() -> None:
    op.drop_index("uq_background_jobs_active_dedupe_key", table_name="background_jobs")
    op.drop_index("ix_background_jobs_heartbeat", table_name="background_jobs")
    op.drop_index("ix_background_jobs_type_created", table_name="background_jobs")
    op.drop_index("ix_background_jobs_status_scheduled", table_name="background_jobs")
    op.drop_table("background_jobs")
    background_job_status.drop(op.get_bind(), checkfirst=True)
    background_job_type.drop(op.get_bind(), checkfirst=True)
