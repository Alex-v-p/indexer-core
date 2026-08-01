"""Add typed subjects and logical-document subject decisions.

Revision ID: 0007_subjects
Revises: 0006_query_jobs
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0007_subjects"
down_revision = "0006_query_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subjects",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("normalized_name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('project', 'topic', 'organization', 'custom')",
            name="ck_subjects_kind",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "kind",
            "normalized_name",
            name="uq_subjects_kind_normalized_name",
        ),
    )
    op.create_index(
        "ix_subjects_kind_archived_at",
        "subjects",
        ["kind", "archived_at"],
        unique=False,
    )

    op.create_table(
        "subject_aliases",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("normalized_name", sa.String(length=255), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"],
            ["subjects.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "subject_id",
            "normalized_name",
            name="uq_subject_aliases_subject_normalized_name",
        ),
    )
    op.create_index(
        "ix_subject_aliases_normalized_name",
        "subject_aliases",
        ["normalized_name"],
        unique=False,
    )

    op.create_table(
        "document_subject_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("subject_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("control_source", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_band", sa.String(length=16), nullable=True),
        sa.Column("rationale", sa.String(length=2000), nullable=True),
        sa.Column("classifier_version", sa.String(length=128), nullable=True),
        sa.Column("policy_version", sa.String(length=128), nullable=True),
        sa.Column(
            "signals",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        # Retain the classified version as audit identity even if that version
        # is later removed from the logical document.
        sa.Column(
            "classified_document_version_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "revision",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state IN ('suggested', 'assigned', 'rejected')",
            name="ck_document_subject_decisions_state",
        ),
        sa.CheckConstraint(
            "control_source IN ('manual', 'automatic')",
            name="ck_document_subject_decisions_control_source",
        ),
        sa.CheckConstraint(
            "confidence_band IS NULL OR confidence_band IN ('low', 'medium', 'high')",
            name="ck_document_subject_decisions_confidence_band",
        ),
        sa.CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_document_subject_decisions_confidence",
        ),
        sa.CheckConstraint(
            "revision > 0",
            name="ck_document_subject_decisions_revision",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(signals) = 'object' AND pg_column_size(signals) <= 16384",
            name="ck_document_subject_decisions_signals",
        ),
        sa.CheckConstraint(
            "control_source <> 'automatic' OR "
            "(confidence IS NOT NULL AND confidence_band IS NOT NULL AND "
            "rationale IS NOT NULL AND classifier_version IS NOT NULL AND "
            "policy_version IS NOT NULL AND classified_document_version_id IS NOT NULL)",
            name="ck_document_subject_decisions_automatic_provenance",
        ),
        sa.CheckConstraint(
            "control_source <> 'manual' OR "
            "(confidence IS NULL AND confidence_band IS NULL AND "
            "classifier_version IS NULL AND policy_version IS NULL AND "
            "classified_document_version_id IS NULL AND signals = '{}'::jsonb)",
            name="ck_document_subject_decisions_manual_provenance",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["subject_id"],
            ["subjects.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_id",
            "subject_id",
            name="uq_document_subject_decisions_document_subject",
        ),
    )
    op.create_index(
        "ix_document_subject_decisions_subject_state",
        "document_subject_decisions",
        ["subject_id", "state"],
        unique=False,
    )
    op.create_index(
        "ix_document_subject_decisions_document_state",
        "document_subject_decisions",
        ["document_id", "state"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_subject_decisions_document_state",
        table_name="document_subject_decisions",
    )
    op.drop_index(
        "ix_document_subject_decisions_subject_state",
        table_name="document_subject_decisions",
    )
    op.drop_table("document_subject_decisions")
    op.drop_index(
        "ix_subject_aliases_normalized_name",
        table_name="subject_aliases",
    )
    op.drop_table("subject_aliases")
    op.drop_index("ix_subjects_kind_archived_at", table_name="subjects")
    op.drop_table("subjects")
