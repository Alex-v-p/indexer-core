"""Add extensible document types and exactly-one content-group assignment.

Revision ID: 0009_document_organization
Revises: 0008_subject_classification_jobs

Compatibility: this migration is additive and deliberately does not transform,
copy, or reinterpret legacy subjects and document-subject decisions. Rollback
drops only the new document-organization objects; legacy data remains intact.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0009_document_organization"
down_revision = "0008_subject_classification_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "document_types",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("key ~ '^[a-z][a-z0-9]*(_[a-z0-9]+)*$'", name="ck_document_types_key"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", name="uq_document_types_key"),
    )
    op.create_index("ix_document_types_archived_key", "document_types", ["archived_at", "key"], unique=False)

    op.create_table(
        "content_groups",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("normalized_name", sa.String(length=80), nullable=False),
        sa.Column("description", sa.String(length=2000), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("normalized_name", name="uq_content_groups_normalized_name"),
    )
    op.create_index("ix_content_groups_archived_name", "content_groups", ["archived_at", "normalized_name"], unique=False)

    op.create_table(
        "content_group_aliases",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_group_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=80), nullable=False),
        sa.Column("normalized_name", sa.String(length=80), nullable=False),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["content_group_id"], ["content_groups.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        # Alias names are unique across the catalogue. The repository also uses
        # a normalized-name advisory lock to prevent canonical/alias ambiguity.
        sa.UniqueConstraint("normalized_name", name="uq_content_group_aliases_normalized_name"),
    )
    op.create_index("ix_content_group_aliases_group", "content_group_aliases", ["content_group_id", "archived_at"], unique=False)

    op.create_table(
        "document_type_decisions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_type_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_band", sa.String(length=16), nullable=True),
        sa.Column("rationale", sa.String(length=2000), nullable=True),
        sa.Column("classifier_version", sa.String(length=128), nullable=True),
        sa.Column("policy_version", sa.String(length=128), nullable=True),
        sa.Column("signals", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("classified_document_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("state IN ('suggested', 'assigned', 'rejected')", name="ck_document_type_decisions_state"),
        sa.CheckConstraint("source IN ('manual', 'automatic')", name="ck_document_type_decisions_source"),
        sa.CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_document_type_decisions_confidence"),
        sa.CheckConstraint("confidence_band IS NULL OR confidence_band IN ('low', 'medium', 'high')", name="ck_document_type_decisions_confidence_band"),
        sa.CheckConstraint("revision > 0", name="ck_document_type_decisions_revision"),
        sa.CheckConstraint("jsonb_typeof(signals) = 'object' AND pg_column_size(signals) <= 16384", name="ck_document_type_decisions_signals"),
        sa.CheckConstraint("source <> 'automatic' OR (confidence IS NOT NULL AND confidence_band IS NOT NULL AND rationale IS NOT NULL AND classifier_version IS NOT NULL AND policy_version IS NOT NULL AND classified_document_version_id IS NOT NULL)", name="ck_document_type_decisions_automatic_provenance"),
        sa.CheckConstraint("source <> 'manual' OR (confidence IS NULL AND confidence_band IS NULL AND classifier_version IS NULL AND policy_version IS NULL AND classified_document_version_id IS NULL AND signals = '{}'::jsonb)", name="ck_document_type_decisions_manual_provenance"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["document_type_id"], ["document_types.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("document_id", "document_type_id", name="uq_document_type_decisions_document_type"),
    )
    op.create_index("ix_document_type_decisions_type_state", "document_type_decisions", ["document_type_id", "state"], unique=False)
    op.create_index("ix_document_type_decisions_document_state", "document_type_decisions", ["document_id", "state"], unique=False)

    op.create_table(
        "document_content_group_assignments",
        # The document primary key makes exactly one current assignment row per
        # logical document a database invariant under concurrent upserts.
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_group_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("unresolved_reason", sa.String(length=1000), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_band", sa.String(length=16), nullable=True),
        sa.Column("rationale", sa.String(length=2000), nullable=True),
        sa.Column("classifier_version", sa.String(length=128), nullable=True),
        sa.Column("policy_version", sa.String(length=128), nullable=True),
        sa.Column("signals", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
        sa.Column("summary_hash", sa.String(length=128), nullable=True),
        sa.Column("classified_document_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.CheckConstraint("state IN ('pending', 'unresolved', 'suggested', 'assigned')", name="ck_document_content_group_assignments_state"),
        sa.CheckConstraint("source IN ('manual', 'automatic')", name="ck_document_content_group_assignments_source"),
        sa.CheckConstraint("((state IN ('pending', 'unresolved') AND content_group_id IS NULL) OR (state IN ('suggested', 'assigned') AND content_group_id IS NOT NULL))", name="ck_document_content_group_assignments_state_group"),
        sa.CheckConstraint("((state = 'unresolved' AND unresolved_reason IS NOT NULL) OR (state <> 'unresolved' AND unresolved_reason IS NULL))", name="ck_document_content_group_assignments_unresolved_reason"),
        sa.CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_document_content_group_assignments_confidence"),
        sa.CheckConstraint("confidence_band IS NULL OR confidence_band IN ('low', 'medium', 'high')", name="ck_document_content_group_assignments_confidence_band"),
        sa.CheckConstraint("revision > 0", name="ck_document_content_group_assignments_revision"),
        sa.CheckConstraint("jsonb_typeof(signals) = 'object' AND pg_column_size(signals) <= 16384", name="ck_document_content_group_assignments_signals"),
        sa.CheckConstraint("source <> 'manual' OR (state = 'assigned' AND content_group_id IS NOT NULL AND confidence IS NULL AND confidence_band IS NULL AND classifier_version IS NULL AND policy_version IS NULL AND summary_hash IS NULL AND classified_document_version_id IS NULL AND signals = '{}'::jsonb)", name="ck_document_content_group_assignments_manual"),
        sa.CheckConstraint("source <> 'automatic' OR state NOT IN ('suggested', 'assigned') OR (confidence IS NOT NULL AND confidence_band IS NOT NULL AND rationale IS NOT NULL AND classifier_version IS NOT NULL AND policy_version IS NOT NULL AND summary_hash IS NOT NULL AND classified_document_version_id IS NOT NULL)", name="ck_document_content_group_assign_auto_resolved_provenance"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["content_group_id"], ["content_groups.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("document_id"),
    )
    op.create_index("ix_document_content_group_assignments_group_state", "document_content_group_assignments", ["content_group_id", "state"], unique=False)

    _seed_document_types()


def _seed_document_types() -> None:
    rows = (
        ("00000000-0000-4000-8000-000000000001", "plan", "Plan"),
        ("00000000-0000-4000-8000-000000000002", "report", "Report"),
        ("00000000-0000-4000-8000-000000000003", "specification", "Specification"),
        ("00000000-0000-4000-8000-000000000004", "research", "Research"),
        ("00000000-0000-4000-8000-000000000005", "presentation", "Presentation"),
        ("00000000-0000-4000-8000-000000000006", "notes", "Notes"),
        ("00000000-0000-4000-8000-000000000007", "reference", "Reference"),
        ("00000000-0000-4000-8000-000000000008", "other", "Other"),
    )
    values = ",\n".join(
        f"('{identifier}'::uuid, '{key}', '{label}', '{{}}'::jsonb)"
        for identifier, key, label in rows
    )
    op.execute(
        sa.text(
            "INSERT INTO document_types (id, key, label, metadata) VALUES "
            + values
            + " ON CONFLICT (key) DO UPDATE SET label = EXCLUDED.label, updated_at = now()"
        )
    )


def downgrade() -> None:
    # Rollback intentionally removes only the additive replacement foundation.
    # Legacy subjects and document_subject_decisions are never touched.
    op.drop_index("ix_document_content_group_assignments_group_state", table_name="document_content_group_assignments")
    op.drop_table("document_content_group_assignments")
    op.drop_index("ix_document_type_decisions_document_state", table_name="document_type_decisions")
    op.drop_index("ix_document_type_decisions_type_state", table_name="document_type_decisions")
    op.drop_table("document_type_decisions")
    op.drop_index("ix_content_group_aliases_group", table_name="content_group_aliases")
    op.drop_table("content_group_aliases")
    op.drop_index("ix_content_groups_archived_name", table_name="content_groups")
    op.drop_table("content_groups")
    op.drop_index("ix_document_types_archived_key", table_name="document_types")
    op.drop_table("document_types")
