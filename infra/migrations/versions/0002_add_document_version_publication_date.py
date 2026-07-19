"""add document version publication date

Revision ID: 0002_version_publication_date
Revises: 0001_core_domain_models
Create Date: 2026-07-18
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_version_publication_date"
down_revision: str | None = "0001_core_domain_models"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_versions",
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_document_versions_published_at",
        "document_versions",
        ["published_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_versions_published_at", table_name="document_versions")
    op.drop_column("document_versions", "published_at")
