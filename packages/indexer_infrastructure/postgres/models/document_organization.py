from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text as sa_text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.indexer_infrastructure.postgres.base import Base

if TYPE_CHECKING:
    from packages.indexer_infrastructure.postgres.models.documents import Document


class DocumentType(Base):
    __tablename__ = "document_types"
    __table_args__ = (
        UniqueConstraint("key", name="uq_document_types_key"),
        CheckConstraint(
            "key ~ '^[a-z][a-z0-9]*(_[a-z0-9]+)*$'",
            name="ck_document_types_key",
        ),
        Index("ix_document_types_archived_key", "archived_at", "key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2_000), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb")
    )
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    decisions: Mapped[list[DocumentTypeDecision]] = relationship(
        "DocumentTypeDecision", back_populates="document_type", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentTypeDecision(Base):
    __tablename__ = "document_type_decisions"
    __table_args__ = (
        UniqueConstraint("document_id", "document_type_id", name="uq_document_type_decisions_document_type"),
        CheckConstraint("state IN ('suggested', 'assigned', 'rejected')", name="ck_document_type_decisions_state"),
        CheckConstraint("source IN ('manual', 'automatic')", name="ck_document_type_decisions_source"),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_document_type_decisions_confidence"),
        CheckConstraint("confidence_band IS NULL OR confidence_band IN ('low', 'medium', 'high')", name="ck_document_type_decisions_confidence_band"),
        CheckConstraint("revision > 0", name="ck_document_type_decisions_revision"),
        CheckConstraint("jsonb_typeof(signals) = 'object' AND pg_column_size(signals) <= 16384", name="ck_document_type_decisions_signals"),
        CheckConstraint(
            "source <> 'automatic' OR (confidence IS NOT NULL AND confidence_band IS NOT NULL AND rationale IS NOT NULL AND classifier_version IS NOT NULL AND policy_version IS NOT NULL AND classified_document_version_id IS NOT NULL)",
            name="ck_document_type_decisions_automatic_provenance",
        ),
        CheckConstraint(
            "source <> 'manual' OR (confidence IS NULL AND confidence_band IS NULL AND classifier_version IS NULL AND policy_version IS NULL AND classified_document_version_id IS NULL AND signals = '{}'::jsonb)",
            name="ck_document_type_decisions_manual_provenance",
        ),
        Index("ix_document_type_decisions_type_state", "document_type_id", "state"),
        Index("ix_document_type_decisions_document_state", "document_id", "state"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    document_type_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("document_types.id", ondelete="CASCADE"), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_band: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rationale: Mapped[str | None] = mapped_column(String(2_000), nullable=True)
    classifier_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    policy_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    signals: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb"))
    classified_document_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=sa_text("1"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="document_type_decisions")
    document_type: Mapped[DocumentType] = relationship("DocumentType", back_populates="decisions")


class ContentGroup(Base):
    __tablename__ = "content_groups"
    __table_args__ = (
        UniqueConstraint("normalized_name", name="uq_content_groups_normalized_name"),
        Index("ix_content_groups_archived_name", "archived_at", "normalized_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2_000), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb"))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    aliases: Mapped[list[ContentGroupAlias]] = relationship(
        "ContentGroupAlias", back_populates="content_group", cascade="all, delete-orphan", passive_deletes=True, order_by="ContentGroupAlias.normalized_name"
    )
    document_assignments: Mapped[list[DocumentContentGroupAssignment]] = relationship(
        "DocumentContentGroupAssignment", back_populates="content_group"
    )


class ContentGroupAlias(Base):
    __tablename__ = "content_group_aliases"
    __table_args__ = (
        UniqueConstraint("normalized_name", name="uq_content_group_aliases_normalized_name"),
        Index("ix_content_group_aliases_group", "content_group_id", "archived_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_group_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("content_groups.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(80), nullable=False)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    content_group: Mapped[ContentGroup] = relationship("ContentGroup", back_populates="aliases")


class DocumentContentGroupAssignment(Base):
    __tablename__ = "document_content_group_assignments"
    __table_args__ = (
        CheckConstraint("state IN ('pending', 'unresolved', 'suggested', 'assigned')", name="ck_document_content_group_assignments_state"),
        CheckConstraint("source IN ('manual', 'automatic')", name="ck_document_content_group_assignments_source"),
        CheckConstraint(
            "((state IN ('pending', 'unresolved') AND content_group_id IS NULL) OR (state IN ('suggested', 'assigned') AND content_group_id IS NOT NULL))",
            name="ck_document_content_group_assignments_state_group",
        ),
        CheckConstraint(
            "((state = 'unresolved' AND unresolved_reason IS NOT NULL) OR (state <> 'unresolved' AND unresolved_reason IS NULL))",
            name="ck_document_content_group_assignments_unresolved_reason",
        ),
        CheckConstraint("confidence IS NULL OR (confidence >= 0 AND confidence <= 1)", name="ck_document_content_group_assignments_confidence"),
        CheckConstraint("confidence_band IS NULL OR confidence_band IN ('low', 'medium', 'high')", name="ck_document_content_group_assignments_confidence_band"),
        CheckConstraint("revision > 0", name="ck_document_content_group_assignments_revision"),
        CheckConstraint("jsonb_typeof(signals) = 'object' AND pg_column_size(signals) <= 16384", name="ck_document_content_group_assignments_signals"),
        CheckConstraint(
            "source <> 'manual' OR (state = 'assigned' AND content_group_id IS NOT NULL AND confidence IS NULL AND confidence_band IS NULL AND classifier_version IS NULL AND policy_version IS NULL AND summary_hash IS NULL AND classified_document_version_id IS NULL AND signals = '{}'::jsonb)",
            name="ck_document_content_group_assignments_manual",
        ),
        CheckConstraint(
            "source <> 'automatic' OR state NOT IN ('suggested', 'assigned') OR (confidence IS NOT NULL AND confidence_band IS NOT NULL AND rationale IS NOT NULL AND classifier_version IS NOT NULL AND policy_version IS NOT NULL AND summary_hash IS NOT NULL AND classified_document_version_id IS NOT NULL)",
            name="ck_document_content_group_assign_auto_resolved_provenance",
        ),
        Index("ix_document_content_group_assignments_group_state", "content_group_id", "state"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    content_group_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("content_groups.id", ondelete="RESTRICT"), nullable=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    unresolved_reason: Mapped[str | None] = mapped_column(String(1_000), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_band: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rationale: Mapped[str | None] = mapped_column(String(2_000), nullable=True)
    classifier_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    policy_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    signals: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb"))
    summary_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    classified_document_version_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default=sa_text("1"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="content_group_assignment")
    content_group: Mapped[ContentGroup | None] = relationship("ContentGroup", back_populates="document_assignments")
