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


class Subject(Base):
    __tablename__ = "subjects"
    __table_args__ = (
        UniqueConstraint(
            "kind",
            "normalized_name",
            name="uq_subjects_kind_normalized_name",
        ),
        CheckConstraint(
            "kind IN ('project', 'topic', 'organization', 'custom')",
            name="ck_subjects_kind",
        ),
        Index("ix_subjects_kind_archived_at", "kind", "archived_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2_000), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    aliases: Mapped[list[SubjectAlias]] = relationship(
        "SubjectAlias",
        back_populates="subject",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="SubjectAlias.normalized_name",
    )
    document_decisions: Mapped[list[DocumentSubjectDecision]] = relationship(
        "DocumentSubjectDecision",
        back_populates="subject",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class SubjectAlias(Base):
    __tablename__ = "subject_aliases"
    __table_args__ = (
        UniqueConstraint(
            "subject_id",
            "normalized_name",
            name="uq_subject_aliases_subject_normalized_name",
        ),
        Index("ix_subject_aliases_normalized_name", "normalized_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(255), nullable=False)
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    subject: Mapped[Subject] = relationship("Subject", back_populates="aliases")


class DocumentSubjectDecision(Base):
    __tablename__ = "document_subject_decisions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "subject_id",
            name="uq_document_subject_decisions_document_subject",
        ),
        CheckConstraint(
            "state IN ('suggested', 'assigned', 'rejected')",
            name="ck_document_subject_decisions_state",
        ),
        CheckConstraint(
            "control_source IN ('manual', 'automatic')",
            name="ck_document_subject_decisions_control_source",
        ),
        CheckConstraint(
            "confidence_band IS NULL OR confidence_band IN ('low', 'medium', 'high')",
            name="ck_document_subject_decisions_confidence_band",
        ),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)",
            name="ck_document_subject_decisions_confidence",
        ),
        CheckConstraint(
            "revision > 0",
            name="ck_document_subject_decisions_revision",
        ),
        CheckConstraint(
            "jsonb_typeof(signals) = 'object' AND pg_column_size(signals) <= 16384",
            name="ck_document_subject_decisions_signals",
        ),
        CheckConstraint(
            "control_source <> 'automatic' OR "
            "(confidence IS NOT NULL AND confidence_band IS NOT NULL AND "
            "rationale IS NOT NULL AND classifier_version IS NOT NULL AND "
            "policy_version IS NOT NULL AND classified_document_version_id IS NOT NULL)",
            name="ck_document_subject_decisions_automatic_provenance",
        ),
        CheckConstraint(
            "control_source <> 'manual' OR "
            "(confidence IS NULL AND confidence_band IS NULL AND "
            "classifier_version IS NULL AND policy_version IS NULL AND "
            "classified_document_version_id IS NULL AND signals = '{}'::jsonb)",
            name="ck_document_subject_decisions_manual_provenance",
        ),
        Index(
            "ix_document_subject_decisions_subject_state",
            "subject_id",
            "state",
        ),
        Index(
            "ix_document_subject_decisions_document_state",
            "document_id",
            "state",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    subject_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("subjects.id", ondelete="CASCADE"),
        nullable=False,
    )
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    control_source: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_band: Mapped[str | None] = mapped_column(String(16), nullable=True)
    rationale: Mapped[str | None] = mapped_column(String(2_000), nullable=True)
    classifier_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    policy_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    signals: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    # This intentionally remains an audit UUID rather than a foreign key: a
    # document version can be deleted without erasing which version was classified.
    classified_document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    revision: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default=sa_text("1"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    document: Mapped[Document] = relationship(
        "Document",
        back_populates="subject_decisions",
    )
    subject: Mapped[Subject] = relationship(
        "Subject",
        back_populates="document_decisions",
    )
