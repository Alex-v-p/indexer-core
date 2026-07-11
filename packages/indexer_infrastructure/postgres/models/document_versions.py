from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, Integer, String, UniqueConstraint, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.indexer_application.dto import DocumentVersionStatus

from packages.indexer_infrastructure.postgres.base import Base
from packages.indexer_infrastructure.postgres.models.shared import enum_values

if TYPE_CHECKING:
    from packages.indexer_infrastructure.postgres.models.citations import Citation
    from packages.indexer_infrastructure.postgres.models.documents import Document
    from packages.indexer_infrastructure.postgres.models.evidence import Evidence
    from packages.indexer_infrastructure.postgres.models.qdrant_chunk_indexes import QdrantChunkIndex


class DocumentVersion(Base):
    """Versioned parser output for a document.

    A new version can be created when a source document is replaced, re-parsed,
    re-chunked, or indexed with different embedding settings.
    """

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_number", name="uq_document_versions_document_version_number"),
        Index("ix_document_versions_document_id_status", "document_id", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    storage_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[DocumentVersionStatus] = mapped_column(
        Enum(DocumentVersionStatus, name="document_version_status", values_callable=enum_values),
        nullable=False,
        default=DocumentVersionStatus.PENDING,
        server_default=DocumentVersionStatus.PENDING.value,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    document: Mapped[Document] = relationship("Document", back_populates="versions")
    qdrant_chunk_indexes: Mapped[list[QdrantChunkIndex]] = relationship(
        "QdrantChunkIndex",
        back_populates="document_version",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="QdrantChunkIndex.ordinal",
    )
    evidence_items: Mapped[list[Evidence]] = relationship("Evidence", back_populates="document_version")
    citations: Mapped[list[Citation]] = relationship("Citation", back_populates="document_version")
