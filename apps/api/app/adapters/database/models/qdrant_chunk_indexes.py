from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.adapters.database.base import Base

if TYPE_CHECKING:
    from app.adapters.database.models.citations import Citation
    from app.adapters.database.models.document_versions import DocumentVersion
    from app.adapters.database.models.documents import Document
    from app.adapters.database.models.evidence import Evidence


class QdrantChunkIndex(Base):
    """Postgres registry row for a chunk indexed in Qdrant.

    This model deliberately does not store chunk text or vectors. It stores the
    stable application ID and the Qdrant point reference needed for deletes,
    re-indexing, provenance, and traceability.
    """

    __tablename__ = "qdrant_chunk_indexes"
    __table_args__ = (
        UniqueConstraint("document_version_id", "ordinal", name="uq_qdrant_chunk_indexes_document_version_ordinal"),
        UniqueConstraint("qdrant_collection", "qdrant_point_id", name="uq_qdrant_chunk_indexes_qdrant_point"),
        Index("ix_qdrant_chunk_indexes_document_id", "document_id"),
        Index("ix_qdrant_chunk_indexes_document_version_id", "document_version_id"),
        Index("ix_qdrant_chunk_indexes_content_hash", "content_hash"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    section_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    qdrant_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    qdrant_point_id: Mapped[str] = mapped_column(String(255), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    document: Mapped[Document] = relationship("Document", back_populates="qdrant_chunk_indexes")
    document_version: Mapped[DocumentVersion] = relationship("DocumentVersion", back_populates="qdrant_chunk_indexes")
    evidence_items: Mapped[list[Evidence]] = relationship("Evidence", back_populates="qdrant_chunk_index")
    citations: Mapped[list[Citation]] = relationship("Citation", back_populates="qdrant_chunk_index")
