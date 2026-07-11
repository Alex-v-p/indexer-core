from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.indexer_infrastructure.postgres.base import Base

if TYPE_CHECKING:
    from packages.indexer_infrastructure.postgres.models.document_versions import DocumentVersion
    from packages.indexer_infrastructure.postgres.models.documents import Document
    from packages.indexer_infrastructure.postgres.models.evidence import Evidence
    from packages.indexer_infrastructure.postgres.models.qdrant_chunk_indexes import QdrantChunkIndex
    from packages.indexer_infrastructure.postgres.models.query_runs import QueryRun


class Citation(Base):
    """Answer citation pointing back to evidence and source location metadata."""

    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("query_run_id", "citation_index", name="uq_citations_query_run_citation_index"),
        Index("ix_citations_query_run_id", "query_run_id"),
        Index("ix_citations_evidence_id", "evidence_id"),
        Index("ix_citations_qdrant_chunk_index_id", "qdrant_chunk_index_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    evidence_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evidence.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    qdrant_chunk_index_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("qdrant_chunk_indexes.id", ondelete="SET NULL"),
        nullable=True,
    )
    citation_index: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    query_run: Mapped[QueryRun] = relationship("QueryRun", back_populates="citations")
    evidence: Mapped[Evidence | None] = relationship("Evidence", back_populates="citations")
    document: Mapped[Document | None] = relationship("Document", back_populates="citations")
    document_version: Mapped[DocumentVersion | None] = relationship("DocumentVersion", back_populates="citations")
    qdrant_chunk_index: Mapped[QdrantChunkIndex | None] = relationship(
        "QdrantChunkIndex",
        back_populates="citations",
    )
