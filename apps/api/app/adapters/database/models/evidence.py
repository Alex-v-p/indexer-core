from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, Text, UniqueConstraint, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.adapters.database.base import Base

if TYPE_CHECKING:
    from app.adapters.database.models.citations import Citation
    from app.adapters.database.models.document_versions import DocumentVersion
    from app.adapters.database.models.documents import Document
    from app.adapters.database.models.qdrant_chunk_indexes import QdrantChunkIndex
    from app.adapters.database.models.query_runs import QueryRun


class Evidence(Base):
    """Retrieved or selected evidence considered during answer generation.

    Evidence stores a text snapshot so a query run remains explainable even if
    the related Qdrant point is later re-indexed or deleted.
    """

    __tablename__ = "evidence"
    __table_args__ = (
        UniqueConstraint("query_run_id", "rank", name="uq_evidence_query_run_rank"),
        Index("ix_evidence_query_run_id", "query_run_id"),
        Index("ix_evidence_qdrant_chunk_index_id", "qdrant_chunk_index_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    qdrant_chunk_index_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("qdrant_chunk_indexes.id", ondelete="SET NULL"),
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
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa_text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())

    query_run: Mapped[QueryRun] = relationship("QueryRun", back_populates="evidence_items")
    qdrant_chunk_index: Mapped[QdrantChunkIndex | None] = relationship(
        "QdrantChunkIndex",
        back_populates="evidence_items",
    )
    document: Mapped[Document | None] = relationship("Document", back_populates="evidence_items")
    document_version: Mapped[DocumentVersion | None] = relationship("DocumentVersion", back_populates="evidence_items")
    citations: Mapped[list[Citation]] = relationship("Citation", back_populates="evidence")
