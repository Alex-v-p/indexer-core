from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, DateTime, Enum, String, func, text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from packages.indexer_application.dto import DocumentStatus

from packages.indexer_infrastructure.postgres.base import Base
from packages.indexer_infrastructure.postgres.models.shared import enum_values

if TYPE_CHECKING:
    from packages.indexer_infrastructure.postgres.models.citations import Citation
    from packages.indexer_infrastructure.postgres.models.document_versions import DocumentVersion
    from packages.indexer_infrastructure.postgres.models.evidence import Evidence
    from packages.indexer_infrastructure.postgres.models.qdrant_chunk_indexes import QdrantChunkIndex


class Document(Base):
    """Top-level source document tracked by the application database.

    Postgres is the source of truth for document ownership, lifecycle, storage,
    and version history. Qdrant is only used as the vector index for searchable
    chunk embeddings.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    storage_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    checksum_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(DocumentStatus, name="document_status", values_callable=enum_values),
        nullable=False,
        default=DocumentStatus.UPLOADED,
        server_default=DocumentStatus.UPLOADED.value,
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

    versions: Mapped[list[DocumentVersion]] = relationship(
        "DocumentVersion",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="DocumentVersion.version_number",
    )
    qdrant_chunk_indexes: Mapped[list[QdrantChunkIndex]] = relationship(
        "QdrantChunkIndex",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="QdrantChunkIndex.ordinal",
    )
    evidence_items: Mapped[list[Evidence]] = relationship("Evidence", back_populates="document")
    citations: Mapped[list[Citation]] = relationship("Citation", back_populates="document")
