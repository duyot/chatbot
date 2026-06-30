import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String, nullable=False, default="pending")
    error_msg = Column(Text)
    # Document-level metadata captured during ingestion.
    mime_type = Column(String, nullable=True)
    page_count = Column(Integer, nullable=True)
    # Free-form doc metadata: ocr_engine, ocr_pages, native_pages, languages, etc.
    doc_metadata = Column(JSONB, nullable=True)


class DocumentParentChunk(Base):
    __tablename__ = "document_parent_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    # Page span this parent was derived from (1-based). source in {native, ocr}.
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    source = Column(String, nullable=True)

    __table_args__ = (
        UniqueConstraint("document_id", "parent_index", name="uq_dpc_doc_idx"),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_parent_chunks.id", ondelete="CASCADE"),
        nullable=True,  # nullable until reingest completes
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
    # Retrieval metadata: page (1-based), source in {native, ocr}, OCR confidence (avg).
    page = Column(Integer, nullable=True)
    source = Column(String, nullable=True)
    ocr_confidence = Column(Float, nullable=True)

    __table_args__ = (
        Index("ix_document_chunks_doc_page", "document_id", "page"),
    )
