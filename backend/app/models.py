import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, ForeignKey, UniqueConstraint, Index, Computed
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
    # LLM-generated description situating this chunk within its source document.
    # Embedded and indexed alongside content; NULL means "not contextualized"
    # (pre-0010 rows, or contextualization disabled/failed) and is fully supported.
    context = Column(Text, nullable=True)
    # Postgres STORED generated column: what BM25 indexes. Declared here (not
    # only in the migration) because tests build their schema from
    # Base.metadata.create_all(), not Alembic. Read-only — never assign to it.
    search_text = Column(
        Text,
        Computed("coalesce(context, '') || ' ' || content", persisted=True),
        nullable=True,
    )
    embedding = Column(Vector(1536))
    # Retrieval metadata: page (1-based), source in {native, ocr}, OCR confidence (avg).
    page = Column(Integer, nullable=True)
    source = Column(String, nullable=True)
    ocr_confidence = Column(Float, nullable=True)
    # Citation geometry: list of normalized rects [[x0,y0,x1,y1], ...] in [0,1] of the
    # page, covering the source line(s) this chunk was derived from. Null/[] when
    # unmappable (e.g. DOCX, or no layout match).
    bbox = Column(JSONB, nullable=True)

    __table_args__ = (
        Index("ix_document_chunks_doc_page", "document_id", "page"),
    )


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String, nullable=False, unique=True, index=True)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    title = Column(String, nullable=False, default="New chat")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    # Bumped whenever a message is added, so the sidebar can order by recency.
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_conversations_user_id", "user_id"),
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    # The document this turn was asked against; switchable per message.
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )
    role = Column(String, nullable=False)  # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    # For assistant messages: the citation chunks returned to the UI.
    citations = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("ix_messages_conversation_id", "conversation_id"),
    )
