"""document + chunk OCR/page metadata

Adds OCR/page metadata captured during ingestion:
  documents:             mime_type, page_count, doc_metadata (JSONB)
  document_parent_chunks: page_start, page_end, source
  document_chunks:        page, source, ocr_confidence (+ (document_id, page) index)

All columns are nullable so existing rows survive; rerun scripts.reingest_all
to populate metadata on previously-ingested documents.

Revision ID: 0006
Revises: 0005
Create Date: 2026-06-30
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("mime_type", sa.String(), nullable=True))
    op.add_column("documents", sa.Column("page_count", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("doc_metadata", JSONB(), nullable=True))

    op.add_column("document_parent_chunks", sa.Column("page_start", sa.Integer(), nullable=True))
    op.add_column("document_parent_chunks", sa.Column("page_end", sa.Integer(), nullable=True))
    op.add_column("document_parent_chunks", sa.Column("source", sa.String(), nullable=True))

    op.add_column("document_chunks", sa.Column("page", sa.Integer(), nullable=True))
    op.add_column("document_chunks", sa.Column("source", sa.String(), nullable=True))
    op.add_column("document_chunks", sa.Column("ocr_confidence", sa.Float(), nullable=True))
    op.create_index(
        "ix_document_chunks_doc_page",
        "document_chunks",
        ["document_id", "page"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_doc_page", table_name="document_chunks")
    op.drop_column("document_chunks", "ocr_confidence")
    op.drop_column("document_chunks", "source")
    op.drop_column("document_chunks", "page")

    op.drop_column("document_parent_chunks", "source")
    op.drop_column("document_parent_chunks", "page_end")
    op.drop_column("document_parent_chunks", "page_start")

    op.drop_column("documents", "doc_metadata")
    op.drop_column("documents", "page_count")
    op.drop_column("documents", "mime_type")
