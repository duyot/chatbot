"""index search_text for the ts_rank fallback

Migration 0003 indexes to_tsvector('english', content). The ts_rank fallback
(app/services/rag/retrieval.py:_keyword_search_tsrank) queries
to_tsvector('english', search_text) instead, so it exercises a completely
different expression than the one 0003 indexed. Postgres cannot use a GIN
index built over `content` to serve a query over `search_text` — without
this migration, the ts_rank fallback path sequential-scans document_chunks
on every keyword search.

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-28
"""
from typing import Sequence, Union
from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_search_text_fts "
        "ON document_chunks USING GIN (to_tsvector('english', search_text))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_search_text_fts")
