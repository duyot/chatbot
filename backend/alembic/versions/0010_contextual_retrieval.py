"""contextual retrieval: chunk context + BM25 search index

Adds document_chunks.context (nullable text, the LLM-generated situating
description) and document_chunks.search_text (STORED generated column
concatenating context and content), then a ParadeDB pg_search BM25 index over
search_text.

Requires the pg_search extension — the paradedb/paradedb image bundles it
alongside pgvector. shared_preload_libraries is NOT needed: pg_search only
requires it on Postgres < 17, and that image ships Postgres 18.

context is nullable so pre-existing rows survive; there is no backfill by
design. Retrieval degrades to content-only search for those rows via the
coalesce in search_text.

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-28
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.add_column("document_chunks", sa.Column("context", sa.Text(), nullable=True))
    op.add_column(
        "document_chunks",
        sa.Column(
            "search_text",
            sa.Text(),
            sa.Computed("coalesce(context, '') || ' ' || content", persisted=True),
            nullable=True,
        ),
    )
    op.execute(
        """
        CREATE INDEX chunks_bm25 ON document_chunks
        USING bm25 (id, search_text) WITH (key_field = 'id')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_bm25")
    op.drop_column("document_chunks", "search_text")
    op.drop_column("document_chunks", "context")
    # pg_search is left installed: other objects may depend on it, and
    # dropping an extension is not this migration's business to undo.
