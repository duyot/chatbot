"""embedding dim 2560 -> 1536 for openai text-embedding-3-small

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-28
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing embeddings are 2560-dim Ollama vectors; they cannot be reused
    # with text-embedding-3-small (1536). Drop the column and re-create it.
    # Chunk rows survive (content/parent_id/etc.); rerun scripts.reingest_all
    # to refill embeddings.
    op.drop_column("document_chunks", "embedding")
    op.add_column(
        "document_chunks",
        sa.Column("embedding", Vector(1536), nullable=True),
    )
    # HNSW index is now possible (cap is 2000 dims; 1536 < 2000).
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
        "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_document_chunks_embedding_hnsw")
    op.drop_column("document_chunks", "embedding")
    op.add_column(
        "document_chunks",
        sa.Column("embedding", Vector(2560), nullable=True),
    )
