"""fix embedding dimension 1536 -> 2560 for qwen3-embedding:4b

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-07
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_table("document_chunks")
    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(2560)),
    )
    # hnsw/ivfflat indexes cap at 2000 dims; qwen3-embedding:4b is 2560 — sequential scan only


def downgrade() -> None:
    op.drop_table("document_chunks")
    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536)),
    )
    op.execute("CREATE INDEX ON document_chunks USING hnsw (embedding vector_cosine_ops)")
