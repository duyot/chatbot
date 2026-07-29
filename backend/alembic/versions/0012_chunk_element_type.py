"""add element_type to document_chunks

Distinguishes table-derived chunks from prose so the eval harness can score
table questions separately and the UI can render markdown tables.

No backfill: layout-aware chunking changes every chunk boundary, so
`python -m scripts.reingest_all` is required after this migration regardless.
Existing rows keep element_type NULL, which reads as "unknown, pre-0012".

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-29
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("element_type", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "element_type")
