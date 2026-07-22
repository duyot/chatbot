"""document_chunks citation bbox

Adds document_chunks.bbox (JSONB): a list of normalized rects
[[x0,y0,x1,y1], ...] in [0,1] of the page, covering the source line(s) each
child chunk was derived from. Nullable so existing rows survive; rerun
scripts.reingest_all to populate geometry on previously-ingested documents.

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-22
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("document_chunks", sa.Column("bbox", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("document_chunks", "bbox")
