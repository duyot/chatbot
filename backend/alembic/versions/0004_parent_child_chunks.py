"""parent-child chunk schema: add document_parent_chunks + parent_id FK

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-16
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_parent_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("parent_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("document_id", "parent_index", name="uq_dpc_doc_idx"),
    )
    op.create_index("ix_dpc_doc", "document_parent_chunks", ["document_id"])

    op.add_column(
        "document_chunks",
        sa.Column("parent_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_dc_parent",
        "document_chunks",
        "document_parent_chunks",
        ["parent_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_dc_parent", "document_chunks", ["parent_id"])


def downgrade() -> None:
    op.drop_index("ix_dc_parent", table_name="document_chunks")
    op.drop_constraint("fk_dc_parent", "document_chunks", type_="foreignkey")
    op.drop_column("document_chunks", "parent_id")

    op.drop_index("ix_dpc_doc", table_name="document_parent_chunks")
    op.drop_table("document_parent_chunks")
