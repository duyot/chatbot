"""One-shot reingest: for every Document with status='done' (or 'failed'),
delete its existing chunks and re-run the ingestion pipeline using the
new parent-child chunker. Per-document transaction; on crash the document
is left in status='failed' for the user to retry from the UI.

Usage:
  python -m scripts.reingest_all                  # all done docs
  python -m scripts.reingest_all --include-failed # also retry previously failed
  python -m scripts.reingest_all --doc-id <uuid>  # single doc
"""
from __future__ import annotations
import argparse
import logging
from uuid import UUID

from sqlalchemy import select, delete

from app.database import SessionLocal
from app.models import Document, DocumentChunk, DocumentParentChunk
from app.services.ingestion import (
    parse_file,
    chunk_text,
    embed_chunks,
    store_chunks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def reingest_one(doc_id: UUID) -> None:
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            logger.error("doc not found: %s", doc_id)
            return
        logger.info("reingest start: %s (%s)", doc_id, doc.file_name)

        # Delete in dependency order; cascades handle children rows
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))
        db.execute(delete(DocumentParentChunk).where(DocumentParentChunk.document_id == doc_id))
        db.commit()

        text = parse_file(doc.file_path, doc.file_name)
        parents, children_per_parent = chunk_text(text)
        flat_children = [c for sub in children_per_parent for c in sub]
        embeddings = embed_chunks(flat_children)
        store_chunks(db, str(doc_id), parents, children_per_parent, embeddings)

        doc.status = "done"
        doc.error_msg = None
        db.commit()
        logger.info("reingest done: %s", doc_id)
    except Exception as e:
        db.rollback()
        doc = db.get(Document, doc_id)
        if doc:
            doc.status = "failed"
            doc.error_msg = f"reingest: {e}"
            db.commit()
        logger.exception("reingest failed: %s", doc_id)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", help="UUID of a single document")
    parser.add_argument("--include-failed", action="store_true",
                        help="Also reingest documents currently in status='failed'")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.doc_id:
            ids = [UUID(args.doc_id)]
        else:
            statuses = ["done"] + (["failed"] if args.include_failed else [])
            rows = db.execute(
                select(Document.id).where(Document.status.in_(statuses))
            ).scalars().all()
            ids = list(rows)
    finally:
        db.close()

    logger.info("reingesting %d documents", len(ids))
    for doc_id in ids:
        reingest_one(doc_id)
    logger.info("all done")


if __name__ == "__main__":
    main()
