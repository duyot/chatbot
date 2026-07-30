"""Render preview page images for existing documents.

Previews self-heal — GET /api/documents/{id}/pages renders anything missing on
first request — so this script is an optimization, not a migration. Run it to
pre-warm previews for documents ingested before page rendering existed, or with
--force after changing page_image_dpi / page_image_format.

Usage:
  python -m scripts.render_pages_all                  # all done docs, skip rendered pages
  python -m scripts.render_pages_all --force           # delete and re-render
  python -m scripts.render_pages_all --doc-id <uuid>   # single doc
"""
from __future__ import annotations
import argparse
import logging
from uuid import UUID

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Document
from app.services.page_images import delete_page_images, render_document_pages

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def render_one(doc_id: UUID, force: bool = False) -> int:
    """Render one document's pages. Returns the page count (0 for non-PDFs and
    on failure). Never raises — one bad document must not stop the sweep."""
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            logger.error("doc not found: %s", doc_id)
            return 0
        if force:
            removed = delete_page_images(doc_id)
            logger.info("removed %d existing images: %s", removed, doc_id)
        images = render_document_pages(doc.file_path, doc_id)
        logger.info("rendered %d pages: %s (%s)", len(images), doc_id, doc.file_name)
        return len(images)
    except Exception:
        logger.exception("render failed: %s", doc_id)
        return 0
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", help="UUID of a single document")
    parser.add_argument("--force", action="store_true",
                        help="Delete existing page images and re-render from scratch")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.doc_id:
            ids = [UUID(args.doc_id)]
        else:
            ids = list(db.execute(
                select(Document.id).where(Document.status == "done")
            ).scalars().all())
    finally:
        db.close()

    logger.info("rendering page images for %d documents", len(ids))
    total = sum(render_one(doc_id, force=args.force) for doc_id in ids)
    logger.info("all done: %d pages across %d documents", total, len(ids))


if __name__ == "__main__":
    main()
