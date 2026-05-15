import logging

from celery.exceptions import MaxRetriesExceededError, Retry

from .celery_app import celery_app
from ..database import SessionLocal
from ..models import Document
from ..services.ingestion import parse_file, chunk_text, embed_chunks, store_chunks

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=10)
def ingest_document(self, document_id: str):
    logger.info("[task:%s] ingest_document started document_id=%s", self.request.id, document_id)
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "processing"
        db.commit()

        text = parse_file(doc.file_path, doc.file_name)
        logger.info("[task:%s] parse complete file=%s text_len=%d", self.request.id, doc.file_name, len(text))

        parents, children_per_parent = chunk_text(text)
        flat_children = [c for sub in children_per_parent for c in sub]
        logger.info(
            "[task:%s] chunked text parents=%d children=%d",
            self.request.id, len(parents), len(flat_children),
        )

        embeddings = embed_chunks(flat_children)
        logger.info("[task:%s] embeddings done count=%d", self.request.id, len(embeddings))

        store_chunks(db, document_id, parents, children_per_parent, embeddings)
        logger.info("[task:%s] stored chunks document_id=%s", self.request.id, document_id)

        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "done"
        db.commit()
        logger.info("[task:%s] ingest_document completed document_id=%s", self.request.id, document_id)
    except Retry:
        raise
    except Exception as exc:
        logger.exception("[task:%s] ingest_document failed document_id=%s", self.request.id, document_id)
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            doc.error_msg = str(exc)[:500]
            db.commit()
        try:
            raise self.retry(exc=exc)
        except MaxRetriesExceededError:
            pass
    finally:
        db.close()
