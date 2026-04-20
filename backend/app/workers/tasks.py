from celery.exceptions import MaxRetriesExceededError, Retry

from .celery_app import celery_app
from ..database import SessionLocal
from ..models import Document
from ..services.ingestion import parse_file, chunk_text, embed_chunks, store_chunks


@celery_app.task(bind=True, max_retries=1, default_retry_delay=10)
def ingest_document(self, document_id: str):
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "processing"
        db.commit()

        text = parse_file(doc.file_path, doc.file_name)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)
        store_chunks(db, document_id, chunks, embeddings)

        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "done"
        db.commit()
    except Retry:
        raise
    except Exception as exc:
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
