import logging
import time

from celery.exceptions import MaxRetriesExceededError, Retry

from .celery_app import celery_app
from ..config import settings
from ..database import SessionLocal
from ..models import Document
from ..observability import bind_trace, emit, timed
from ..services.contextualizer import contextualize_with_stats
from ..services.page_images import render_document_pages
from ..services.ingestion import (
    parse_document,
    chunk_document,
    embed_chunks,
    store_chunks,
    build_embedding_input,
)
from ..services.ocr_client import ParseTimeout

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=1, default_retry_delay=10)
def ingest_document(self, document_id: str):
    # Reuse the Celery task id as the trace id: the two already appear together
    # in worker.log, and a retry gets the same id, which is what you want when
    # correlating a second attempt against the first.
    bind_trace(self.request.id)
    logger.info("[task:%s] ingest_document started document_id=%s", self.request.id, document_id)
    emit("ingest.start", document_id=document_id, task_id=str(self.request.id))
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 1)

    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == document_id).first()
        doc.status = "processing"
        db.commit()

        parsed = parse_document(doc.file_path, doc.file_name)
        logger.info(
            "[task:%s] parse complete file=%s pages=%d text_len=%d",
            self.request.id, doc.file_name, len(parsed.pages), len(parsed.text),
        )

        # Preview page images. Deliberately non-fatal: previews are cosmetic and
        # must never keep a document out of Q&A. Anything missed here is
        # rendered lazily on first preview by GET /{id}/pages.
        try:
            images = render_document_pages(doc.file_path, document_id)
            logger.info(
                "[task:%s] page images rendered count=%d document_id=%s",
                self.request.id, len(images), document_id,
            )
        except Exception:
            logger.warning(
                "[task:%s] page image render failed document_id=%s",
                self.request.id, document_id, exc_info=True,
            )

        parents, children_per_parent = chunk_document(parsed)
        if not parents:
            raise ValueError("No extractable text found in document (empty or OCR failed)")

        ctx_stats = None
        if settings.contextual_embeddings_enabled:
            contexts, ctx_stats = contextualize_with_stats(parsed, children_per_parent)
            for children, child_contexts in zip(children_per_parent, contexts):
                for child, ctx in zip(children, child_contexts):
                    child.context = ctx
            logger.info(
                "[task:%s] contextualized %d/%d children tier=%s",
                self.request.id, ctx_stats["contextualized_children"],
                ctx_stats["total_children"], ctx_stats["tier"],
            )
        else:
            logger.info("[task:%s] contextual embeddings disabled, skipping", self.request.id)

        flat_children = [
            build_embedding_input(c.context, c.content)
            for sub in children_per_parent for c in sub
        ]
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
        doc.mime_type = parsed.metadata.get("mime_type")
        doc.page_count = parsed.metadata.get("page_count")
        metadata = dict(parsed.metadata)
        if ctx_stats is not None:
            metadata["contextualization"] = ctx_stats
        doc.doc_metadata = metadata
        db.commit()
        logger.info("[task:%s] ingest_document completed document_id=%s", self.request.id, document_id)
        emit(
            "ingest.done",
            document_id=document_id,
            ms=elapsed_ms(),
            pages=parsed.metadata.get("page_count"),
            parents=len(parents),
            children=len(flat_children),
            embeddings=len(embeddings),
            contextualization=ctx_stats,
            engine=parsed.metadata.get("engine", "legacy"),
        )
    except Retry:
        raise
    except ParseTimeout as exc:
        # Deliberately not retried: a parse that exceeded parse_timeout_s once
        # will exceed it again, and each attempt costs another full timeout of
        # worker time. Surface it and let the user retry explicitly.
        logger.error(
            "[task:%s] parse timed out, not retrying document_id=%s",
            self.request.id, document_id,
        )
        emit(
            "ingest.failed",
            document_id=document_id,
            reason="parse_timeout",
            retried=False,
            ms=elapsed_ms(),
            error=str(exc),
        )
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            doc.error_msg = str(exc)[:500]
            db.commit()
    except Exception as exc:
        logger.exception("[task:%s] ingest_document failed document_id=%s", self.request.id, document_id)
        emit(
            "ingest.failed",
            document_id=document_id,
            reason="exception",
            retried=True,
            ms=elapsed_ms(),
            error=str(exc),
            error_type=type(exc).__name__,
        )
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
