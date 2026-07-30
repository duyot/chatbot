import logging
import os
import uuid
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
import json
import time

from ..database import get_db, SessionLocal
from ..models import Document, User
from ..schemas import DocumentResponse, DocumentListItem, DocumentPageItem
from ..security import get_current_user
from ..config import settings
from ..services import page_images

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.get("", response_model=list[DocumentListItem])
def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return db.query(Document).filter(Document.status == "done").all()

EXTENSION_MAP = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if file.content_type not in EXTENSION_MAP:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_mb}MB limit")

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    ext = EXTENSION_MAP[file.content_type]
    file_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")

    try:
        with open(file_path, "wb") as f:
            f.write(content)
        doc = Document(file_name=file.filename, file_path=file_path)
        db.add(doc)
        db.commit()
        db.refresh(doc)
    except Exception:
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise

    logger.info("upload_document: queued document_id=%s file=%s size=%d", doc.id, file.filename, len(content))
    from ..workers.tasks import ingest_document
    ingest_document.delay(str(doc.id))

    return doc


def _get_document_or_404(db: Session, document_id: str) -> Document:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.get("/{document_id}/pages", response_model=list[DocumentPageItem])
def list_document_pages(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Manifest of rendered preview pages, in order.

    Normally a directory scan. Documents ingested before page rendering existed
    — and any whose ingestion-time render failed — are rendered here on first
    preview, so no backfill step is required for previews to work. Returns []
    for non-PDFs, which the UI previews by other means.
    """
    doc = _get_document_or_404(db, document_id)

    pages = page_images.list_page_images(doc.id)
    if not pages:
        pages = page_images.render_document_pages(doc.file_path, doc.id)

    return [
        DocumentPageItem(page=p.page, width=p.width, height=p.height)
        for p in pages
    ]


@router.get("/{document_id}/pages/{page}")
def get_document_page_image(
    document_id: str,
    page: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """One rendered page image. `page` is 1-based and typed int, so the path is
    built from a validated integer and never from caller-supplied text."""
    doc = _get_document_or_404(db, document_id)
    if page < 1:
        raise HTTPException(status_code=404, detail="Page not found")

    path = page_images.page_image_path(doc.id, page)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Page not found")

    return FileResponse(
        path,
        media_type=page_images.media_type(),
        # Content at a given path never changes (a re-render replaces the whole
        # file), so this is safe to hold for a day. `private` keeps it out of
        # shared caches since the route is authenticated.
        headers={"Cache-Control": "private, max-age=86400"},
    )


@router.get("/{document_id}/file")
def get_document_file(
    document_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = _get_document_or_404(db, document_id)
    if not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="Document not found")

    return FileResponse(
        doc.file_path,
        media_type=doc.mime_type or "application/octet-stream",
        filename=doc.file_name,
        content_disposition_type="inline",
    )


@router.get("/{document_id}/status")
def stream_status(document_id: str, db: Session = Depends(get_db)):
    exists = db.query(Document).filter(Document.id == document_id).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Document not found")

    logger.info("stream_status: opened document_id=%s", document_id)

    def event_stream():
        while True:
            inner_db = SessionLocal()
            try:
                doc = inner_db.query(Document).filter(Document.id == document_id).first()
                status = doc.status if doc else "failed"
                error_msg = doc.error_msg if doc else None
            finally:
                inner_db.close()

            if status == "done":
                logger.info("stream_status: done document_id=%s", document_id)
                yield f"data: {json.dumps({'status': 'done', 'message': 'Document ready for Q&A.'})}\n\n"
                break
            elif status == "failed":
                logger.warning("stream_status: failed document_id=%s error=%s", document_id, error_msg)
                yield f"data: {json.dumps({'status': 'failed', 'message': error_msg or 'Ingestion failed.'})}\n\n"
                break
            else:
                yield f"data: {json.dumps({'status': status, 'message': 'Ingesting document...'})}\n\n"
            time.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
