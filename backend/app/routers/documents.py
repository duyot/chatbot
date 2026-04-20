import os
import uuid
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import json
import time

from ..database import get_db
from ..models import Document
from ..schemas import DocumentResponse
from ..config import settings
from ..workers.tasks import ingest_document

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/png",
    "image/jpeg",
    "image/webp",
}

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File exceeds {settings.max_upload_mb}MB limit")

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1]
    file_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")

    with open(file_path, "wb") as f:
        f.write(content)

    doc = Document(file_name=file.filename, file_path=file_path)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    ingest_document.delay(str(doc.id))

    return doc
