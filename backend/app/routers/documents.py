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

router = APIRouter(prefix="/api/documents", tags=["documents"])

EXTENSION_MAP = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
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

    from ..workers.tasks import ingest_document
    ingest_document.delay(str(doc.id))

    return doc
