import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db, SessionLocal
from ..models import Document
from ..schemas import ChatRequest
from ..services.rag import agentic_rag_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == request.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "done":
        raise HTTPException(status_code=400, detail="Document not ready for querying")

    logger.info("chat_stream: start document_id=%s query=%.120s", request.document_id, request.message)

    async def event_stream():
        stream_db = SessionLocal()
        try:
            async for event in agentic_rag_stream(request.document_id, request.message, stream_db):
                yield f"data: {json.dumps(event)}\n\n"
            logger.info("chat_stream: done document_id=%s", request.document_id)
        except Exception as exc:
            logger.exception("chat_stream: error document_id=%s", request.document_id)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            stream_db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
