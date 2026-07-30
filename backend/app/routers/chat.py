import json
import logging
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db, SessionLocal
from ..models import Conversation, Document, Message, User
from ..observability import bind_trace, emit
from ..schemas import ChatRequest
from ..security import get_current_user
from ..services.rag import agentic_rag_stream

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _make_title(message: str, limit: int = 60) -> str:
    """Derive a conversation title from the first user message."""
    title = " ".join(message.split())
    if len(title) > limit:
        title = title[:limit].rstrip() + "…"
    return title or "New chat"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    doc = db.query(Document).filter(Document.id == request.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "done":
        raise HTTPException(status_code=400, detail="Document not ready for querying")

    # Capture primitives before the generator closure (the request-scoped `db`
    # and `current_user` must not be used inside the long-lived stream).
    user_id = current_user.id
    document_uuid = uuid.UUID(request.document_id)
    message = request.message
    requested_conversation_id = request.conversation_id

    # One trace id per chat request, carried by every log record and trace event
    # produced downstream (rewrite -> retrieve -> rerank -> generate). Bound here
    # rather than in a middleware because only the AI path needs correlating.
    trace_id = bind_trace()
    logger.info(
        "chat_stream: start document_id=%s query=%.120s", request.document_id, message
    )
    emit(
        "chat.request",
        document_id=request.document_id,
        conversation_id=requested_conversation_id,
        user_id=str(user_id),
        question_chars=len(message),
    )

    async def event_stream():
        # The generator body runs after the request handler returns, in a fresh
        # context that did not inherit the binding above.
        bind_trace(trace_id)
        stream_db = SessionLocal()
        try:
            # Resolve an existing conversation (must belong to this user) or
            # create a new one for this chat.
            if requested_conversation_id:
                try:
                    conv_uuid = uuid.UUID(requested_conversation_id)
                except (ValueError, TypeError):
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid conversation id'})}\n\n"
                    return
                conv = (
                    stream_db.query(Conversation)
                    .filter(
                        Conversation.id == conv_uuid,
                        Conversation.user_id == user_id,
                    )
                    .first()
                )
                if conv is None:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Conversation not found'})}\n\n"
                    return
            else:
                conv = Conversation(user_id=user_id, title=_make_title(message))
                stream_db.add(conv)
                stream_db.commit()
                stream_db.refresh(conv)

            yield f"data: {json.dumps({'type': 'conversation', 'conversation_id': str(conv.id), 'title': conv.title})}\n\n"

            # Persist the user's prompt up front.
            stream_db.add(
                Message(
                    conversation_id=conv.id,
                    document_id=document_uuid,
                    role="user",
                    content=message,
                )
            )
            stream_db.commit()

            answer_parts: list[str] = []
            citations = None
            try:
                async for event in agentic_rag_stream(
                    request.document_id, message, stream_db
                ):
                    if event.get("type") == "token":
                        answer_parts.append(event.get("content", ""))
                    elif event.get("type") == "citations":
                        citations = event.get("chunks")
                    yield f"data: {json.dumps(event)}\n\n"
                logger.info("chat_stream: done document_id=%s", request.document_id)
            except Exception as exc:
                logger.exception("chat_stream: error document_id=%s", request.document_id)
                yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
                return

            # Persist the assistant answer only on clean completion.
            answer = "".join(answer_parts)
            if answer:
                stream_db.add(
                    Message(
                        conversation_id=conv.id,
                        document_id=document_uuid,
                        role="assistant",
                        content=answer,
                        citations=citations,
                    )
                )
                conv.updated_at = datetime.now(timezone.utc)
                stream_db.commit()
        finally:
            stream_db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
