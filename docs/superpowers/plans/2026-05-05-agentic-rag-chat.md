# Agentic RAG Chat Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "chat with document" feature where a user selects an ingested document, types a question, and receives a real streamed answer from a ReAct agent backed by pgvector and qwen3.

**Architecture:** A `DocumentSelector` dropdown populates from `GET /api/documents`. On submit, `useChat` POSTs to `POST /api/chat/stream`, which runs an agentic loop: qwen3 calls a `search_document` tool (pgvector cosine search) up to 3 times, then streams the final generation. SSE events carry tokens, citations, and a done signal.

**Tech Stack:** FastAPI, SQLAlchemy + pgvector, `langchain-ollama`, `langchain-core`, Ollama (qwen3 + qwen3-embedding:4b), React 19, Vite.

---

## File Map

**Backend — new files:**
- `backend/app/services/rag.py` — `make_search_tool`, `agentic_rag_stream` async generator
- `backend/app/routers/chat.py` — `POST /api/chat/stream` SSE endpoint
- `backend/tests/test_rag.py` — unit tests for the RAG service
- `backend/tests/test_chat.py` — pre-flight validation tests for the chat endpoint

**Backend — modified files:**
- `backend/requirements.txt` — add `langchain-ollama`, `langchain-core`
- `backend/app/config.py` — add `ollama_chat_model: str = "qwen3"`
- `backend/app/schemas.py` — add `DocumentListItem`, `ChatRequest`
- `backend/app/services/ingestion.py` — add `embed_text` single-string helper (additive, no change to `embed_chunks`)
- `backend/app/routers/documents.py` — add `GET /api/documents` (status=done filter)
- `backend/app/main.py` — register chat router

**Frontend — new files:**
- `src/components/DocumentSelector.jsx`
- `src/components/DocumentSelector.css`
- `src/hooks/useChat.js`

**Frontend — modified files:**
- `src/components/ChatMessage.jsx` — streaming cursor + citations collapsible
- `src/components/ChatThread.jsx` — pass new message props; suppress typing indicator during streaming
- `src/pages/ChatPage.jsx` — remove canned replies, add `DocumentSelector`, wire `useChat`

---

## Task 1: Add dependencies and config

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`

- [ ] **Step 1: Add Python packages to requirements.txt**

Open `backend/requirements.txt` and append these two lines at the end:

```
langchain-ollama==0.3.3
langchain-core==0.3.56
```

- [ ] **Step 2: Add the chat model config key**

In `backend/app/config.py`, add one field to the `Settings` class after `ollama_embedding_model`:

```python
ollama_chat_model: str = "qwen3"
```

The full `Settings` class becomes:
```python
class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str = ""
    ollama_base_url: str = "http://172.22.6.30:30002"
    ollama_embedding_model: str = "qwen3-embedding:4b"
    ollama_chat_model: str = "qwen3"
    upload_dir: str = "./uploads"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env")
```

- [ ] **Step 3: Install the new packages inside the Docker container or venv**

If running locally with a venv:
```bash
cd backend && pip install langchain-ollama==0.3.3 langchain-core==0.3.56
```

If using Docker Compose, rebuild the backend image:
```bash
docker compose build backend
```

- [ ] **Step 4: Verify the import works**

```bash
cd backend && python -c "from langchain_ollama import ChatOllama; print('ok')"
```

Expected output: `ok`

- [ ] **Step 5: Commit**

```bash
git add backend/requirements.txt backend/app/config.py
git commit -m "chore: add langchain-ollama dependency and ollama_chat_model config"
```

---

## Task 2: Add schemas and extract embed_text helper

**Files:**
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/services/ingestion.py`
- Test: `backend/tests/test_ingestion.py`

- [ ] **Step 1: Write the failing test for embed_text**

Add this test to `backend/tests/test_ingestion.py`:

```python
def test_embed_text_calls_ollama_and_returns_vector(mocker):
    mock_response = mocker.MagicMock()
    mock_response.json.return_value = {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_response.raise_for_status = mocker.MagicMock()
    mocker.patch("httpx.Client.post", return_value=mock_response)

    from app.services.ingestion import embed_text
    result = embed_text("hello world")

    assert result == [0.1, 0.2, 0.3]
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd backend && pytest tests/test_ingestion.py::test_embed_text_calls_ollama_and_returns_vector -v
```

Expected: `FAILED` — `ImportError: cannot import name 'embed_text'`

- [ ] **Step 3: Add embed_text to ingestion.py**

In `backend/app/services/ingestion.py`, add this function **before** `embed_chunks` (it does not replace `embed_chunks` — both coexist):

```python
def embed_text(text: str) -> List[float]:
    with httpx.Client() as client:
        response = client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.ollama_embedding_model, "input": [text]},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]
```

- [ ] **Step 4: Run the test to confirm it passes**

```bash
cd backend && pytest tests/test_ingestion.py::test_embed_text_calls_ollama_and_returns_vector -v
```

Expected: `PASSED`

- [ ] **Step 5: Add new schemas**

Replace the entire contents of `backend/app/schemas.py` with:

```python
from datetime import datetime
from pydantic import BaseModel
from uuid import UUID


class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    status: str

    model_config = {"from_attributes": True}


class DocumentListItem(BaseModel):
    id: UUID
    file_name: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    document_id: str
    message: str
```

- [ ] **Step 6: Run all existing backend tests to confirm nothing broke**

```bash
cd backend && pytest -v
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/app/schemas.py backend/app/services/ingestion.py backend/tests/test_ingestion.py
git commit -m "feat: add embed_text helper and DocumentListItem/ChatRequest schemas"
```

---

## Task 3: Build rag.py — search tool

**Files:**
- Create: `backend/app/services/rag.py`
- Create: `backend/tests/test_rag.py`

- [ ] **Step 1: Write the failing tests for make_search_tool**

Create `backend/tests/test_rag.py`:

```python
import uuid
import pytest
from app.models import Document, DocumentChunk


def test_search_tool_returns_formatted_chunks(mocker, db):
    mocker.patch("app.services.rag.embed_text", return_value=[0.1] * 1536)

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="test.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    chunk = DocumentChunk(
        document_id=doc_id,
        chunk_index=0,
        content="The revenue was $100M in Q3.",
        embedding=[0.1] * 1536,
    )
    db.add(chunk)
    db.flush()

    from app.services.rag import make_search_tool

    retrieved = []
    tool = make_search_tool(str(doc_id), db, retrieved)
    result = tool.invoke({"query": "Q3 revenue"})

    assert "The revenue was $100M in Q3." in result
    assert len(retrieved) == 1
    assert retrieved[0].chunk_index == 0


def test_search_tool_returns_no_results_message(mocker, db):
    mocker.patch("app.services.rag.embed_text", return_value=[0.9] * 1536)

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="empty.pdf", file_path="/tmp/e.pdf", status="done")
    db.add(doc)
    db.flush()

    from app.services.rag import make_search_tool

    retrieved = []
    tool = make_search_tool(str(doc_id), db, retrieved)
    result = tool.invoke({"query": "something"})

    assert result == "No relevant content found."
    assert retrieved == []
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd backend && pytest tests/test_rag.py -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'app.services.rag'`

- [ ] **Step 3: Create rag.py with make_search_tool only**

Create `backend/app/services/rag.py`:

```python
from typing import List
from sqlalchemy.orm import Session
from langchain_core.tools import tool

from ..models import DocumentChunk
from .ingestion import embed_text


def make_search_tool(document_id: str, db: Session, retrieved_chunks: list):
    @tool
    def search_document(query: str) -> str:
        """Search the document for chunks relevant to the query.
        Call with different phrasings if the first results are insufficient."""
        vector = embed_text(query)
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(5)
            .all()
        )
        retrieved_chunks.extend(chunks)
        return "\n\n---\n\n".join(c.content for c in chunks) or "No relevant content found."

    return search_document
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
cd backend && pytest tests/test_rag.py -v
```

Expected: both tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/rag.py backend/tests/test_rag.py
git commit -m "feat: add make_search_tool for pgvector cosine search"
```

---

## Task 4: Build rag.py — full streaming agent

**Files:**
- Modify: `backend/app/services/rag.py`
- Modify: `backend/tests/test_rag.py`

- [ ] **Step 1: Write the failing tests for agentic_rag_stream**

Add these tests to the end of `backend/tests/test_rag.py`:

```python
import asyncio


def test_agentic_rag_stream_yields_tokens_and_citations(mocker, db):
    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="test.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    chunk = DocumentChunk(
        document_id=doc_id,
        chunk_index=0,
        content="The revenue was $100M in Q3.",
        embedding=[0.1] * 1536,
    )
    db.add(chunk)
    db.flush()

    mocker.patch("app.services.rag.embed_text", return_value=[0.1] * 1536)

    mock_ai_with_tc = mocker.MagicMock()
    mock_ai_with_tc.tool_calls = [{"id": "tc1", "args": {"query": "Q3 revenue"}}]

    mock_ai_no_tc = mocker.MagicMock()
    mock_ai_no_tc.tool_calls = []

    mock_llm_with_tools = mocker.MagicMock()
    mock_llm_with_tools.ainvoke = mocker.AsyncMock(
        side_effect=[mock_ai_with_tc, mock_ai_no_tc]
    )

    mock_token1 = mocker.MagicMock()
    mock_token1.content = "Revenue"
    mock_token2 = mocker.MagicMock()
    mock_token2.content = " was $100M."

    async def mock_astream(_messages):
        for tok in [mock_token1, mock_token2]:
            yield tok

    mock_llm = mocker.MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm.astream = mock_astream

    mocker.patch("app.services.rag.ChatOllama", return_value=mock_llm)

    from app.services.rag import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "What was Q3 revenue?", db)]

    events = asyncio.run(run())

    token_events = [e for e in events if e["type"] == "token"]
    citation_event = next(e for e in events if e["type"] == "citations")
    done_event = next((e for e in events if e["type"] == "done"), None)

    assert len(token_events) == 2
    assert token_events[0]["content"] == "Revenue"
    assert token_events[1]["content"] == " was $100M."
    assert len(citation_event["chunks"]) == 1
    assert citation_event["chunks"][0]["chunk_index"] == 0
    assert done_event is not None


def test_agentic_rag_stream_skips_empty_token_content(mocker, db):
    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.flush()

    mocker.patch("app.services.rag.embed_text", return_value=[0.1] * 1536)

    mock_ai_no_tc = mocker.MagicMock()
    mock_ai_no_tc.tool_calls = []

    mock_llm_with_tools = mocker.MagicMock()
    mock_llm_with_tools.ainvoke = mocker.AsyncMock(return_value=mock_ai_no_tc)

    mock_empty = mocker.MagicMock()
    mock_empty.content = ""
    mock_real = mocker.MagicMock()
    mock_real.content = "Hello"

    async def mock_astream(_messages):
        for tok in [mock_empty, mock_real]:
            yield tok

    mock_llm = mocker.MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm.astream = mock_astream

    mocker.patch("app.services.rag.ChatOllama", return_value=mock_llm)

    from app.services.rag import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "hi", db)]

    events = asyncio.run(run())
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["content"] == "Hello"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd backend && pytest tests/test_rag.py::test_agentic_rag_stream_yields_tokens_and_citations tests/test_rag.py::test_agentic_rag_stream_skips_empty_token_content -v
```

Expected: `FAILED` — `ImportError: cannot import name 'agentic_rag_stream'`

- [ ] **Step 3: Complete rag.py with agentic_rag_stream**

Replace the full contents of `backend/app/services/rag.py`:

```python
from typing import AsyncGenerator
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from ..config import settings
from ..models import DocumentChunk
from .ingestion import embed_text

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided document. "
    "Use the search_document tool to retrieve relevant context before answering. "
    "You may search up to 3 times with different query phrasings if the first results are insufficient. "
    "Once you have sufficient context, provide a clear and thorough answer."
)


def make_search_tool(document_id: str, db: Session, retrieved_chunks: list):
    @tool
    def search_document(query: str) -> str:
        """Search the document for chunks relevant to the query.
        Call with different phrasings if the first results are insufficient."""
        vector = embed_text(query)
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(5)
            .all()
        )
        retrieved_chunks.extend(chunks)
        return "\n\n---\n\n".join(c.content for c in chunks) or "No relevant content found."

    return search_document


async def agentic_rag_stream(
    document_id: str,
    message: str,
    db: Session,
) -> AsyncGenerator[dict, None]:
    retrieved_chunks: list = []
    search_tool = make_search_tool(document_id, db, retrieved_chunks)

    llm = ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url)
    llm_with_tools = llm.bind_tools([search_tool])

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(message)]

    # Phase 1: Tool-calling rounds (not streamed — model is reasoning/acting)
    for _ in range(3):
        response = await llm_with_tools.ainvoke(messages)
        if not response.tool_calls:
            break
        messages.append(response)
        for tc in response.tool_calls:
            result = search_tool.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Phase 2: Final answer generation (streamed)
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    # Emit deduplicated citations (truncated to 400 chars for UI)
    seen: set = set()
    unique_chunks = []
    for c in retrieved_chunks:
        if c.chunk_index not in seen:
            seen.add(c.chunk_index)
            unique_chunks.append({"chunk_index": c.chunk_index, "content": c.content[:400]})

    yield {"type": "citations", "chunks": unique_chunks}
    yield {"type": "done"}
```

- [ ] **Step 4: Run all rag tests to confirm they pass**

```bash
cd backend && pytest tests/test_rag.py -v
```

Expected: all 4 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/rag.py backend/tests/test_rag.py
git commit -m "feat: add agentic_rag_stream — ReAct agent with qwen3 tool-calling and SSE streaming"
```

---

## Task 5: Add GET /api/documents endpoint

**Files:**
- Modify: `backend/app/routers/documents.py`
- Test: `backend/tests/test_upload.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `backend/tests/test_upload.py`:

```python
def test_list_documents_returns_only_done(client, db):
    from app.models import Document
    import uuid

    done = Document(id=uuid.uuid4(), file_name="done.pdf", file_path="/tmp/d.pdf", status="done")
    pending = Document(id=uuid.uuid4(), file_name="pending.pdf", file_path="/tmp/p.pdf", status="pending")
    db.add_all([done, pending])
    db.flush()

    response = client.get("/api/documents")
    assert response.status_code == 200
    names = [d["file_name"] for d in response.json()]
    assert "done.pdf" in names
    assert "pending.pdf" not in names


def test_list_documents_returns_empty_when_none_done(client, db):
    from app.models import Document
    import uuid

    doc = Document(id=uuid.uuid4(), file_name="proc.pdf", file_path="/tmp/p.pdf", status="processing")
    db.add(doc)
    db.flush()

    response = client.get("/api/documents")
    assert response.status_code == 200
    assert response.json() == []
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd backend && pytest tests/test_upload.py::test_list_documents_returns_only_done tests/test_upload.py::test_list_documents_returns_empty_when_none_done -v
```

Expected: `FAILED` — `404 Not Found`

- [ ] **Step 3: Add the endpoint to documents.py**

In `backend/app/routers/documents.py`, update the import line at the top to include `DocumentListItem`:

```python
from ..schemas import DocumentResponse, DocumentListItem
```

Then add this route before the existing `@router.post("/upload", ...)`:

```python
@router.get("", response_model=list[DocumentListItem])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.status == "done").all()
```

- [ ] **Step 4: Run the tests to confirm they pass**

```bash
cd backend && pytest tests/test_upload.py::test_list_documents_returns_only_done tests/test_upload.py::test_list_documents_returns_empty_when_none_done -v
```

Expected: both `PASSED`

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/documents.py backend/tests/test_upload.py
git commit -m "feat: add GET /api/documents — returns status=done documents only"
```

---

## Task 6: Build POST /api/chat/stream endpoint

**Files:**
- Create: `backend/app/routers/chat.py`
- Create: `backend/tests/test_chat.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_chat.py`:

```python
import uuid


def test_chat_stream_404_for_unknown_document(client):
    response = client.post(
        "/api/chat/stream",
        json={"document_id": str(uuid.uuid4()), "message": "hello"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_chat_stream_400_for_processing_document(client, db):
    from app.models import Document

    doc = Document(
        id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="processing"
    )
    db.add(doc)
    db.flush()

    response = client.post(
        "/api/chat/stream",
        json={"document_id": str(doc.id), "message": "hello"},
    )
    assert response.status_code == 400
    assert "not ready" in response.json()["detail"].lower()


def test_chat_stream_returns_event_stream_for_done_document(client, db, mocker):
    from app.models import Document

    doc = Document(
        id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done"
    )
    db.add(doc)
    db.flush()

    async def fake_rag(document_id, message, db):
        yield {"type": "token", "content": "Hello"}
        yield {"type": "citations", "chunks": []}
        yield {"type": "done"}

    mocker.patch("app.routers.chat.agentic_rag_stream", side_effect=fake_rag)

    response = client.post(
        "/api/chat/stream",
        json={"document_id": str(doc.id), "message": "hello"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"type": "token"' in response.text
    assert '"type": "done"' in response.text
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd backend && pytest tests/test_chat.py -v
```

Expected: `FAILED` — `404 Not Found` (router not registered yet)

- [ ] **Step 3: Create chat.py**

Create `backend/app/routers/chat.py`:

```python
import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db, SessionLocal
from ..models import Document
from ..schemas import ChatRequest
from ..services.rag import agentic_rag_stream

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == request.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != "done":
        raise HTTPException(status_code=400, detail="Document not ready for querying")

    async def event_stream():
        stream_db = SessionLocal()
        try:
            async for event in agentic_rag_stream(request.document_id, request.message, stream_db):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"
        finally:
            stream_db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Commit chat.py before mounting**

```bash
git add backend/app/routers/chat.py backend/tests/test_chat.py
git commit -m "feat: add POST /api/chat/stream router with pre-flight validation"
```

---

## Task 7: Register chat router and verify all backend tests pass

**Files:**
- Modify: `backend/app/main.py`

- [ ] **Step 1: Register the chat router in main.py**

Replace the full contents of `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routers import documents
from .routers import chat

app = FastAPI(title="Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Run all backend tests**

```bash
cd backend && pytest -v
```

Expected: all tests pass, including the 3 `test_chat.py` tests.

- [ ] **Step 3: Commit**

```bash
git add backend/app/main.py
git commit -m "feat: register chat router — backend complete"
```

---

## Task 8: Build DocumentSelector component

**Files:**
- Create: `src/components/DocumentSelector.jsx`
- Create: `src/components/DocumentSelector.css`

- [ ] **Step 1: Create DocumentSelector.css**

Create `src/components/DocumentSelector.css`:

```css
.doc-selector-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 20px;
  border-bottom: 1px solid rgba(0, 0, 0, 0.08);
  background: #fafafc;
}

.doc-selector-label {
  font-family: var(--font-body);
  font-size: 13px;
  font-weight: 400;
  color: rgba(0, 0, 0, 0.48);
  white-space: nowrap;
}

.doc-selector-select {
  flex: 1;
  height: 32px;
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 8px;
  padding: 0 10px;
  font-family: var(--font-body);
  font-size: 14px;
  background: #fff;
  color: #1d1d1f;
  outline: none;
  cursor: pointer;
}

.doc-selector-select:focus {
  border-color: #0071e3;
  box-shadow: 0 0 0 2px rgba(0, 113, 227, 0.18);
}

.doc-selector-select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
```

- [ ] **Step 2: Create DocumentSelector.jsx**

Create `src/components/DocumentSelector.jsx`:

```jsx
import { useState, useEffect, useCallback } from 'react'
import './DocumentSelector.css'

export default function DocumentSelector({ onChange }) {
  const [documents, setDocuments] = useState([])

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch('/api/documents')
      if (res.ok) setDocuments(await res.json())
    } catch (_) {}
  }, [])

  useEffect(() => {
    fetchDocs()
    window.addEventListener('focus', fetchDocs)
    return () => window.removeEventListener('focus', fetchDocs)
  }, [fetchDocs])

  return (
    <div className="doc-selector-row">
      <label className="doc-selector-label" htmlFor="doc-select">
        Chat with:
      </label>
      <select
        id="doc-select"
        className="doc-selector-select"
        defaultValue=""
        onChange={(e) => onChange(e.target.value || null)}
      >
        <option value="">Select a document…</option>
        {documents.map((d) => (
          <option key={d.id} value={d.id}>
            {d.file_name}
          </option>
        ))}
      </select>
    </div>
  )
}
```

- [ ] **Step 3: Start the dev server and verify no import errors**

```bash
npm run dev
```

Open `http://localhost:5173`. Check the browser console — no errors expected (the component exists but isn't wired into ChatPage yet).

- [ ] **Step 4: Commit**

```bash
git add src/components/DocumentSelector.jsx src/components/DocumentSelector.css
git commit -m "feat: add DocumentSelector component with focus-triggered refresh"
```

---

## Task 9: Build useChat hook

**Files:**
- Create: `src/hooks/useChat.js`

- [ ] **Step 1: Create useChat.js**

Create `src/hooks/useChat.js`:

```js
import { useReducer, useCallback } from 'react'

let nextId = 1
function makeId() { return nextId++ }

const initialState = { messages: [], pending: false }

function reducer(state, action) {
  switch (action.type) {
    case 'ADD_USER_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload], pending: true }

    case 'START_ASSISTANT_MESSAGE':
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: action.id, role: 'assistant', content: '', streaming: true, citations: null, error: false },
        ],
      }

    case 'APPEND_TOKEN':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, content: m.content + action.content } : m
        ),
      }

    case 'SET_CITATIONS':
      return {
        ...state,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, citations: action.chunks } : m
        ),
      }

    case 'END_STREAMING':
      return {
        ...state,
        pending: false,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, streaming: false } : m
        ),
      }

    case 'SET_ERROR':
      return {
        ...state,
        pending: false,
        messages: state.messages.map((m) =>
          m.id === action.id ? { ...m, content: action.message, error: true, streaming: false } : m
        ),
      }

    default:
      return state
  }
}

export function useChat() {
  const [state, dispatch] = useReducer(reducer, initialState)

  const handleSend = useCallback(async (message, documentId) => {
    dispatch({
      type: 'ADD_USER_MESSAGE',
      payload: { id: makeId(), role: 'user', content: message, streaming: false, citations: null, error: false },
    })

    const assistantId = makeId()
    dispatch({ type: 'START_ASSISTANT_MESSAGE', id: assistantId })

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ document_id: documentId, message }),
      })

      if (!response.ok) {
        dispatch({
          type: 'SET_ERROR',
          id: assistantId,
          message: `Request failed (${response.status}). Please try again.`,
        })
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete last line in buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          let event
          try {
            event = JSON.parse(line.slice(6))
          } catch (_) {
            continue
          }

          if (event.type === 'token') {
            dispatch({ type: 'APPEND_TOKEN', id: assistantId, content: event.content })
          } else if (event.type === 'citations') {
            dispatch({ type: 'SET_CITATIONS', id: assistantId, chunks: event.chunks })
          } else if (event.type === 'done') {
            dispatch({ type: 'END_STREAMING', id: assistantId })
          } else if (event.type === 'error') {
            dispatch({ type: 'SET_ERROR', id: assistantId, message: event.message })
          }
        }
      }
    } catch (_) {
      dispatch({
        type: 'SET_ERROR',
        id: assistantId,
        message: 'Network error. Please check your connection and try again.',
      })
    }
  }, [])

  return { messages: state.messages, pending: state.pending, handleSend }
}
```

- [ ] **Step 2: Commit**

```bash
git add src/hooks/useChat.js
git commit -m "feat: add useChat hook with streaming SSE state management"
```

---

## Task 10: Update ChatMessage and ChatThread

**Files:**
- Modify: `src/components/ChatMessage.jsx`
- Modify: `src/components/ChatMessage.css`
- Modify: `src/components/ChatThread.jsx`

- [ ] **Step 1: Append cursor and citations styles to ChatMessage.css**

Append to the end of `src/components/ChatMessage.css`:

```css
/* Streaming cursor */
.chat-msg-cursor {
  display: inline-block;
  width: 2px;
  height: 1em;
  background: var(--color-primary);
  margin-left: 2px;
  vertical-align: text-bottom;
  animation: blink 0.9s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* Citations */
.chat-msg-citations {
  margin-top: 8px;
  border-top: 1px solid rgba(0, 0, 0, 0.08);
  padding-top: 8px;
}

.chat-msg-citations-toggle {
  font-family: var(--font-body);
  font-size: 13px;
  font-weight: 600;
  color: rgba(0, 0, 0, 0.48);
  cursor: pointer;
  list-style: none;
  user-select: none;
}

.chat-msg-citations-toggle::-webkit-details-marker {
  display: none;
}

.chat-msg-citations-toggle::before {
  content: '▶ ';
  font-size: 10px;
}

details[open] .chat-msg-citations-toggle::before {
  content: '▼ ';
}

.chat-msg-citations-list {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  list-style: none;
  padding: 0;
}

.chat-msg-citation-item {
  display: flex;
  gap: 8px;
  align-items: flex-start;
}

.chat-msg-citation-index {
  font-family: var(--font-body);
  font-size: 12px;
  font-weight: 600;
  color: var(--color-primary);
  white-space: nowrap;
  padding-top: 2px;
}

.chat-msg-citation-text {
  font-family: var(--font-body);
  font-size: 13px;
  color: rgba(0, 0, 0, 0.6);
  line-height: 1.5;
}

/* Error state */
.chat-msg--error .chat-msg-body .msg-para {
  color: #c0392b;
}
```

- [ ] **Step 2: Replace ChatMessage.jsx**

Replace the full contents of `src/components/ChatMessage.jsx`:

```jsx
import './ChatMessage.css'

function renderContent(content) {
  const lines = content.split('\n')
  const elements = []
  let listItems = []
  let key = 0

  function flushList() {
    if (listItems.length > 0) {
      elements.push(
        <ol key={key++} className="msg-list">
          {listItems.map((li, i) => (
            <li key={i} className="msg-list-item">
              <span className="msg-list-label">{li.label}</span>
              <span className="msg-list-body" dangerouslySetInnerHTML={{ __html: li.body }} />
            </li>
          ))}
        </ol>
      )
      listItems = []
    }
  }

  for (const line of lines) {
    if (!line.trim()) continue

    const listMatch = line.match(/^(\d+)\.\s+\*\*(.+?)\*\*[:]\s*(.*)$/)
    if (listMatch) {
      listItems.push({ label: listMatch[2] + ':', body: listMatch[3] })
      continue
    }

    flushList()

    const html = line.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    elements.push(<p key={key++} className="msg-para" dangerouslySetInnerHTML={{ __html: html }} />)
  }

  flushList()
  return elements
}

export default function ChatMessage({ role, content, streaming, citations, error }) {
  const isUser = role === 'user'

  return (
    <div className={`chat-msg ${isUser ? 'chat-msg--user' : 'chat-msg--assistant'}${error ? ' chat-msg--error' : ''}`}>
      {isUser ? (
        <div className="chat-msg-user-row">
          <div className="chat-msg-user-avatar">AN</div>
          <p className="chat-msg-user-text">{content}</p>
          <button className="chat-msg-edit-btn" title="Edit">
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <rect x="1.5" y="1.5" width="11" height="11" rx="2" stroke="currentColor" strokeWidth="1.3" />
              <path d="M4.5 9.5l1-3 5-5 2 2-5 5-3 1z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      ) : (
        <div className="chat-msg-assistant">
          <div className="chat-msg-brand">
            <span className="chat-msg-brand-label">CHAT A.I +</span>
            <button className="chat-msg-brand-info" title="About">
              <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                <circle cx="7" cy="7" r="6" stroke="currentColor" strokeWidth="1.3" />
                <path d="M7 6v4M7 4.5v.5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            </button>
          </div>
          <div className="chat-msg-body">
            {renderContent(content)}
            {streaming && <span className="chat-msg-cursor" aria-hidden="true" />}
          </div>
          {citations && citations.length > 0 && (
            <details className="chat-msg-citations">
              <summary className="chat-msg-citations-toggle">
                Sources ({citations.length})
              </summary>
              <ul className="chat-msg-citations-list">
                {citations.map((c) => (
                  <li key={c.chunk_index} className="chat-msg-citation-item">
                    <span className="chat-msg-citation-index">§{c.chunk_index + 1}</span>
                    <p className="chat-msg-citation-text">{c.content}</p>
                  </li>
                ))}
              </ul>
            </details>
          )}
          {!streaming && !error && (
            <div className="chat-msg-actions">
              <div className="chat-msg-reactions">
                <button className="chat-msg-action-btn" title="Thumbs up">
                  <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                    <path d="M5 13V7.5L8 2l1 .5v4h4l-1 6H5zM2 7.5h3V13H2V7.5z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
                  </svg>
                </button>
                <span className="chat-msg-action-divider" />
                <button className="chat-msg-action-btn" title="Thumbs down">
                  <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                    <path d="M10 2v5.5L7 13l-1-.5v-4H2l1-6h7zM13 7.5h-3V2h3v5.5z" stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round" />
                  </svg>
                </button>
                <span className="chat-msg-action-divider" />
                <button className="chat-msg-action-btn" title="Copy">
                  <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                    <rect x="5" y="5" width="8" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.3" />
                    <path d="M3 10H2.5A1.5 1.5 0 011 8.5v-7A1.5 1.5 0 012.5 0h7A1.5 1.5 0 0111 1.5V2" stroke="currentColor" strokeWidth="1.3" />
                  </svg>
                </button>
                <span className="chat-msg-action-divider" />
                <button className="chat-msg-action-btn" title="More">
                  <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
                    <circle cx="3.5" cy="7.5" r="1" fill="currentColor" />
                    <circle cx="7.5" cy="7.5" r="1" fill="currentColor" />
                    <circle cx="11.5" cy="7.5" r="1" fill="currentColor" />
                  </svg>
                </button>
              </div>
              <button className="chat-msg-regenerate">
                <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
                  <path d="M2 7a5 5 0 109.5-2.2M11.5 2v3h-3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                Regenerate
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: Replace ChatThread.jsx**

Replace the full contents of `src/components/ChatThread.jsx`:

```jsx
import { useEffect, useRef } from 'react'
import ChatMessage from './ChatMessage'
import './ChatThread.css'

export default function ChatThread({ messages, pending }) {
  const bottomRef = useRef(null)
  const isStreaming = messages.some((m) => m.streaming)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, pending])

  return (
    <div className="chat-thread">
      {messages.map((msg) => (
        <ChatMessage
          key={msg.id}
          role={msg.role}
          content={msg.content}
          streaming={msg.streaming}
          citations={msg.citations}
          error={msg.error}
        />
      ))}
      {pending && !isStreaming && (
        <div className="chat-thread-typing">
          <span className="chat-thread-typing-label">CHAT A.I +</span>
          <span className="chat-thread-typing-dots">
            <span /><span /><span />
          </span>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  )
}
```

- [ ] **Step 4: Verify in browser**

```bash
npm run dev
```

Open `http://localhost:5173`. The UI looks unchanged (ChatPage not yet wired). No console errors.

- [ ] **Step 5: Commit**

```bash
git add src/components/ChatMessage.jsx src/components/ChatMessage.css src/components/ChatThread.jsx
git commit -m "feat: add streaming cursor and citations collapsible to ChatMessage"
```

---

## Task 11: Wire ChatPage — final integration

**Files:**
- Modify: `src/pages/ChatPage.jsx`
- Modify: `src/pages/ChatPage.css`

- [ ] **Step 1: Append hint style to ChatPage.css**

Append to the end of `src/pages/ChatPage.css`:

```css
.chat-page-hint {
  padding: 8px 20px;
  font-family: var(--font-body);
  font-size: 13px;
  color: rgba(0, 0, 0, 0.38);
  text-align: center;
}
```

- [ ] **Step 2: Replace ChatPage.jsx**

Replace the full contents of `src/pages/ChatPage.jsx`:

```jsx
import { useState, useCallback } from 'react'
import Sidebar from '../components/Sidebar'
import ChatThread from '../components/ChatThread'
import Composer from '../components/Composer'
import UpgradeTab from '../components/UpgradeTab'
import UploadToast from '../components/UploadToast'
import DocumentSelector from '../components/DocumentSelector'
import { useUpload } from '../hooks/useUpload'
import { useChat } from '../hooks/useChat'
import './ChatPage.css'

export default function ChatPage() {
  const [selectedDocumentId, setSelectedDocumentId] = useState(null)
  const [showToast, setShowToast] = useState(false)
  const { messages, pending, handleSend } = useChat()

  const upload = useUpload({
    onComplete: () => setShowToast(true),
  })

  const onSend = useCallback(
    (text) => {
      if (!selectedDocumentId) return
      handleSend(text, selectedDocumentId)
    },
    [selectedDocumentId, handleSend]
  )

  return (
    <div className="chat-page-outer">
      <div className="chat-page-card">
        <Sidebar activeTitle="Chat with Document" />
        <div className="chat-main">
          <DocumentSelector onChange={setSelectedDocumentId} />
          <ChatThread messages={messages} pending={pending} />
          {!selectedDocumentId && (
            <p className="chat-page-hint">Select a document above to start chatting.</p>
          )}
          <Composer
            onSend={onSend}
            onFileSelect={upload.uploadFile}
            uploadStatus={upload.status}
            disabled={pending || !selectedDocumentId}
          />
        </div>
        <div className="chat-page-upgrade">
          <UpgradeTab />
        </div>
      </div>
      {showToast && (
        <UploadToast
          status={upload.status}
          fileName={upload.fileName}
          error={upload.error}
          onDismiss={() => {
            setShowToast(false)
            upload.reset()
          }}
        />
      )}
    </div>
  )
}
```

- [ ] **Step 3: Start dev server**

```bash
npm run dev
```

- [ ] **Step 4: Manual E2E verification — golden path**

With the full Docker stack running (`docker compose up`):

1. Navigate to `http://localhost:5173`.
2. Upload a PDF via the Composer's upload button. Wait for the "Document ready for Q&A" toast.
3. Switch to another browser tab, then come back (triggers `focus` re-fetch on `DocumentSelector`). The uploaded document should appear in the dropdown.
4. Select the document. The "Select a document above" hint disappears and the send button activates.
5. Type a question about the document content and press Enter.
6. **Verify:** user message appears immediately in the thread.
7. **Verify:** typing dots appear briefly (while the agent is making its tool calls), then the assistant message appears and tokens stream in one by one.
8. **Verify:** after generation ends, a "Sources (N)" disclosure appears below the answer.
9. **Verify:** clicking "Sources" expands chunk excerpts with `§1`, `§2`, … labels.

- [ ] **Step 5: Manual verification — error paths**

1. With no document selected: the send button is disabled (greyed out); pressing Enter does nothing.
2. With backend stopped (stop Docker): select a doc, send a message — an error bubble appears in red in the chat thread.

- [ ] **Step 6: Run ESLint**

```bash
npm run lint
```

Expected: no errors or warnings.

- [ ] **Step 7: Commit**

```bash
git add src/pages/ChatPage.jsx src/pages/ChatPage.css
git commit -m "feat: wire ChatPage with DocumentSelector and useChat — agentic RAG chat complete"
```

---

## Spec Coverage Checklist

| Spec requirement | Covered by |
|---|---|
| `ollama_chat_model` config key | Task 1 |
| `langchain-ollama`, `langchain-core` deps | Task 1 |
| `embed_text` shared helper | Task 2 |
| `DocumentListItem`, `ChatRequest` schemas | Task 2 |
| `make_search_tool` (pgvector cosine search) | Task 3 |
| `agentic_rag_stream` (ReAct loop + streaming) | Task 4 |
| `GET /api/documents` (status=done filter) | Task 5 |
| `POST /api/chat/stream` (pre-flight + SSE) | Task 6 |
| Register chat router in `main.py` | Task 7 |
| `DocumentSelector` component | Task 8 |
| `useChat` hook (streaming SSE state) | Task 9 |
| `ChatMessage` cursor + citations | Task 10 |
| `ChatThread` prop pass-through + typing gate | Task 10 |
| `ChatPage` wired end-to-end | Task 11 |
| Error handling (404, 400, SSE error, network) | Tasks 6, 9, 10 |
