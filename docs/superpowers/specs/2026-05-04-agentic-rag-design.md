# Agentic RAG Chat Feature — Design Spec

**Date:** 2026-05-04
**Status:** Approved

---

## 1. Overview

Add a "chat with document" feature to the existing chatbot UI. The user selects one of their uploaded documents, types a question, and receives a streamed answer grounded in that document's content. The backend uses an Agentic RAG pipeline — a ReAct agent powered by `qwen3` (via Ollama) that calls a vector search tool iteratively before generating a final answer. Source citations are shown below each answer.

---

## 2. Goals

- Let users query any fully-ingested document by selecting it from a dropdown.
- Replace the current canned-reply placeholder with a real LLM-backed response.
- Stream tokens to the UI as they are generated (low time-to-first-token).
- Show which document chunks backed the answer (citations).
- Keep the implementation simple: one document per query, no multi-turn memory, no web-search fallback.

---

## 3. Architecture

```
Frontend                        Backend (FastAPI)              Ollama Server
─────────                       ──────────────────             ─────────────
DocumentSelector ──GET /api/documents──────────────────────►  (pgvector query)
                ◄─ [{id, name, status}] ─────────────────────

Composer (doc selected)
  │ POST /api/chat/stream
  │ {document_id, message}
  ▼
ChatPage (fetch + ReadableStream)
  │                             ┌─ rag.py (ReAct Agent) ─────► embed query
  │◄── SSE token events ───────┤      qwen3 thinks                (qwen3-embedding:4b)
  │◄── SSE citations event ────┤      → tool: search_document()
  │◄── SSE done event ─────────┤      → observe chunks            pgvector cosine search
                                │      → tool call again? (0–2x)  (by document_id)
                                │      → stream final answer ─────► qwen3
                                └──────────────────────────────────◄ tokens
```

---

## 4. Backend

### 4.1 New Config Key

`backend/app/config.py` — add:

```python
ollama_chat_model: str = "qwen3"
```

### 4.2 New Dependencies

`backend/requirements.txt` — add:

```
langchain-ollama
langchain-core
```

### 4.3 `GET /api/documents`

Added to the existing `backend/app/routers/documents.py` router.

- Returns all documents where `status = "done"`.
- Response shape: `[{id, file_name, uploaded_at}]`
- Used exclusively to populate the frontend document selector.

### 4.4 `POST /api/chat/stream`

New router: `backend/app/routers/chat.py`, mounted at `/api/chat` in `main.py`.

**Request body:**
```json
{ "document_id": "uuid", "message": "What are the key findings?" }
```

**Pre-flight validation (synchronous, before stream opens):**
- Document must exist → `404` if not.
- Document `status` must be `"done"` → `400` if not.

**Response:** `Content-Type: text/event-stream`

**SSE event protocol:**
```
data: {"type": "token",     "content": "The key findings are..."}
data: {"type": "token",     "content": " three areas..."}
data: {"type": "citations", "chunks": [{"chunk_index": 2, "content": "...excerpt..."}]}
data: {"type": "done"}

# On any error after stream opens:
data: {"type": "error", "message": "Agent failed to retrieve context"}
```

Tokens stream during final generation only. Citations are a single event emitted after the last token, before `done`.

### 4.5 Agentic RAG Service — `backend/app/services/rag.py`

This is the core of the feature.

**Embedding reuse:**
The existing `embed_chunks` function in `ingestion.py` is refactored to expose a shared `embed_text(text: str) -> List[float]` helper used at both ingestion time and query time.

**Search tool (request-scoped closure):**
```python
def make_search_tool(document_id: str, db: Session, retrieved_chunks: list) -> Tool:
    @tool
    def search_document(query: str) -> str:
        """Search the document for chunks relevant to the query.
           Call with different phrasings if the first result is insufficient."""
        vector = embed_text(query)
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(5)
            .all()
        )
        retrieved_chunks.extend(chunks)
        return "\n\n---\n\n".join(c.content for c in chunks)
    return search_document
```

**ReAct agent loop:**

```python
llm = ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url)
llm_with_tools = llm.bind_tools([search_tool])

system_prompt = """You are a helpful assistant that answers questions strictly
based on the provided document. Use the search_document tool to retrieve
relevant context. You may search up to 3 times with different query phrasings
if needed. Once you have sufficient context, provide a clear and thorough answer."""

messages = [SystemMessage(system_prompt), HumanMessage(user_message)]
retrieved_chunks: list = []

# Tool-call iterations (not streamed — model is reasoning/acting)
for _ in range(3):
    response = await llm_with_tools.ainvoke(messages)
    messages.append(response)
    if not response.tool_calls:
        break  # model finished retrieving, ready to generate
    for tc in response.tool_calls:
        result = await search_tool.ainvoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

# Final streaming generation
async for chunk in llm.astream(messages):
    yield {"type": "token", "content": chunk.content}

# Deduplicate by chunk_index, truncate content for citations
seen = set()
unique_chunks = []
for c in retrieved_chunks:
    if c.chunk_index not in seen:
        seen.add(c.chunk_index)
        unique_chunks.append({"chunk_index": c.chunk_index, "content": c.content[:400]})

yield {"type": "citations", "chunks": unique_chunks}
yield {"type": "done"}
```

**Bounds:** Max 3 tool-call iterations. In practice qwen3 typically uses 1–2.

---

## 5. Frontend

### 5.1 `DocumentSelector` component

New file: `src/components/DocumentSelector.jsx`

- Fetches `GET /api/documents` on mount.
- Renders a `<select>` with a default "Select a document…" placeholder option.
- Placed above the `Composer` in `ChatPage`.
- Re-fetches when the window regains focus (picks up newly-ingested documents).

```
┌─────────────────────────────────────────────────┐
│  Chat with document:  [select document ▼]       │
├─────────────────────────────────────────────────┤
│  [textarea]                          [📎] [↑]   │
└─────────────────────────────────────────────────┘
```

### 5.2 `useChat` hook

New file: `src/hooks/useChat.js`

Replaces the canned-reply `setTimeout` in `ChatPage`. Owns the full message list state and streaming logic.

**State shape per message:**
```js
{
  id: number,
  role: 'user' | 'assistant',
  content: string,
  streaming: boolean,       // true while tokens are arriving
  citations: Array | null,  // set when citations event arrives
  error: boolean,           // true if error event received
}
```

**`handleSend(message, documentId)` flow:**
1. Append user message to state.
2. Open `fetch('POST /api/chat/stream', {document_id, message})`.
3. Append empty assistant message with `streaming: true`.
4. Read `response.body` as a `ReadableStream`, decode line by line.
5. Parse each `data: {...}` line:
   - `token` → append `content` to the streaming message.
   - `citations` → set `citations` on the message.
   - `done` → set `streaming: false`.
   - `error` → set `error: true`, `content` = error message, `streaming: false`.
6. On fetch network failure → same error path.

### 5.3 `ChatMessage` changes

Two new rendering cases:

- **`streaming: true`** — append a blinking `|` cursor after content.
- **`citations` array present** — render a collapsible "Sources" disclosure below the answer, showing `chunk_index` and the first 200 characters of each chunk's content.

### 5.4 `ChatPage` changes

- Add `selectedDocumentId` state (starts `null`).
- Render `<DocumentSelector onChange={setSelectedDocumentId} />` above `<Composer>`.
- Pass `selectedDocumentId` to `useChat.handleSend`.
- Disable Composer send button when `selectedDocumentId` is `null` and show a helper label: _"Select a document above to start chatting."_
- Remove `SEED_MESSAGES` and `CANNED_REPLIES` — `useChat` owns the message list.

---

## 6. Error Handling

| Scenario | Behaviour |
|---|---|
| Document not found | `404` before stream opens |
| Document not yet ingested | `400` before stream opens |
| pgvector / Ollama error during tool call | `error` SSE event, stream closes |
| Agent exceeds 3 iterations without answer | `error` SSE event, stream closes |
| Embedding fails at query time | `error` SSE event, stream closes |
| Network failure on frontend | Error bubble in chat thread |
| No document selected | Composer send button disabled — no request made |

No automatic retry on either side. User re-sends on failure.

---

## 7. Files Changed

| File | Change |
|---|---|
| `backend/requirements.txt` | Add `langchain-ollama`, `langchain-core` |
| `backend/app/config.py` | Add `ollama_chat_model: str = "qwen3"` |
| `backend/app/services/ingestion.py` | Extract `embed_text(query)` shared helper |
| `backend/app/services/rag.py` | **New** — ReAct agent, search tool, streaming generator |
| `backend/app/routers/documents.py` | Add `GET /api/documents` |
| `backend/app/routers/chat.py` | **New** — `POST /api/chat/stream` SSE endpoint |
| `backend/app/main.py` | Register chat router |
| `src/components/DocumentSelector.jsx` | **New** — document `<select>` with fetch |
| `src/components/DocumentSelector.css` | **New** — styles |
| `src/hooks/useChat.js` | **New** — streaming state + fetch logic |
| `src/components/ChatMessage.jsx` | Add streaming cursor + citations collapsible |
| `src/pages/ChatPage.jsx` | Wire DocumentSelector + useChat, remove canned replies |

---

## 8. Out of Scope

- Multi-document querying per message.
- Conversation memory / multi-turn context across messages.
- Web-search fallback when document context is insufficient.
- Re-ranking retrieved chunks (e.g., cross-encoder reranker).
- Authentication / per-user document isolation.
