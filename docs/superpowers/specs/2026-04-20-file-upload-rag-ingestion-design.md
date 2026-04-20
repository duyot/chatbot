# File Upload & RAG Ingestion Pipeline — Design Spec

**Date:** 2026-04-20  
**Status:** Approved

---

## Overview

Add file upload capability to the chat UI so users can upload documents (PDF, DOCX, images). Each upload is stored to PostgreSQL, then processed asynchronously by a Celery worker that chunks the text, embeds it via OpenAI, and stores the result in pgvector — preparing the document for RAG retrieval. The user is notified via SSE when ingestion is complete.

---

## 1. Architecture

Five services communicating in a pipeline:

```
React (Vite) → FastAPI → Redis → Celery Worker → PostgreSQL + pgvector
                  ↑                                      |
                  └──────── SSE status stream ───────────┘
```

| Service | Technology | Role |
|---------|-----------|------|
| `frontend` | React 19 + Vite, served by nginx | Upload UI, SSE listener, toast notification |
| `backend` | FastAPI (Python) | REST API: upload endpoint + SSE status stream |
| `worker` | Celery (same image as backend, different command) | Document ingestion: parse → chunk → embed → store |
| `redis` | Redis 7 | Celery broker and results backend |
| `db` | pgvector/pgvector (PostgreSQL 16 + pgvector extension) | Document metadata + vector chunks |

All five are defined in `docker-compose.yml`. The connection string for PostgreSQL and the Redis URL are configurable via environment variables.

---

## 2. Database Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  file_name   TEXT        NOT NULL,
  file_path   TEXT        NOT NULL,
  uploaded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  status      TEXT        NOT NULL DEFAULT 'pending',
  -- status values: pending | processing | done | failed
  error_msg   TEXT
);

CREATE TABLE document_chunks (
  id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID        NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER     NOT NULL,
  content     TEXT        NOT NULL,
  embedding   vector(1536)  -- OpenAI text-embedding-3-small produces 1536 dims
);

CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops);
```

Uploaded files are saved to a local directory (configurable via `UPLOAD_DIR` env var, default `./uploads`). `file_path` stores the relative path within that directory.

---

## 3. API Design

### POST `/api/documents/upload`

- **Content-Type:** `multipart/form-data`
- **Field:** `file` — the uploaded file (PDF, DOCX, or image)
- **Accepted MIME types:** `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `image/png`, `image/jpeg`, `image/webp`
- **Max file size:** 20 MB (configurable via `MAX_UPLOAD_MB` env var)
- **Response 200:**
  ```json
  { "id": "<uuid>", "file_name": "report.pdf", "status": "pending" }
  ```
- **Response 400:** unsupported file type or size exceeded
- **Steps:** validate → save file to disk → insert `documents` row → enqueue Celery task → return response

### GET `/api/documents/{id}/status`

- **Response:** `text/event-stream` (SSE)
- **Event format:**
  ```
  data: {"status": "processing", "message": "Ingesting document..."}

  data: {"status": "done", "message": "Document ready for Q&A."}
  ```
- **Behavior:** polls the `documents` table every 2 seconds; closes the stream when status reaches `done` or `failed`
- **Response 404:** document ID not found

---

## 4. Ingestion Pipeline (Celery Task)

The task `ingest_document(document_id)` runs these steps in order, updating `documents.status` at each stage:

1. **Set status → `processing`**
2. **Parse** — extract plain text from the file:
   - PDF: `PyMuPDF` (`fitz.open`)
   - DOCX: `python-docx` (`Document.paragraphs`)
   - Image: store a placeholder note (`"[image: {filename}]"`) — no OCR in v1
3. **Chunk** — split extracted text using LangChain `RecursiveCharacterTextSplitter`:
   - `chunk_size=1000`, `chunk_overlap=200`
4. **Embed** — call `openai.embeddings.create(model="text-embedding-3-small", input=batch)` in batches of 100 chunks
5. **Store** — bulk insert all `(document_id, chunk_index, content, embedding)` rows into `document_chunks`
6. **Set status → `done`**
7. **On any exception:** set status → `failed`, write exception message to `error_msg`

OpenAI API key is read from the `OPENAI_API_KEY` environment variable.

---

## 5. Frontend Changes

### Modified: `src/components/Composer.jsx`

- Add a paperclip/upload icon button to the left of the send button (right side of the composer row). The model icon and textarea stay in their existing positions.
- Add a hidden `<input type="file" accept=".pdf,.docx,image/*">` triggered by the upload button click.
- When a file is selected, display a **file chip** below the composer input showing the filename and an × to deselect.
- The upload is submitted independently of the chat message — clicking the upload button (or selecting a file) immediately triggers the upload; it does not wait for the user to press Send.
- While upload + ingestion is in progress, the upload button shows a spinner and is disabled.

### New: `src/hooks/useUpload.js`

Encapsulates:
- `uploadFile(file)` — POSTs to `/api/documents/upload`, returns `{id}`
- Opens an `EventSource` to `/api/documents/{id}/status`
- Manages state: `{ status: 'idle' | 'uploading' | 'processing' | 'done' | 'failed', fileName, error }`
- Calls a `onComplete(fileName)` callback when SSE emits `done`

### New: `src/components/UploadToast.jsx`

- Fixed-position toast, bottom-right
- Shows on `done`: green checkmark, "Document ready — {filename} has been ingested and is ready for Q&A."
- Shows on `failed`: red X, error message
- Auto-dismisses after 5 seconds; also has a manual × close button

---

## 6. Backend File Structure

```
backend/
  app/
    main.py               # FastAPI app factory, CORS config, router registration
    config.py             # Pydantic Settings — DB URL, Redis URL, OpenAI key, upload dir
    models.py             # SQLAlchemy ORM models (Document, DocumentChunk)
    schemas.py            # Pydantic request/response schemas
    database.py           # SQLAlchemy engine + session factory
    routers/
      documents.py        # POST /upload, GET /{id}/status
    services/
      ingestion.py        # parse_file(), chunk_text(), embed_chunks(), store_chunks()
    workers/
      celery_app.py       # Celery instance (broker=Redis, backend=Redis)
      tasks.py            # ingest_document task
  requirements.txt
  Dockerfile              # Multi-stage: build + runtime
  alembic/                # DB migrations
    env.py
    versions/
      0001_initial.py     # creates documents + document_chunks tables
```

---

## 7. Configuration (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | *(required)* | PostgreSQL connection string, e.g. `postgresql+psycopg2://user:pass@db:5432/chatbot` (sync driver used by both FastAPI and Celery for v1 simplicity) |
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker + backend |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for embeddings |
| `UPLOAD_DIR` | `./uploads` | Directory where uploaded files are saved |
| `MAX_UPLOAD_MB` | `20` | Max file size in MB |
| `CORS_ORIGINS` | `http://localhost:3000` | Allowed CORS origins for the frontend |

The `docker-compose.yml` passes these via an `.env` file at the project root. The `.env` file is in `.gitignore`.

---

## 8. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Unsupported file type | FastAPI returns 400 before saving anything |
| File exceeds size limit | FastAPI returns 400 before saving anything |
| OpenAI API error during embedding | Task catches exception, sets status `failed`, writes error to DB |
| Celery worker crash mid-task | Celery retries once (max_retries=1) with 10s delay; on final failure sets status `failed` |
| SSE client disconnects | FastAPI generator detects disconnect and stops polling |
| Document ID not found in SSE endpoint | Returns HTTP 404 immediately |

---

## 9. Out of Scope (v1)

- OCR for image files (images get a placeholder chunk)
- Multi-file upload in a single request
- RAG retrieval / using the chunks to answer questions (this spec prepares the data; retrieval is a separate feature)
- User authentication / per-user document isolation
- Cloud storage (S3, GCS) — local disk only in v1
- Chunk deletion or re-ingestion on re-upload of the same filename
