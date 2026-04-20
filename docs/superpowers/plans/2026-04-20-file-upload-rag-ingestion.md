# File Upload & RAG Ingestion Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a file upload button to the chat Composer, store document metadata in PostgreSQL, run Celery-backed ingestion (chunk → embed via OpenAI → store in pgvector), and notify the user via SSE when ingestion is complete.

**Architecture:** Five Docker services: `frontend` (nginx/React), `backend` (FastAPI), `worker` (Celery — same image, different CMD), `redis` (broker), `db` (pgvector/pgvector:pg16). Upload triggers a Celery task; the frontend opens an SSE stream to poll document status until it reaches `done` or `failed`.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2 (sync/psycopg2), Alembic, Celery 5 + Redis, pgvector, OpenAI SDK, langchain-text-splitters, PyMuPDF, python-docx; React 19/Vite (frontend).

---

## File Map

| Path | Action | Responsibility |
|------|--------|---------------|
| `docker-compose.yml` | Modify | Add `db`, `redis`, `backend`, `worker` services |
| `.env.example` | Create | Environment variable template |
| `.gitignore` | Modify | Add `.env`, `uploads/`, `backend/__pycache__` |
| `vite.config.js` | Modify | Add dev proxy `/api → http://localhost:8000` |
| `backend/requirements.txt` | Create | Python dependencies |
| `backend/Dockerfile` | Create | Build backend + worker image |
| `backend/app/__init__.py` | Create | Package marker |
| `backend/app/config.py` | Create | Pydantic Settings from env vars |
| `backend/app/database.py` | Create | SQLAlchemy engine + session + `get_db` |
| `backend/app/models.py` | Create | `Document` + `DocumentChunk` ORM models |
| `backend/app/schemas.py` | Create | Pydantic response schemas |
| `backend/app/main.py` | Create | FastAPI app factory + CORS + router registration |
| `backend/app/routers/__init__.py` | Create | Package marker |
| `backend/app/routers/documents.py` | Create | `POST /api/documents/upload` + `GET /api/documents/{id}/status` |
| `backend/app/services/__init__.py` | Create | Package marker |
| `backend/app/services/ingestion.py` | Create | `parse_file`, `chunk_text`, `embed_chunks`, `store_chunks` |
| `backend/app/workers/__init__.py` | Create | Package marker |
| `backend/app/workers/celery_app.py` | Create | Celery instance wired to Redis |
| `backend/app/workers/tasks.py` | Create | `ingest_document` Celery task |
| `backend/alembic.ini` | Create | Alembic config (url sourced from env) |
| `backend/alembic/env.py` | Create | Alembic migration env |
| `backend/alembic/script.py.mako` | Create | Migration template |
| `backend/alembic/versions/0001_initial.py` | Create | Creates both tables + ivfflat index |
| `backend/tests/__init__.py` | Create | Package marker |
| `backend/tests/conftest.py` | Create | Test DB setup + FastAPI TestClient fixture |
| `backend/tests/test_upload.py` | Create | Upload endpoint + SSE endpoint tests |
| `backend/tests/test_ingestion.py` | Create | Service-layer unit tests |
| `backend/tests/test_tasks.py` | Create | Celery task tests (sync `apply()`) |
| `src/hooks/useUpload.js` | Create | Upload + SSE state hook |
| `src/components/UploadToast.jsx` | Create | Completion/failure toast |
| `src/components/UploadToast.css` | Create | Toast styles |
| `src/components/Composer.jsx` | Modify | Upload button + file chip |
| `src/components/Composer.css` | Modify | Upload button + chip styles |
| `src/pages/ChatPage.jsx` | Modify | Wire `useUpload` + render `UploadToast` |

---

## Task 1: Infrastructure — docker-compose, env, gitignore

**Files:**
- Modify: `docker-compose.yml`
- Create: `.env.example`
- Modify: `.gitignore`

- [ ] **Step 1: Replace docker-compose.yml**

```yaml
# docker-compose.yml
services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - uploads:/app/uploads
    depends_on:
      - db
      - redis
    restart: unless-stopped

  worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    command: celery -A app.workers.celery_app.celery_app worker --loglevel=info
    env_file: .env
    volumes:
      - uploads:/app/uploads
    depends_on:
      - db
      - redis
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: chatbot
      POSTGRES_USER: chatbot
      POSTGRES_PASSWORD: chatbot
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  uploads:
  pgdata:
```

- [ ] **Step 2: Create .env.example**

```bash
# .env.example
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@db:5432/chatbot
REDIS_URL=redis://redis:6379/0
OPENAI_API_KEY=sk-...
UPLOAD_DIR=./uploads
MAX_UPLOAD_MB=20
CORS_ORIGINS=http://localhost:3000
```

- [ ] **Step 3: Add entries to .gitignore**

Append to the existing `.gitignore`:
```
.env
uploads/
backend/__pycache__/
backend/**/__pycache__/
backend/.pytest_cache/
backend/alembic/versions/*.pyc
```

- [ ] **Step 4: Verify docker-compose syntax**

```bash
docker compose config --quiet
```
Expected: no output (exits 0). Fix any YAML errors before continuing.

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml .env.example .gitignore
git commit -m "feat: add multi-service docker-compose for backend/worker/redis/db"
```

---

## Task 2: Backend Python scaffold

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/Dockerfile`
- Create: `backend/app/__init__.py`
- Create: `backend/app/routers/__init__.py`
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/workers/__init__.py`
- Create: `backend/app/config.py`
- Create: `backend/app/main.py`

- [ ] **Step 1: Create backend/requirements.txt**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
sqlalchemy==2.0.35
psycopg2-binary==2.9.9
alembic==1.13.3
celery[redis]==5.4.0
redis==5.1.1
openai==1.54.0
python-multipart==0.0.12
langchain-text-splitters==0.3.2
pymupdf==1.24.11
python-docx==1.1.2
pgvector==0.3.2
pydantic-settings==2.5.2
pytest==8.3.3
httpx==0.27.2
pytest-mock==3.14.0
```

- [ ] **Step 2: Create backend/Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Create package markers**

Create these four empty files:
- `backend/app/__init__.py` (empty)
- `backend/app/routers/__init__.py` (empty)
- `backend/app/services/__init__.py` (empty)
- `backend/app/workers/__init__.py` (empty)

- [ ] **Step 4: Create backend/app/config.py**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str
    upload_dir: str = "./uploads"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 5: Create backend/app/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routers import documents

app = FastAPI(title="Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)

@app.get("/health")
def health():
    return {"status": "ok"}
```

- [ ] **Step 6: Verify the app starts locally**

From `backend/` directory (with `.env` copied from `.env.example` and filled in):
```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot \
OPENAI_API_KEY=test \
uvicorn app.main:app --port 8000
```
Expected: `Application startup complete.` — then Ctrl-C.

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "feat: scaffold FastAPI backend with config and health endpoint"
```

---

## Task 3: Database layer — models + Alembic migration

**Files:**
- Create: `backend/app/database.py`
- Create: `backend/app/models.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`
- Create: `backend/alembic/script.py.mako`
- Create: `backend/alembic/versions/0001_initial.py`

- [ ] **Step 1: Create backend/app/database.py**

```python
import uuid
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from .config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 2: Create backend/app/models.py**

```python
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status = Column(String, nullable=False, default="pending")
    error_msg = Column(Text)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536))
```

- [ ] **Step 3: Create backend/alembic.ini**

```ini
[alembic]
script_location = alembic
prepend_sys_path = .
sqlalchemy.url =

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
```

- [ ] **Step 4: Create backend/alembic/env.py**

```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context
from app.config import settings
from app.database import Base
import app.models  # noqa: F401 — registers models with Base.metadata

config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 5: Create backend/alembic/script.py.mako**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

revision: str = ${repr(up_revision)}
down_revision: Union[str, None] = ${repr(down_revision)}
branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}

def upgrade() -> None:
    ${upgrades if upgrades else "pass"}

def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 6: Create backend/alembic/versions/0001_initial.py**

```python
"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-20
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("file_name", sa.String(), nullable=False),
        sa.Column("file_path", sa.String(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("status", sa.String(), nullable=False, server_default="pending"),
        sa.Column("error_msg", sa.Text()),
    )
    op.create_table(
        "document_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536)),
    )
    op.execute("CREATE INDEX ON document_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

def downgrade() -> None:
    op.drop_table("document_chunks")
    op.drop_table("documents")
```

- [ ] **Step 7: Run migration against the local db service**

Start the db service:
```bash
docker compose up db -d
```

Then run the migration from `backend/`:
```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot \
OPENAI_API_KEY=test \
alembic upgrade head
```

Expected output ends with:
```
INFO  [alembic.runtime.migration] Running upgrade  -> 0001, initial schema
```

- [ ] **Step 8: Verify tables exist**

```bash
docker compose exec db psql -U chatbot -d chatbot -c "\dt"
```

Expected: shows `documents` and `document_chunks`.

- [ ] **Step 9: Commit**

```bash
git add backend/
git commit -m "feat: add SQLAlchemy models and Alembic migration for documents + chunks"
```

---

## Task 4: Document upload endpoint + tests

**Files:**
- Create: `backend/app/schemas.py`
- Create: `backend/app/routers/documents.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_upload.py`

- [ ] **Step 1: Create backend/app/schemas.py**

```python
from pydantic import BaseModel
from uuid import UUID

class DocumentResponse(BaseModel):
    id: UUID
    file_name: str
    status: str

    model_config = {"from_attributes": True}
```

- [ ] **Step 2: Create backend/app/routers/documents.py** (upload endpoint only — SSE added in Task 5)

```python
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

    from ..workers.tasks import ingest_document
    ingest_document.delay(str(doc.id))

    return doc
```

- [ ] **Step 3: Create backend/tests/__init__.py** (empty)

- [ ] **Step 4: Create backend/tests/conftest.py**

```python
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("UPLOAD_DIR", "/tmp/test-uploads")

from app.main import app
from app.database import Base, get_db

TEST_DB_URL = os.environ["DATABASE_URL"]
engine = create_engine(TEST_DB_URL)
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session", autouse=True)
def setup_tables():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db():
    conn = engine.connect()
    trans = conn.begin()
    session = TestingSession(bind=conn)
    yield session
    session.close()
    trans.rollback()
    conn.close()

@pytest.fixture
def client(db):
    def override_get_db():
        yield db
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
```

> **Note:** The test DB (`chatbot_test`) must exist in the running PostgreSQL. Create it once:
> ```bash
> docker compose exec db psql -U chatbot -c "CREATE DATABASE chatbot_test"
> ```

- [ ] **Step 5: Write failing test in backend/tests/test_upload.py**

```python
import io
import pytest

def test_upload_pdf_returns_pending(client, mocker):
    mocker.patch("app.routers.documents.ingest_document.delay")
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
    response = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_name"] == "report.pdf"
    assert data["status"] == "pending"
    assert "id" in data

def test_upload_rejects_unsupported_type(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("script.py", b"print('hi')", "text/x-python")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]

def test_upload_rejects_oversized_file(client, mocker):
    mocker.patch("app.config.settings.max_upload_mb", 0)
    fake_pdf = io.BytesIO(b"%PDF-1.4 " + b"x" * 1025)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("big.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 400
```

- [ ] **Step 6: Run tests — expect failures (router incomplete)**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_upload.py -v
```

Expected: `ImportError` or similar because `ingest_document` task doesn't exist yet. That's expected at this point.

- [ ] **Step 7: Create a stub tasks module so tests can import**

Create `backend/app/workers/celery_app.py`:
```python
from celery import Celery
from ..config import settings

celery_app = Celery(
    "chatbot",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
```

Create `backend/app/workers/tasks.py` (stub — full implementation in Task 7):
```python
from .celery_app import celery_app

@celery_app.task(bind=True, max_retries=1, default_retry_delay=10)
def ingest_document(self, document_id: str):
    pass  # implemented in Task 7
```

- [ ] **Step 8: Run tests — expect them to pass**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_upload.py -v
```

Expected:
```
tests/test_upload.py::test_upload_pdf_returns_pending PASSED
tests/test_upload.py::test_upload_rejects_unsupported_type PASSED
tests/test_upload.py::test_upload_rejects_oversized_file PASSED
3 passed
```

- [ ] **Step 9: Commit**

```bash
git add backend/
git commit -m "feat: add document upload endpoint with file validation"
```

---

## Task 5: SSE status endpoint + tests

**Files:**
- Modify: `backend/app/routers/documents.py`
- Modify: `backend/tests/test_upload.py`

- [ ] **Step 1: Write failing test first — append to test_upload.py**

```python
def test_status_stream_returns_done(client, db):
    from app.models import Document
    import uuid

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="test.pdf", file_path="/tmp/test.pdf", status="done")
    db.add(doc)
    db.flush()

    response = client.get(f"/api/documents/{doc_id}/status")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"status": "done"' in response.text

def test_status_stream_404_for_unknown_id(client):
    import uuid
    response = client.get(f"/api/documents/{uuid.uuid4()}/status")
    assert response.status_code == 404
```

- [ ] **Step 2: Run — expect 404 (endpoint not yet added)**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_upload.py::test_status_stream_returns_done tests/test_upload.py::test_status_stream_404_for_unknown_id -v
```

Expected: `FAILED` — 404 or 405 because the endpoint doesn't exist.

- [ ] **Step 3: Add SSE endpoint to backend/app/routers/documents.py**

Append to `documents.py` after the existing `upload_document` function:

```python
@router.get("/{document_id}/status")
def stream_status(document_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    def event_stream():
        while True:
            db.refresh(doc)
            status = doc.status
            if status == "done":
                data = json.dumps({"status": "done", "message": "Document ready for Q&A."})
                yield f"data: {data}\n\n"
                break
            elif status == "failed":
                data = json.dumps({"status": "failed", "message": doc.error_msg or "Ingestion failed."})
                yield f"data: {data}\n\n"
                break
            else:
                data = json.dumps({"status": status, "message": "Ingesting document..."})
                yield f"data: {data}\n\n"
            time.sleep(2)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

- [ ] **Step 4: Run tests — expect pass**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_upload.py -v
```

Expected: all 5 tests `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/documents.py backend/tests/test_upload.py
git commit -m "feat: add SSE status streaming endpoint for document ingestion"
```

---

## Task 6: Ingestion service — parse, chunk, embed, store

**Files:**
- Create: `backend/app/services/ingestion.py`
- Create: `backend/tests/test_ingestion.py`

- [ ] **Step 1: Write failing tests in backend/tests/test_ingestion.py**

```python
import os
import pytest
from unittest.mock import MagicMock, patch

def test_chunk_text_splits_long_content():
    from app.services.ingestion import chunk_text
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1200  # chunk_size=1000 + some overlap buffer

def test_chunk_text_short_content_stays_one_chunk():
    from app.services.ingestion import chunk_text
    text = "Short paragraph."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Short paragraph."

def test_parse_docx(tmp_path):
    from docx import Document as DocxDocument
    from app.services.ingestion import parse_file
    docx_path = tmp_path / "test.docx"
    doc = DocxDocument()
    doc.add_paragraph("Hello from DOCX")
    doc.save(str(docx_path))
    result = parse_file(str(docx_path), "test.docx")
    assert "Hello from DOCX" in result

def test_parse_image_returns_placeholder():
    from app.services.ingestion import parse_file
    result = parse_file("/any/path/photo.png", "photo.png")
    assert result == "[image: photo.png]"

def test_embed_chunks_calls_openai_and_returns_vectors():
    from app.services.ingestion import embed_chunks
    fake_embedding = [0.1] * 1536
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding)]
    with patch("app.services.ingestion.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.embeddings.create.return_value = mock_response
        result = embed_chunks(["some text"])
    assert len(result) == 1
    assert len(result[0]) == 1536

def test_store_chunks_inserts_rows():
    from app.services.ingestion import store_chunks
    from app.models import DocumentChunk
    import uuid

    mock_db = MagicMock()
    doc_id = str(uuid.uuid4())
    chunks = ["chunk one", "chunk two"]
    embeddings = [[0.1] * 1536, [0.2] * 1536]

    store_chunks(mock_db, doc_id, chunks, embeddings)

    mock_db.bulk_save_objects.assert_called_once()
    saved_objects = mock_db.bulk_save_objects.call_args[0][0]
    assert len(saved_objects) == 2
    assert isinstance(saved_objects[0], DocumentChunk)
    assert saved_objects[0].chunk_index == 0
    assert saved_objects[1].chunk_index == 1
    mock_db.commit.assert_called_once()
```

- [ ] **Step 2: Run — expect ImportError (module doesn't exist)**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_ingestion.py -v
```

Expected: `ModuleNotFoundError: No module named 'app.services.ingestion'`

- [ ] **Step 3: Create backend/app/services/ingestion.py**

```python
import os
from typing import List
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from sqlalchemy.orm import Session

from ..config import settings
from ..models import DocumentChunk

def parse_file(file_path: str, file_name: str) -> str:
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        return f"[image: {file_name}]"

def chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=settings.openai_api_key)
    embeddings: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        response = client.embeddings.create(model="text-embedding-3-small", input=batch)
        embeddings.extend([item.embedding for item in response.data])
    return embeddings

def store_chunks(db: Session, document_id: str, chunks: List[str], embeddings: List[List[float]]) -> None:
    rows = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=i,
            content=chunk,
            embedding=embedding,
        )
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]
    db.bulk_save_objects(rows)
    db.commit()
```

- [ ] **Step 4: Run tests — expect pass**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_ingestion.py -v
```

Expected:
```
tests/test_ingestion.py::test_chunk_text_splits_long_content PASSED
tests/test_ingestion.py::test_chunk_text_short_content_stays_one_chunk PASSED
tests/test_ingestion.py::test_parse_docx PASSED
tests/test_ingestion.py::test_parse_image_returns_placeholder PASSED
tests/test_ingestion.py::test_embed_chunks_calls_openai_and_returns_vectors PASSED
tests/test_ingestion.py::test_store_chunks_inserts_rows PASSED
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ingestion.py backend/tests/test_ingestion.py
git commit -m "feat: add ingestion service (parse, chunk, embed, store)"
```

---

## Task 7: Celery worker task — full implementation

**Files:**
- Modify: `backend/app/workers/celery_app.py` (already created in Task 4 stub)
- Modify: `backend/app/workers/tasks.py` (replace stub)
- Create: `backend/tests/test_tasks.py`

- [ ] **Step 1: Write failing test in backend/tests/test_tasks.py**

```python
import uuid
import pytest
from unittest.mock import patch, MagicMock

def make_doc(doc_id=None, status="pending", file_path="/tmp/test.pdf", file_name="test.pdf"):
    doc = MagicMock()
    doc.id = doc_id or str(uuid.uuid4())
    doc.status = status
    doc.file_path = file_path
    doc.file_name = file_name
    doc.error_msg = None
    return doc

def test_ingest_document_sets_done_on_success():
    from app.workers.tasks import ingest_document
    doc_id = str(uuid.uuid4())
    mock_doc = make_doc(doc_id=doc_id)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch("app.workers.tasks.SessionLocal", return_value=mock_db), \
         patch("app.workers.tasks.parse_file", return_value="some text"), \
         patch("app.workers.tasks.chunk_text", return_value=["chunk1"]), \
         patch("app.workers.tasks.embed_chunks", return_value=[[0.1] * 1536]), \
         patch("app.workers.tasks.store_chunks"):
        ingest_document.apply(args=[doc_id])

    assert mock_doc.status == "done"

def test_ingest_document_sets_failed_on_exception():
    from app.workers.tasks import ingest_document
    doc_id = str(uuid.uuid4())
    mock_doc = make_doc(doc_id=doc_id)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch("app.workers.tasks.SessionLocal", return_value=mock_db), \
         patch("app.workers.tasks.parse_file", side_effect=RuntimeError("parse error")):
        ingest_document.apply(args=[doc_id])

    assert mock_doc.status == "failed"
    assert "parse error" in (mock_doc.error_msg or "")
```

- [ ] **Step 2: Run — expect failures (task is a stub)**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_tasks.py -v
```

Expected: `FAILED` — the stub task does nothing, so `mock_doc.status` stays `"pending"`.

- [ ] **Step 3: Replace the stub in backend/app/workers/tasks.py with full implementation**

```python
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
```

- [ ] **Step 4: Run tests — expect pass**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/test_tasks.py -v
```

Expected:
```
tests/test_tasks.py::test_ingest_document_sets_done_on_success PASSED
tests/test_tasks.py::test_ingest_document_sets_failed_on_exception PASSED
2 passed
```

- [ ] **Step 5: Run full backend test suite**

```bash
cd backend
DATABASE_URL=postgresql+psycopg2://chatbot:chatbot@localhost:5432/chatbot_test \
OPENAI_API_KEY=test \
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/app/workers/tasks.py backend/tests/test_tasks.py
git commit -m "feat: implement ingest_document Celery task with retry on failure"
```

---

## Task 8: Vite dev proxy

**Files:**
- Modify: `vite.config.js`

- [ ] **Step 1: Update vite.config.js**

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

- [ ] **Step 2: Verify dev server starts**

```bash
npm run dev
```

Expected: server starts on `http://localhost:5173` (or similar), no errors. Ctrl-C.

- [ ] **Step 3: Commit**

```bash
git add vite.config.js
git commit -m "feat: proxy /api to backend in Vite dev server"
```

---

## Task 9: useUpload hook

**Files:**
- Create: `src/hooks/useUpload.js`

- [ ] **Step 1: Create src/hooks/useUpload.js**

```js
import { useState, useRef } from 'react'

export function useUpload({ onComplete } = {}) {
  const [state, setState] = useState({
    status: 'idle',   // idle | uploading | processing | done | failed
    fileName: null,
    error: null,
  })
  const eventSourceRef = useRef(null)

  async function uploadFile(file) {
    setState({ status: 'uploading', fileName: file.name, error: null })

    const formData = new FormData()
    formData.append('file', file)

    let docId
    try {
      const res = await fetch('/api/documents/upload', { method: 'POST', body: formData })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      docId = data.id
    } catch (err) {
      setState({ status: 'failed', fileName: file.name, error: err.message })
      return
    }

    setState(s => ({ ...s, status: 'processing' }))

    const es = new EventSource(`/api/documents/${docId}/status`)
    eventSourceRef.current = es

    es.onmessage = (e) => {
      const payload = JSON.parse(e.data)
      if (payload.status === 'done') {
        es.close()
        setState({ status: 'done', fileName: file.name, error: null })
        onComplete?.(file.name)
      } else if (payload.status === 'failed') {
        es.close()
        setState({ status: 'failed', fileName: file.name, error: payload.message })
      }
      // 'processing' events: no state change needed — already set above
    }

    es.onerror = () => {
      es.close()
      setState(s => ({ ...s, status: 'failed', error: 'Connection to server lost.' }))
    }
  }

  function reset() {
    eventSourceRef.current?.close()
    setState({ status: 'idle', fileName: null, error: null })
  }

  return { ...state, uploadFile, reset }
}
```

- [ ] **Step 2: Manual verification note**

This hook is tested implicitly through the Composer integration in Task 11. No unit test framework is installed yet; manual verification is sufficient for this step.

- [ ] **Step 3: Commit**

```bash
git add src/hooks/useUpload.js
git commit -m "feat: add useUpload hook for file upload and SSE status tracking"
```

---

## Task 10: UploadToast component

**Files:**
- Create: `src/components/UploadToast.jsx`
- Create: `src/components/UploadToast.css`

- [ ] **Step 1: Create src/components/UploadToast.jsx**

```jsx
import { useEffect } from 'react'
import './UploadToast.css'

export default function UploadToast({ status, fileName, error, onDismiss }) {
  useEffect(() => {
    if (status !== 'done' && status !== 'failed') return
    const t = setTimeout(onDismiss, 5000)
    return () => clearTimeout(t)
  }, [status, onDismiss])

  if (status !== 'done' && status !== 'failed') return null

  const isDone = status === 'done'

  return (
    <div className={`upload-toast upload-toast--${isDone ? 'done' : 'failed'}`} role="alert">
      <div className="upload-toast-icon" aria-hidden="true">
        {isDone ? (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M2 5l2.5 2.5L8 3" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        ) : (
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <path d="M2 2l6 6M8 2l-6 6" stroke="#fff" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        )}
      </div>
      <div className="upload-toast-body">
        <div className="upload-toast-title">{isDone ? 'Document ready' : 'Ingestion failed'}</div>
        <div className="upload-toast-msg">
          {isDone
            ? `${fileName} has been ingested and is ready for Q&A.`
            : (error || 'Something went wrong.')}
        </div>
      </div>
      <button className="upload-toast-close" onClick={onDismiss} aria-label="Dismiss">✕</button>
    </div>
  )
}
```

- [ ] **Step 2: Create src/components/UploadToast.css**

```css
.upload-toast {
  position: fixed;
  bottom: 20px;
  right: 20px;
  background: #1d1d1f;
  color: #fff;
  border-radius: 12px;
  padding: 12px 14px;
  max-width: 300px;
  box-shadow: rgba(0, 0, 0, 0.25) 0 4px 20px;
  display: flex;
  align-items: flex-start;
  gap: 10px;
  z-index: 1000;
  animation: toast-in 0.2s ease;
}

@keyframes toast-in {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: translateY(0); }
}

.upload-toast-icon {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-top: 1px;
}

.upload-toast--done  .upload-toast-icon { background: #30d158; }
.upload-toast--failed .upload-toast-icon { background: #ff453a; }

.upload-toast-body { flex: 1; min-width: 0; }

.upload-toast-title {
  font-size: 13px;
  font-weight: 600;
  margin-bottom: 2px;
}

.upload-toast-msg {
  font-size: 12px;
  color: rgba(255, 255, 255, 0.65);
  line-height: 1.4;
  word-break: break-word;
}

.upload-toast-close {
  background: none;
  border: none;
  color: rgba(255, 255, 255, 0.4);
  cursor: pointer;
  font-size: 12px;
  padding: 0;
  flex-shrink: 0;
  line-height: 1;
}

.upload-toast-close:hover {
  color: rgba(255, 255, 255, 0.8);
}
```

- [ ] **Step 3: Commit**

```bash
git add src/components/UploadToast.jsx src/components/UploadToast.css
git commit -m "feat: add UploadToast component for ingestion completion notification"
```

---

## Task 11: Composer upload button + ChatPage wiring

**Files:**
- Modify: `src/components/Composer.jsx`
- Modify: `src/components/Composer.css`
- Modify: `src/pages/ChatPage.jsx`

- [ ] **Step 1: Replace src/components/Composer.jsx**

```jsx
import { useState, useRef } from 'react'
import './Composer.css'

export default function Composer({ onSend, onFileSelect, uploadStatus, disabled }) {
  const [value, setValue] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const textareaRef = useRef(null)
  const fileInputRef = useRef(null)

  function handleSubmit(e) {
    e.preventDefault()
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.focus()
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      handleSubmit(e)
    }
  }

  function handleChange(e) {
    setValue(e.target.value)
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 120) + 'px'
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setSelectedFile(file)
    onFileSelect(file)
    e.target.value = ''
  }

  function handleRemoveFile() {
    setSelectedFile(null)
  }

  const isUploading = uploadStatus === 'uploading' || uploadStatus === 'processing'
  const canSend = value.trim().length > 0 && !disabled

  return (
    <div className="composer-wrapper">
      {selectedFile && (
        <div className="composer-chip-row">
          <div className="composer-chip">
            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <rect x="2" y="1" width="10" height="14" rx="2" stroke="#0071e3" strokeWidth="1.5" />
              <path d="M5 5h6M5 8h4" stroke="#0071e3" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <span className="composer-chip-name">{selectedFile.name}</span>
            {isUploading && <span className="composer-chip-spinner" aria-label="Uploading" />}
            {!isUploading && (
              <button className="composer-chip-remove" onClick={handleRemoveFile} aria-label="Remove file">✕</button>
            )}
          </div>
        </div>
      )}
      <form className="composer" onSubmit={handleSubmit}>
        <div className="composer-model-icon" aria-hidden="true">
          <svg width="18" height="18" viewBox="0 0 18 18">
            <defs>
              <linearGradient id="mg" x1="0" y1="0" x2="1" y2="1">
                <stop offset="0%" stopColor="#0071e3" />
                <stop offset="100%" stopColor="#2997ff" />
              </linearGradient>
            </defs>
            <circle cx="9" cy="9" r="9" fill="url(#mg)" />
            <circle cx="9" cy="9" r="4" fill="white" opacity="0.85" />
          </svg>
        </div>
        <textarea
          ref={textareaRef}
          className="composer-input"
          placeholder="What's in your mind?..."
          value={value}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={disabled}
        />
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.docx,image/png,image/jpeg,image/webp"
          className="composer-file-input"
          onChange={handleFileChange}
          aria-label="Upload file"
        />
        <button
          type="button"
          className={`composer-upload ${isUploading ? 'composer-upload--busy' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          disabled={isUploading}
          aria-label="Upload file"
          title="Upload PDF, DOCX, or image"
        >
          {isUploading ? (
            <span className="composer-upload-spinner" />
          ) : (
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
              <path d="M2 11v2a1 1 0 001 1h10a1 1 0 001-1v-2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <path d="M8 2v8M5 5l3-3 3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>
        <button
          type="submit"
          className={`composer-send ${canSend ? 'composer-send--active' : ''}`}
          disabled={!canSend}
          aria-label="Send"
        >
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d="M9 14V4M9 4L5 8M9 4L13 8" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </form>
    </div>
  )
}
```

- [ ] **Step 2: Add upload button + chip styles to src/components/Composer.css**

Append to the existing `Composer.css`:
```css
.composer-file-input {
  display: none;
}

.composer-upload {
  flex-shrink: 0;
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background: #f5f5f7;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  color: rgba(0, 0, 0, 0.55);
  transition: background 0.15s;
}

.composer-upload:hover:not(:disabled) {
  background: #e8e8ed;
}

.composer-upload--busy {
  cursor: not-allowed;
  opacity: 0.6;
}

.composer-upload-spinner,
.composer-chip-spinner {
  display: inline-block;
  width: 12px;
  height: 12px;
  border: 1.5px solid rgba(0, 113, 227, 0.25);
  border-top-color: #0071e3;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.composer-chip-row {
  padding: 0 14px 6px;
}

.composer-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: #f5f5f7;
  border-radius: 8px;
  padding: 5px 10px;
  font-size: 12px;
  color: #1d1d1f;
}

.composer-chip-name {
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.composer-chip-remove {
  background: none;
  border: none;
  color: rgba(0, 0, 0, 0.35);
  cursor: pointer;
  font-size: 11px;
  padding: 0;
  line-height: 1;
}

.composer-chip-remove:hover {
  color: rgba(0, 0, 0, 0.6);
}
```

- [ ] **Step 3: Update src/pages/ChatPage.jsx to wire useUpload + UploadToast**

Replace the full `ChatPage.jsx` file:

```jsx
import { useReducer, useCallback, useState } from 'react'
import Sidebar from '../components/Sidebar'
import ChatThread from '../components/ChatThread'
import Composer from '../components/Composer'
import UpgradeTab from '../components/UpgradeTab'
import UploadToast from '../components/UploadToast'
import { useUpload } from '../hooks/useUpload'
import './ChatPage.css'

const CANNED_REPLIES = [
  `Sure, I can help you get started with creating a chatbot using GPT in Python. Here are the basic steps you'll need to follow:

1. **Install the required libraries:** You'll need to install the transformers library from Hugging Face to use GPT. You can install it using pip.

2. **Load the pre-trained model:** GPT comes in several sizes and versions, so you'll need to choose the one that fits your needs. You can load a pre-trained GPT model. This loads the 1.3B parameter version of GPT-Neo, which is a powerful and relatively recent model.

3. **Create a chatbot loop:** You'll need to create a loop that takes user input, generates a response using the GPT model, and outputs it to the user. Here's an example loop that uses the input() function to get user input and the gpt() function to generate a response.

4. **Add some personality to the chatbot:** While GPT can generate text, it doesn't have any inherent personality or style. You can make your chatbot more interesting by adding custom prompts or responses that reflect your desired personality.

These are just the basic steps to get started with a GPT chatbot in Python. Depending on your requirements, you may need to add more features or complexity to the chatbot. Good luck!`,
  `Chatbots can be used for a wide range of purposes, including:

**Customer service:** Chatbots can handle frequently asked questions, provide basic support, and help customers resolve issues without human intervention.

**E-commerce:** Chatbots can assist users in finding products, tracking orders, and completing purchases within a conversational interface.

**Healthcare:** Medical chatbots can provide symptom checking, appointment scheduling, and medication reminders to patients.

**Education:** Chatbots can deliver personalized learning experiences, answer student questions, and provide tutoring support around the clock.`,
  `That's a great question! Here's what you should know:

The key difference lies in how each approach handles state and side effects. When building scalable systems, it's generally better to start with a simple architecture and only introduce complexity when the need arises.

A few things to keep in mind:
1. Start with the simplest solution that works
2. Measure before optimizing
3. Document your architectural decisions
4. Review and refactor regularly`,
]

let nextId = 1
function makeId() { return nextId++ }

const SEED_MESSAGES = [
  { id: makeId(), role: 'user', content: 'Create a chatbot gpt using python language what will be step for that' },
  { id: makeId(), role: 'assistant', content: CANNED_REPLIES[0] },
  { id: makeId(), role: 'user', content: 'What is use of that chatbot ?' },
  { id: makeId(), role: 'assistant', content: CANNED_REPLIES[1] },
]

function messagesReducer(state, action) {
  switch (action.type) {
    case 'ADD_MESSAGE':
      return { ...state, messages: [...state.messages, action.payload], pending: false }
    case 'SET_PENDING':
      return { ...state, pending: true }
    default:
      return state
  }
}

export default function ChatPage() {
  const [state, dispatch] = useReducer(messagesReducer, {
    messages: SEED_MESSAGES,
    pending: false,
  })
  const [showToast, setShowToast] = useState(false)

  const upload = useUpload({
    onComplete: () => setShowToast(true),
  })

  const handleSend = useCallback((text) => {
    const userMsg = { id: makeId(), role: 'user', content: text }
    dispatch({ type: 'ADD_MESSAGE', payload: userMsg })
    dispatch({ type: 'SET_PENDING' })
    const reply = CANNED_REPLIES[Math.floor(Math.random() * CANNED_REPLIES.length)]
    setTimeout(() => {
      dispatch({ type: 'ADD_MESSAGE', payload: { id: makeId(), role: 'assistant', content: reply } })
    }, 500)
  }, [])

  return (
    <div className="chat-page-outer">
      <div className="chat-page-card">
        <Sidebar activeTitle="Create Chatbot GPT..." />
        <div className="chat-main">
          <ChatThread messages={state.messages} pending={state.pending} />
          <Composer
            onSend={handleSend}
            onFileSelect={upload.uploadFile}
            uploadStatus={upload.status}
            disabled={state.pending}
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

- [ ] **Step 4: Start the dev server and manually verify the UI**

```bash
npm run dev
```

Open `http://localhost:5173`. Verify:
1. The upload button (↑ icon) appears to the left of the send button in the composer.
2. Clicking it opens the system file picker — accepts PDF, DOCX, PNG, JPEG, WEBP.
3. Selecting a file shows the filename chip below the composer with a spinner.
4. Without a running backend, the upload fails gracefully — no crash, no broken layout.

- [ ] **Step 5: Run ESLint**

```bash
npm run lint
```

Expected: no errors. Fix any `no-unused-vars` or import issues before committing.

- [ ] **Step 6: Commit**

```bash
git add src/
git commit -m "feat: add file upload button, chip, and toast notification to chat UI"
```

---

## Task 12: End-to-end smoke test

This task verifies all five services work together.

- [ ] **Step 1: Create a local .env file from the template**

```bash
cp .env.example .env
```

Edit `.env`: fill in your `OPENAI_API_KEY`. The DB credentials can stay as the defaults.

- [ ] **Step 2: Build and start all services**

```bash
docker compose up --build -d
```

- [ ] **Step 3: Run the Alembic migration**

```bash
docker compose exec backend alembic upgrade head
```

Expected: `Running upgrade  -> 0001, initial schema`

- [ ] **Step 4: Verify health endpoint**

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok"}`

- [ ] **Step 5: Upload a small PDF via curl**

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@/path/to/small.pdf;type=application/pdf"
```

Expected response (note the `id`):
```json
{"id":"<uuid>","file_name":"small.pdf","status":"pending"}
```

- [ ] **Step 6: Watch SSE stream**

```bash
curl -N "http://localhost:8000/api/documents/<uuid>/status"
```

Expected: within ~30 seconds (depending on doc size) you should see:
```
data: {"status": "processing", "message": "Ingesting document..."}

data: {"status": "done", "message": "Document ready for Q&A."}
```

- [ ] **Step 7: Verify chunks in database**

```bash
docker compose exec db psql -U chatbot -d chatbot \
  -c "SELECT chunk_index, left(content,60) FROM document_chunks ORDER BY chunk_index LIMIT 5;"
```

Expected: rows with chunk text.

- [ ] **Step 8: Open the frontend and test end-to-end**

Open `http://localhost:3000`. Upload the same PDF — the chip should appear, spin, then the green toast should appear.

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "feat: complete file upload + RAG ingestion pipeline (end-to-end verified)"
```

---

## Self-Review Checklist

**Spec coverage:**
| Spec requirement | Covered by task |
|-----------------|----------------|
| Upload button in composer (PDF/DOCX/image) | Task 11 |
| Store metadata (id, file_name, path, uploaded_date) to PostgreSQL | Tasks 3, 4 |
| Configurable connection string | Tasks 1, 2 (env vars) |
| Background job: chunking | Tasks 6, 7 |
| Background job: embedding (OpenAI) | Tasks 6, 7 |
| Background job: store to pgvector | Tasks 3, 6, 7 |
| Notify user on completion (SSE) | Tasks 5, 9, 10, 11 |
| docker-compose with all 5 services | Task 1 |
| Error handling (bad file type, size, API error, worker crash) | Tasks 4, 7 (+ spec section 8) |

All spec requirements are covered. ✓
