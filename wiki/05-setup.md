# Local Setup

## Prerequisites
- Docker + Docker Compose (runs all 5 services: `frontend`, `backend`, `worker`, `ocr`, `redis`, `db`).
- An OpenRouter API key (`OPENROUTER_API_KEY`) — required for chat, embeddings, and reranking; nothing works end-to-end without it.
- For frontend-only local dev (outside Docker): Node.js 22+ (matches `Dockerfile`'s `node:22-alpine` build stage).
- For backend-only local dev (outside Docker): Python 3.12 (matches `backend/Dockerfile`'s `python:3.12-slim`).

## Environment variables (`.env`, copy from `.env.example`)

| Var | Purpose |
|---|---|
| `DATABASE_URL` | Postgres DSN |
| `REDIS_URL` | Celery broker/result backend |
| `OPENAI_API_KEY` | Unused by default (kept for back-compat) |
| `OPENROUTER_API_KEY` | **Required** — auth for chat/embeddings/rerank |
| `OPENROUTER_BASE_URL` | OpenRouter API base (default `https://openrouter.ai/api/v1`) |
| `OPENROUTER_CHAT_MODEL` | Chat LLM slug |
| `OPENAI_EMBEDDING_MODEL` | Embedding model slug (routed via OpenRouter) |
| `EMBEDDING_DIM` | Embedding output dim (must match the `document_chunks.embedding` column, currently 1536) |
| `RERANKER_MODEL` | Reranker model slug |
| `UPLOAD_DIR` | Local dir for uploaded source files |
| `MAX_UPLOAD_MB` | Upload size cap |
| `CORS_ORIGINS` | Comma-separated allowed origins |
| `JWT_SECRET_KEY` | **Required in production** — JWT signing key (`openssl rand -hex 32`) |
| `JWT_ALGORITHM` | JWT signing algorithm (default `HS256`) |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT TTL (default 1440 = 24h) |
| `OCR_ENABLED` | Gate on whether ingestion calls the OCR microservice |
| `OCR_SERVICE_URL` | OCR microservice base URL (default `http://ocr:8080`) |
| `OCR_TIMEOUT_S` | HTTP timeout for OCR calls |
| `OCR_MIN_TEXT_CHARS` | Native-text-layer threshold below which a PDF page is OCR'd |
| `OCR_DPI` | Rasterization DPI for scanned pages |
| `RERANK_NATIVE_BOOST`, `RERANK_LOWCONF_PENALTY`, `RERANK_LOWCONF_THRESHOLD` | Metadata-aware rerank score adjustments (no-ops by default) |

Retrieval/agent tunables (`vector_top_k`, `fts_top_k`, `rrf_k`, `rerank_top_n`, `rerank_score_floor`, `max_retrieval_retries`, `strict_grader`) have code defaults in `backend/app/config.py` and are not currently in `.env.example` — override via `.env` if needed.

## Run everything with Docker Compose

```bash
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY at minimum; set JWT_SECRET_KEY for anything beyond local dev
docker compose up --build
```

This builds and starts, in dependency order: `db` (Postgres+pgvector) → `redis` → `ocr` → `backend` (runs `alembic upgrade head` then `uvicorn`) → `worker` (Celery) → `frontend` (nginx, serves the built SPA and proxies `/api/`).

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000 (also reachable via the frontend's `/api/` proxy)
- Postgres: exposed on host port 5434 (mapped from container 5432, per `docker-compose.yml`)

## Create a login account

There is no signup endpoint — accounts are created via a CLI script:

```bash
docker compose exec backend python -m scripts.create_user <username> <password>
```

(see `backend/scripts/create_user.py` for the exact CLI signature).

## Frontend-only dev loop (faster iteration, hits a running backend)

```bash
npm install
npm run dev       # Vite dev server with HMR; proxies /api -> http://localhost:8000 (vite.config.js)
```

Requires the backend (and its dependencies) already running, e.g. via `docker compose up backend worker db redis ocr`.

## Verify it's working

1. `curl http://localhost:8000/health` → `{"status": "ok"}` (backend, `backend/app/main.py:46-48`).
2. `curl http://localhost:8080/health` → `{"status": "ok"}` (OCR service, only reachable from inside the compose network unless port-forwarded).
3. Open http://localhost:3000, log in with a user created via `create_user`, upload a PDF/DOCX/image, and wait for the upload toast to report ingestion `done` (driven by the `/api/documents/{id}/status` SSE stream).
4. Ask a question about the uploaded document in the chat box — a streamed answer with page citations confirms the full pipeline (upload → OCR/parse → chunk → embed → hybrid retrieve → rerank → generate) is working.
5. `docker compose logs -f worker` / `backend` (or tail `logs/worker.log`, `logs/backend.log`) to debug ingestion or chat failures.

## Backend test suite & eval harness

```bash
cd backend
pytest                                   # unit/integration tests (real Postgres via conftest.py, eval marker excluded by default)
python -m evals.run_eval --name <name>   # golden-set RAG quality eval (backend/evals/golden_set.yaml)
python -m evals.run_eval --compare <baseline_name> <name>
```

`backend/tests/conftest.py` refuses to run unless the target DB name contains `"test"` (it does a full `drop_all` at teardown) — point `DATABASE_URL` at a dedicated test database, not the dev one.
