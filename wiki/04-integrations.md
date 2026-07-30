# Integrations

All external calls in this system funnel through **OpenRouter** (chat, embeddings, rerank) or the **in-house OCR microservice**; the only queue is **Redis/Celery**. There is no direct OpenAI, Anthropic, or other third-party API call outside OpenRouter's OpenAI-compatible surface.

## OpenRouter — chat LLM

- **What's called**: `POST {OPENROUTER_BASE_URL}/chat/completions` (OpenAI-compatible), via `langchain_openai.ChatOpenAI` in `backend/app/services/rag/nodes.py:20-26` (`_chat_llm()`), model = `settings.openrouter_chat_model` (default `anthropic/claude-haiku-4.5`).
- **Where used**: every LLM step in the RAG graph — `rewrite_query`, `rewrite_and_retry`, `grade_chunks` (strict path only), `generate_answer`, `faithfulness_check` (`backend/app/services/rag/nodes.py`).
- **Auth**: `api_key=settings.openrouter_api_key` (env `OPENROUTER_API_KEY`), passed as a Bearer token by the OpenAI SDK under the hood.
- **Failure handling**: none explicit at the call site — an exception raised by `ChatOpenAI.ainvoke`/`.with_structured_output` propagates up through the LangGraph node, is caught by `backend/app/routers/chat.py:105-108`, which yields an SSE `{"type": "error", ...}` event and skips persisting an assistant message. No retry/backoff around chat completions.

## OpenRouter — embeddings

- **What's called**: `POST {OPENROUTER_BASE_URL}/embeddings` (OpenAI-compatible), via a raw `openai` SDK client in `backend/app/services/ingestion.py` (`_openai_client()`, `embed_chunks()`) and `backend/app/services/rag/retrieval.py` (`embed_text`, imported from `ingestion.py`, used to embed the search query in `hybrid_search()`).
- **Model / shape**: `settings.openai_embedding_model` (default `qwen/qwen3-embedding-8b`), with `dimensions=settings.embedding_dim` (1536) to truncate matryoshka-style embeddings down to the `document_chunks.embedding` column width. Ingestion batches 100 chunks per call (`ingestion.py:237-254`).
- **Auth**: same `OPENROUTER_API_KEY` / base URL as chat.
- **Failure handling**: no explicit try/except around the embedding call in `embed_chunks()` — an error here propagates to `ingest_document`'s outer try/except (`backend/app/workers/tasks.py`), which sets `Document.status="failed"` + `error_msg` and retries the whole Celery task once (`max_retries=1`).

## OpenRouter — reranker

- **What's called**: `POST {OPENROUTER_BASE_URL}/rerank`, via raw `httpx.Client` in `backend/app/services/rag/reranker.py:34-96` (`rerank()`) — a dedicated cross-encoder endpoint, not `/chat/completions`.
- **Model**: `settings.reranker_model` (default `anthropic/claude-haiku-4.5`; the module docstring documents `nvidia/llama-nemotron-rerank-vl-1b-v2:free` as the intended dedicated-reranker override, with scores roughly in `[0, 1]`).
- **Request**: `{"model": ..., "query": ..., "documents": [{"text": chunk.content}], "top_n": ...}`. **Response**: `{"results": [{"index": int, "relevance_score": float}]}`.
- **Auth**: same `OPENROUTER_API_KEY`.
- **Failure handling**: explicit try/except (`reranker.py:63-72`) — any exception (timeout, HTTP error, bad JSON) is logged and the function falls back to `[(chunk, 0.0) for chunk in chunks[:top_n]]`, degrading gracefully to plain RRF-fusion order instead of failing the whole chat request. 60s client timeout. The module docstring also records two previously-broken integration attempts (misusing `/chat/completions` structured output, and misusing Ollama's `/api/embed`) as history for future maintainers.

## OCR microservice (in-house, `ocr-service/`)

- **What's called**: `POST {OCR_SERVICE_URL}/ocr` (multipart image upload) from `backend/app/services/ocr_client.py:27-61` (`ocr_image()`), called by `backend/app/services/ingestion.py` only for PDF pages under the native-text-layer threshold (`OCR_MIN_TEXT_CHARS`, default 20 chars) and for all image uploads. Health-checked at `GET /health` (used by `docker-compose.yml`'s healthcheck, not called by the backend at runtime).
- **Auth**: none — internal-network-only service (`ocr` hostname, port 8080, not exposed to the host in `docker-compose.yml`).
- **Config**: `OCR_ENABLED` (gate), `OCR_SERVICE_URL` (default `http://ocr:8080`), `OCR_TIMEOUT_S` (default 60s), `OCR_DPI` (rasterization resolution for scanned PDF pages, default 200).
- **Failure handling**: `ocr_client.py` raises a dedicated `OCRError` on non-2xx/timeout; `services/ingestion.py` catches this **per-page** and degrades just that page to empty text rather than failing the whole document — a single bad scan doesn't block ingestion of the rest of the document.

## Redis — Celery broker + result backend

- **What's called**: standard Celery protocol over `settings.redis_url` (`redis://redis:6379/0`), configured in `backend/app/workers/celery_app.py:9-17`. Used for exactly one task type, `ingest_document` (`backend/app/workers/tasks.py`), dispatched via `.delay()` from `backend/app/routers/documents.py:68`.
- **Auth**: none (unauthenticated Redis, internal docker network only).
- **Failure handling**: task-level retry (`max_retries=1`, `default_retry_delay=10s`) via `self.retry(exc=exc)` inside `ingest_document`'s except block; a second failure raises `MaxRetriesExceededError`, which is caught and swallowed so the task doesn't dead-letter loudly (`tasks.py:56-62`). No dead-letter queue or alerting is configured. Celery's result backend (also Redis) is configured but not polled anywhere in the app — ingestion progress is tracked via the `documents.status` DB column instead, not Celery task results.

## Not integrated (for context)
- `OPENAI_API_KEY` (`.env.example`) is present for forward-compatibility only — no code path currently calls OpenAI directly; all chat/embedding/rerank traffic goes through OpenRouter.
- No external logging/monitoring/APM service is integrated — logs go to rotating local files (`/app/logs/backend.log`, `/app/logs/worker.log`, `/app/logs/ai_trace.jsonl`) and stdout only.
- **AI trace log** (`backend/app/observability.py`): a third sink, `/app/logs/ai_trace.jsonl`, holds one JSON object per event for the ingestion + retrieval + LLM path — retrieved chunk ids/scores, reranker request and response, prompts sent to OpenRouter, token usage, per-stage latency. Every line carries a `trace_id` that also prefixes the corresponding `backend.log` / `worker.log` lines, so one chat request or ingest task can be reconstructed across all three files. Verbosity is `settings.ai_trace_level` (`off` / `summary` / `full`); at `full` the file contains complete document text, so treat it as sensitive. Rotating, 10 MB × 5, same `logs/` bind mount as the others.
