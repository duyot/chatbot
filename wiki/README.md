# Chatbot — Document Q&A Wiki

**What it does**: Users upload a document (PDF, DOCX, PNG/JPEG/WebP), the system OCRs/parses and chunks it, and users can then ask questions about it in a chat interface that streams grounded, page-cited answers via an agentic retrieval-augmented-generation (RAG) pipeline.
**Who uses it**: Authenticated end users (no self-service signup — accounts are provisioned via a CLI script) chatting against their own uploaded documents.
**Frontend**: React 19 + Vite (JavaScript/JSX, no TypeScript), `react-router-dom` v7, served by nginx in production.
**Backend**: FastAPI + SQLAlchemy + Celery/Redis, Postgres 18 via ParadeDB (bundling `pg_search` for BM25 alongside `pgvector`) for hybrid keyword + vector search, and a LangGraph state machine for the agentic RAG pipeline.
**External services**: OpenRouter (chat LLM, embeddings, reranker) and an in-house RapidOCR microservice.

## Pages

1. [Architecture](01-architecture.md) — component diagram, responsibilities, entry points.
2. [Flows](02-flows.md) — sequence diagrams for auth, upload/ingestion, chat/RAG, conversation history, and startup.
3. [Data Model](03-data-model.md) — ERD and table-by-table read/write ownership.
4. [Integrations](04-integrations.md) — OpenRouter (chat/embeddings/rerank), OCR microservice, Redis/Celery — auth and failure handling.
5. [Setup](05-setup.md) — prerequisites, env vars, run commands, how to verify it's working.

## Not covered by this wiki

- `.worktrees/`, `.understand-anything/`, `dist/`, vendored `.venv`/`node_modules` dependencies, and other generated/vendored content — skipped per scope.
- `features_planning/*` and `docs/superpowers/*` — in-progress planning docs, not the shipped implementation; not traced as source of truth here.
- `DESIGN.md` — a visual/UI design-system reference (colors, typography, spacing), not a code-architecture document.
- Component-level CSS files (`*.css`) — styling only, not structural/behavioral.
- `backend/evals/` internals and the ragas-based eval methodology — mentioned in [Setup](05-setup.md) as a command to run, but the eval harness itself wasn't traced in depth.
- Fine-grained frontend UI polish components (`UpgradeTab.jsx`, toast/thinking-indicator animation timing) — noted in [Flows](02-flows.md) only where they intersect a traced flow.
- No CI/CD configuration was found in the repo to document (`docker-compose.yml` + Dockerfiles are the only deployment artifacts present).
