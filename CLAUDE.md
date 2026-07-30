# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project State

A working document-Q&A app, **not** a scaffold. Four services under `docker-compose.yml`: a React SPA behind nginx (`src/`, `Dockerfile`, `nginx.conf`), a FastAPI backend (`backend/app/`), a Celery ingestion worker (`backend/app/workers/`), and a RapidOCR sidecar (`ocr-service/`), against Postgres 18 + ParadeDB and Redis. See `wiki/` for the architecture, flows, and data model.

## Stack

- **React 19.2** with `StrictMode` (see `src/main.jsx`)
- **Vite 8** as build tool and dev server (`vite.config.js` only registers `@vitejs/plugin-react`)
- **ESLint 9** flat config (`eslint.config.js`) — uses `@eslint/js` recommended, `eslint-plugin-react-hooks`, and `eslint-plugin-react-refresh` (Vite variant)
- **JavaScript (JSX)**, not TypeScript. No `tsconfig.json`, no type tooling.
- **No frontend test framework.** Adding one means picking and installing it (e.g. Vitest + React Testing Library, or Playwright for E2E). The backend has pytest (`backend/tests/`, config in `backend/pytest.ini`).

## Commands

```bash
npm run dev       # Start Vite dev server with HMR
npm run build     # Production build to dist/
npm run preview   # Preview the production build
npm run lint      # ESLint over the whole repo
```

To lint a single file: `npx eslint path/to/file.jsx`.

## ESLint notes

The config in `eslint.config.js` has one non-default rule worth knowing about:

```js
'no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]' }]
```

Unused variables are errors **unless** they start with an uppercase letter or underscore. This exists so unused imports of React component types / constants don't break the lint, but unused lowercase locals will fail CI/lint.

## Architecture

Entry path is standard Vite:

- `index.html` → loads `/src/main.jsx`
- `src/main.jsx` mounts `<App />` inside `StrictMode` on `#root`
- `src/App.jsx` is the current (scaffold) root component
- `public/` holds static assets served from `/` (e.g. `/icons.svg`, `/favicon.svg`) — reference them by absolute path in JSX, not via `import`

When the chatbot UI is built, follow the common layout that matches the existing rules:
- Keep components in `src/` split into small files (≤400 lines); extract hooks into their own files.
- Components that fetch or mutate data should go through a repository-style abstraction rather than calling `fetch` inline — see the repository pattern in the user's global rules.

## Working in this repo

- This is a **`.jsx` project, not `.tsx`**. The global TypeScript rules in the user's config describe the preferred style, but apply them via JSDoc annotations where helpful rather than adding TypeScript unilaterally. Ask before introducing TypeScript, since it would mean adding `tsconfig.json`, a type-aware ESLint setup, and converting files.
- There is no `.env` handling wired up. Any API keys (OpenAI, Anthropic, etc.) for the chatbot must go through `import.meta.env.VITE_*` with Vite's env loading — do not hardcode.
- Vite config is intentionally minimal. Before adding plugins (proxy, path aliases, PWA, etc.), discuss the trade-off first.
- There is no Git repo initialized in this working directory and no CI config. Don't assume either exists when writing commands or hooks.

## Backend — RAG pipeline

The agentic RAG service lives in `backend/app/services/rag/` and is a LangGraph state machine. Entry point: `agentic_rag_stream(document_id, message, db)`. Six nodes: rewrite_query, retrieve_and_rerank, grade_chunks, rewrite_and_retry, generate_answer, faithfulness_check. Chat LLM is `anthropic/claude-haiku-4.5` via OpenRouter; embeddings are `qwen/qwen3-embedding-8b` truncated to 1536 dims (via `dimensions=`) and routed through OpenRouter's OpenAI-compatible `/v1/embeddings`.

Retrieval is hybrid: pgvector cosine similarity plus a keyword arm over
`search_text` (`context || content`), fused with **weighted** RRF (0.8 semantic
/ 0.2 keyword, both in settings — untuned defaults, not measured; see the eval
note below). The keyword arm is ParadeDB `pg_search` BM25 when the extension is
available and falls back to Postgres `ts_rank` otherwise — detected once per
process by `retrieval.bm25_available()`. Results are reranked via OpenRouter's
`/v1/rerank` cross-encoder on context + content (default
`anthropic/claude-haiku-4.5`; set `nvidia/llama-nemotron-rerank-vl-1b-v2:free`
for a dedicated reranker), then we return parent chunks (~1500 tokens) to the
LLM while children (~300 tokens) are what gets retrieved.

At ingestion, `services/contextualizer.py` generates a context string per child
chunk situating it in its source document, sending the document as a 1-hour
prompt-cached block. **The first call must complete before the rest fan out**
across `contextualizer_max_workers` threads — a cache entry is only readable
once the first response starts streaming, so a concurrent fan-out would make
every call pay full input price instead of a cached rate. Documents over
`contextualizer_full_doc_token_limit` (100k tokens) fall back to a generated
doc summary plus the child's own page. Per-chunk failures are non-fatal.
Disable with `contextual_embeddings_enabled=False`. See
`docs/superpowers/specs/2026-07-28-contextual-retrieval-design.md`.

See `features/chat_with_doc/rag_enhancement.md` for the flow diagram and `docs/superpowers/specs/2026-05-15-agentic-rag-enhancement-design.md` for the design rationale.

### Status string

The Document model uses `status="done"` for fully-ingested documents (used by `/api/documents` filter at `backend/app/routers/documents.py`). Do not write `status="ready"` — that's a spec inconsistency, "done" is the production value.

### Evaluation

Before changing retrieval or agent code, run the eval harness:

```bash
cd backend
python -m evals.run_eval --name <name>
python -m evals.run_eval --compare <baseline_name> <name>
```

Golden set: `backend/evals/golden_set.yaml`. The `@pytest.mark.eval` marker excludes the golden-set test from default `pytest` runs (`addopts = -m "not eval"` in `backend/pytest.ini`).

### Document preview (page images)

PDF previews are **server-rendered page images**, not client-side pdf.js. At
ingestion `services/page_images.py` rasterizes each PDF page with PyMuPDF into
`uploads/pages/{document_id}/{page:04d}.webp` (150 DPI, WebP q80 via Pillow —
PyMuPDF cannot encode WebP itself). Rendering is **idempotent** (existing pages
are reused) and **non-fatal**: a failed render never blocks `status="done"`,
because previews are cosmetic and Q&A is the product.

The UI reads `GET /api/documents/{id}/pages` for a manifest of
`{page, width, height}`, then fetches each image from
`GET /api/documents/{id}/pages/{n}`. **The manifest endpoint renders missing
pages on demand**, so documents ingested before this feature self-heal on first
preview — no backfill is required. To pre-warm them anyway, or to re-render
after changing `page_image_dpi`/`page_image_format`:

```bash
cd backend
python -m scripts.render_pages_all            # skip already-rendered pages
python -m scripts.render_pages_all --force    # delete and re-render
```

Frontend side is `src/hooks/usePageImages.js` + `src/components/PageImageViewer.jsx`.
Two constraints worth knowing before changing them:
- Page images are fetched via `apiFetch` into object URLs, **not** a bare
  `<img src>` — the route is authenticated and an `<img>` cannot send the
  `Authorization` header.
- `PageImageViewer` is rendered with `key={document.id}`. That's what lets
  `usePageImages` hold zero reset logic; React 19's lint rules reject the
  reset-in-effect and mutate-ref-during-render alternatives.

Citation highlighting needs no DPI-specific handling: `DocumentChunk.bbox` rects
are normalized to the source page rect, and a rendered page is geometrically
similar to it at any DPI, so the same rects overlay the image directly.

Only PDFs get page images. Images are previewed from the source file; DOCX falls
back to a download link.

### Reingestion after schema changes

After a chunk-schema migration (e.g. parent-child), run `python -m scripts.reingest_all` (or `--doc-id <uuid>` for a single document) to rebuild chunks from source documents in `uploads/`.

Migration `0010` added `context`; there is **no backfill**. Documents ingested
before it retrieve on content alone until re-ingested. Run
`python -m scripts.reingest_all` to contextualize them.