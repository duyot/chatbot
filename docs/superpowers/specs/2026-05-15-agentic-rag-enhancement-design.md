# Agentic RAG Enhancement — Design Spec

- **Date:** 2026-05-15
- **Status:** Approved — superseded in part on 2026-06-28 (see Historical note)
- **Supersedes (partially):** `2026-05-04-agentic-rag-design.md` (v1 feature design — this enhances its retrieval and agent loop)
- **Target module:** `backend/app/services/rag.py` → `backend/app/services/rag/`

> **Historical note (2026-06-28):** The stack moved off self-hosted Ollama/TEI.
> Chat LLM is now `anthropic/claude-haiku-4.5` via OpenRouter, embeddings are
> OpenAI `text-embedding-3-small` (1536d), and the reranker is LLM-as-reranker
> via OpenRouter (no more bge-reranker-v2-m3 cross-encoder). The retrieval flow
> (hybrid + RRF + rerank + parent-child) described below is unchanged; only
> the model substrate moved. Content below is the original 2026-05-15 design.

## 1. Problem statement

The current agentic RAG implementation in `backend/app/services/rag.py` fails to answer questions whose answers are **directly stated in the source document**. Three root causes:

1. **Chunk granularity mismatch.** Flat 1000-token chunks (with 200 overlap) dilute the embedding signal of any single fact, so a chunk containing the exact answer ranks below loosely-related chunks.
2. **Brittle query preprocessing.** `_preprocess_query` uses regex to strip question framing. It only handles a small set of prefixes/suffixes, breaks on phrasing variations, and cannot resolve pronouns or context from prior turns.
3. **Weak agent loop.** Up to 3 `bind_tools` rounds with no relevance grading, no retry strategy, no rerank, and no faithfulness check. The agent has no way to recognise that retrieval failed and rephrase.

## 2. Goals

- Fix the "answer is in the doc but agent misses it" failure mode.
- Replace the regex query preprocessor with an LLM-based rewriter that handles arbitrary phrasing.
- Add quantitative evaluation so future changes can be validated.
- Stay **local-only** — no paid APIs. Open-source models only, runnable on the existing Ollama stack.
- Keep the module maintainable: small files (<200 lines each), single responsibility per file.

## 3. Non-goals

- Multi-document chat. We keep "chat with one selected document" semantics.
- Multi-hop reasoning across documents. Single-document, single-hop is the target.
- Query decomposition with parallel sub-agents (the GiovanniPasq fan-out pattern). Not needed for single-doc lookup; would triple latency.
- Streaming retry on ungrounded answers. Streamed tokens are committed; we surface a UI warning instead.
- Fine-tuning any models.

## 4. Architecture

### 4.1 Module layout

Replace the single `rag.py` with a package:

```
backend/app/services/rag/
  __init__.py          # public API: agentic_rag_stream()
  graph.py             # LangGraph wiring (state, nodes registered, compile)
  nodes.py             # rewrite_query, retrieve_and_rerank, grade_chunks,
                       # rewrite_and_retry, generate_answer, faithfulness_check
  retrieval.py         # hybrid_search(), rrf_fuse(), rerank(), fetch_parents()
  reranker.py          # FlashRank singleton (lazy-loaded)
  prompts.py           # all system/instruction prompts (one constant each)
  state.py             # AgentState TypedDict + helpers
```

### 4.2 LangGraph state machine

```
START
  │
  ▼
rewrite_query  ─────────► clean, self-contained query (LLM, structured output)
  │
  ▼
retrieve_and_rerank ────► hybrid (vector + FTS) → RRF fuse → cross-encoder
  │                        rerank → fetch parents
  ▼
grade_chunks ───────────► rule-based (rerank score >= 0.05) by default;
  │                        strict LLM grader gated behind a config flag
  │
  ├── has_enough? ──► YES ──► generate_answer ──► faithfulness_check ──► END
  │
  └── NO, retry_count < 2 ──► rewrite_and_retry (new phrasing, dedup
  │                            attempted_queries) ──► retrieve_and_rerank
  │
  └── NO, retries exhausted ──► generate_answer (with "not found" framing) ──► END
```

Hard cap: 2 retries (worst case 3 retrieval rounds — same budget as today).

### 4.3 Agent state

```python
class AgentState(TypedDict):
    question: str
    rewritten_query: str
    attempted_queries: list[str]
    retry_count: int
    retrieved_children: list[DocumentChunk]
    parents: list[DocumentParentChunk]
    graded_useful: bool
    answer: str
    citations: list[dict]
    notes: list[str]
```

## 5. Data model changes

### 5.1 Schema

New table `document_parent_chunks` for large LLM-context chunks. Existing `document_chunks` gains a `parent_id` FK and becomes the child (retrieval) table.

```sql
CREATE TABLE document_parent_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    parent_index    INTEGER NOT NULL,
    content         TEXT NOT NULL,
    UNIQUE (document_id, parent_index)
);
CREATE INDEX ix_dpc_doc ON document_parent_chunks(document_id);

ALTER TABLE document_chunks
    ADD COLUMN parent_id UUID REFERENCES document_parent_chunks(id) ON DELETE CASCADE;
CREATE INDEX ix_dc_parent ON document_chunks(parent_id);
```

The existing FTS GIN index on `document_chunks.content` (`0003_add_fts_gin_index.py`) stays — child chunks remain the keyword-search target.

### 5.2 Chunking strategy

Sizes are in **tokens**, measured via `RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base", ...)`. This is a switch from today's character-based splitter (`chunk_size=1000` characters), so the new ingestion uses token counts consistently.

| Layer  | Size (tokens)  | Overlap (tokens) | Purpose                  |
|--------|----------------|------------------|--------------------------|
| Parent | 1500           | 0                | Fed to LLM for answering |
| Child  | 300            | 50               | Embedded + FTS indexed   |

Splitting algorithm: first split the document into parents (1500/0). For each parent, split into children (300/50). Each child row stores `parent_id` and `chunk_index` (global child index across the document, preserved for citations).

Embedding is computed for children only. Retrieval picks children; we dedup their parent IDs; we feed parent text to the LLM.

### 5.3 Re-ingestion

- Alembic migration `0004_parent_child_chunks.py` creates the new table and column.
- One-shot script `backend/scripts/reingest_all.py` re-runs ingestion for every document where `status='ready'`. Per-document transaction so a crash leaves `status='failed'` rather than partial rows.
- Source PDFs/DOCXs in `uploads/` are the source of truth. No data loss risk.

## 6. Retrieval pipeline (`retrieval.py`)

Four stages, plain functions, no LLM calls in this file.

### 6.1 Hybrid search

Pull `top_k=30` from each leg (today is 8). Vector leg = pgvector cosine. FTS leg = Postgres `ts_rank` with `plainto_tsquery('english', :q)`. Both target `document_chunks` (child) rows.

### 6.2 RRF fusion

Reciprocal Rank Fusion with `k=60` (TREC standard). Replaces the current "vector first, then FTS extras" merge — RRF is rank-based and immune to score-distribution mismatch between cosine and `ts_rank`.

```python
def rrf_fuse(vec_hits, fts_rows, k=60):
    scores = defaultdict(float)
    for rank, chunk in enumerate(vec_hits):
        scores[chunk.id] += 1.0 / (k + rank)
    for rank, row in enumerate(fts_rows):
        scores[row.id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 6.3 Cross-encoder rerank

Use **FlashRank** with `ms-marco-MiniLM-L-12-v2` (~120MB ONNX). Runs in ~50-100ms on CPU. Local-only, no API. Singleton, lazy-loaded.

```python
def rerank(query, chunks, top_n=6):
    if not chunks:
        return []
    passages = [{"id": str(c.id), "text": c.content} for c in chunks]
    results = get_reranker().rerank(RerankRequest(query=query, passages=passages))
    by_id = {str(c.id): c for c in chunks}
    return [(by_id[r["id"]], r["score"]) for r in results[:top_n]]
```

Rationale for FlashRank over `bge-reranker-v2-m3`: FlashRank is ONNX-quantized and fast on CPU. `bge-reranker-v2-m3` (568M params) is more accurate but slow without a GPU. Swap is one-line if quality demands it later.

### 6.4 Fetch parents

Dedup parent IDs from the reranked children, preserve first-appearance order, return `DocumentParentChunk` rows. These — not children — are concatenated into the LLM prompt.

### 6.5 Tunable constants (in `config.py`)

| Setting              | Default |
|----------------------|---------|
| `vector_top_k`       | 30      |
| `fts_top_k`          | 30      |
| `rrf_k`              | 60      |
| `rerank_top_n`       | 6       |
| `rerank_score_floor` | 0.05    |

If all rerank scores are below `rerank_score_floor`, retrieval returns the `NO_RELEVANT_CHUNKS` sentinel and `grade_chunks` marks `graded_useful=False`.

## 7. Agent nodes (`nodes.py`)

### 7.1 `rewrite_query`

One LLM call. Output enforced via `.with_structured_output(QueryRewrite)`:

```python
class QueryRewrite(BaseModel):
    rewritten_query: str
    intent: Literal["lookup", "summary", "reasoning", "unclear"]
```

Prompt asks the LLM to: strip framing, resolve pronouns/ellipsis against the question, preserve proper nouns/codes/field names verbatim, and for named-field lookups return just the field name (e.g., `Corporate Name`). `intent=="unclear"` short-circuits to a clarification response — we do not guess.

This **replaces** `_preprocess_query` entirely. The regex is deleted, not kept as a fallback.

### 7.2 `retrieve_and_rerank`

No LLM call. Wires the four `retrieval.py` stages. Appends `rewritten_query` to `attempted_queries`. Records counts in `notes` for logs.

### 7.3 `grade_chunks`

Two paths, chosen by config:

- **Fast path (default):** `graded_useful = max(rerank_score) >= rerank_score_floor AND len(retrieved_children) >= 1`. No LLM cost.
- **Strict path:** one LLM call — "Is at least one of these passages sufficient to answer? YES or NO." Behind a feature flag for A/B.

### 7.4 `rewrite_and_retry`

Only entered when `not graded_useful AND retry_count < 2`. One LLM call, prompt:

> "Previous queries returned no useful results: {attempted_queries}. Propose ONE alternative query for the same intent. Use synonyms or different framing. Output just the query string."

Updates `rewritten_query`, increments `retry_count`, loops back to `retrieve_and_rerank`.

### 7.5 `generate_answer`

Streamed. Prompt branches on `graded_useful`:

- `True`:
  > "Answer the user's question using ONLY the document context below. Quote relevant text when it answers directly. If the answer isn't in the context, say so plainly. Do not invent details."
- `False` (retries exhausted):
  > "The document doesn't appear to contain information to answer this question. Briefly state what the document does cover, based on the context below, and tell the user the question wasn't answered."

Only node that streams tokens to the client.

### 7.6 `faithfulness_check`

Runs after streaming completes. One LLM call: "Is every factual claim in the draft answer supported by the context? YES or NO."

If `NO`: do not re-stream. Append a UI warning to the citations payload:

```json
{"type": "warning", "message": "Some claims may not be fully supported by the document."}
```

Re-streaming after the user has already seen tokens is poor UX. Buffered-stream-with-retry is documented as a known limitation for v2.

### 7.7 Graph wiring (`graph.py`)

```python
g = StateGraph(AgentState)
g.add_node("rewrite_query", rewrite_query)
g.add_node("retrieve", retrieve_and_rerank)
g.add_node("grade", grade_chunks)
g.add_node("retry", rewrite_and_retry)
g.add_node("answer", generate_answer)
g.add_node("check", faithfulness_check)

g.set_entry_point("rewrite_query")
g.add_edge("rewrite_query", "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", route_after_grade,
                        {"answer": "answer", "retry": "retry", "give_up": "answer"})
g.add_edge("retry", "retrieve")
g.add_edge("answer", "check")
g.add_edge("check", END)

graph = g.compile()
```

Streaming uses `graph.astream_events(v="v2")` and filters for `on_chat_model_stream` events from the `answer` node.

## 8. Evaluation harness

### 8.1 Layout

```
backend/evals/
  __init__.py
  golden_set.yaml          # Q/A pairs grouped by document
  run_eval.py              # CLI: python -m evals.run_eval [--name X] [--compare A B]
  metrics.py               # RAGAS wrapper + custom checks
  results/                 # JSON results per run
    baseline_2026-05-15.json
```

### 8.2 Golden set

~15-20 Q/A pairs built by hand from documents already in `uploads/`, covering categories:

- `named_field_lookup` — the failing case today
- `fact_lookup`
- `summarization`
- `not_in_doc` — negative test
- `pronoun_dependent` — tests rewriter

Format per entry:

```yaml
- document: "ACME_Corp_Registration.pdf"
  question: "What is the Corporate Name?"
  expected_answer: "ACME Corporation Ltd."
  expected_chunks:
    - "Corporate Name: ACME Corporation"
  category: "named_field_lookup"
```

### 8.3 Metrics

Four RAGAS metrics (judge = `ollama_chat_model` from `config.py`):

| Metric                  | What it measures                                      | Target  |
|-------------------------|-------------------------------------------------------|---------|
| `faithfulness`          | Answer claims supported by retrieved context          | ≥ 0.85  |
| `answer_relevancy`      | Answer addresses the question asked                   | ≥ 0.80  |
| `context_precision`     | Retrieved chunks relevant to the question             | ≥ 0.75  |
| `context_recall`        | Retrieval found the chunks that contain the answer    | ≥ 0.80  |

Two cheap custom checks (no LLM judge):

- `answered` — agent produced a non-empty answer (vs. "couldn't find")
- `expected_substring_match` — for short-fact questions, agent output contains `expected_answer` (case-insensitive). Directly tests the failure mode.

### 8.4 Runner

```bash
python -m evals.run_eval --name baseline
python -m evals.run_eval --name after_rerank
python -m evals.run_eval --compare baseline after_rerank
```

Per-run JSON output (`backend/evals/results/<name>_<date>.json`):

```json
{
  "run_name": "after_rerank",
  "timestamp": "2026-05-16T14:00:00Z",
  "git_sha": "abc123",
  "config_snapshot": {"rerank_top_n": 6},
  "per_question": [
    {"question": "...", "metrics": {"faithfulness": 0.91}, "answered": true, "expected_substring_match": true}
  ],
  "summary": {"faithfulness_mean": 0.91, "answered_rate": 0.94}
}
```

### 8.5 Test integration

- Pytest marker `@pytest.mark.eval` excludes slow eval tests from default `pytest` run.
- `backend/tests/test_rag.py` keeps fast unit tests with mocked LLM/DB.
- Add `backend/tests/test_rag_graph.py` for LangGraph wiring tests with mocked nodes.

## 9. Dependencies

All open-source, local:

```
ragas
datasets        # ragas dependency
langgraph
flashrank
```

No paid APIs introduced.

## 10. Rollout — four phases

Each phase is independently mergeable, with eval run between phases.

### Phase 0 — Baseline (½ day)
- Build `evals/golden_set.yaml` (~15-20 pairs).
- Add `run_eval.py`, `metrics.py`, RAGAS dep.
- Run against current `rag.py`. Commit `evals/results/baseline_<date>.json` as the regression target.

### Phase 1 — Parent-child chunking + reingest (1-2 days)
- Alembic migration `0004_parent_child_chunks.py`.
- `ingestion.py`: produce parents + children.
- New `DocumentParentChunk` model + relationship.
- `scripts/reingest_all.py`, run once locally.
- Re-run eval.

### Phase 2 — Retrieval upgrade: RRF + reranker (1 day)
- Add `flashrank`, `reranker.py` singleton.
- `retrieval.py`: `hybrid_search → rrf_fuse → rerank → fetch_parents`.
- Swap internals of existing `agentic_rag_stream` to call `retrieval.py` and return parents to the LLM.
- Re-run eval. **The failing case should resolve here.**

### Phase 3 — LangGraph CRAG-lite + LLM query rewriter (2-3 days)
- Add `langgraph`.
- Create `rag/` package, move code in.
- `rewrite_query` node (structured output) replaces `_preprocess_query`.
- Wire five nodes + conditional edges.
- Entry point swapped to `graph.astream_events(...)`.
- Delete: `_preprocess_query`, `make_search_tool`, the `bind_tools` loop. No fallbacks kept.
- Re-run eval.

### Phase 4 — Polish & observability (½ day)
- Per-node `notes` breadcrumbs in logs.
- Optional debug payload: `attempted_queries`, `retry_count` in the streamed `done` event.
- Update `features/chat_with_doc/rag_enhancement.md` and `CLAUDE.md`.

**Total: 5-7 working days.** Failure mode should be fixed by end of Phase 2.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| FlashRank model download in container (~120MB) | Bake into Docker image, or volume-mount cache dir |
| RAGAS judge slow with local Ollama (~5-15 min/run) | Cache responses by `(question, git_sha)` so unchanged questions skip re-judging |
| `langgraph.astream_events` stability with ChatOllama | Keep ability to fall back to manual `astream` on `generate_answer` node; smoke-test in Phase 3 |
| Reingest partial failure leaving inconsistent rows | Per-document transaction; on crash set `status='failed'`, user retries from UI |
| Re-stream-on-ungrounded not supported | Documented limitation. Surface as UI warning only. v2 could buffer tokens server-side. |

## 12. Out of scope (explicit)

- TypeScript migration of frontend (project is `.jsx`, per `CLAUDE.md`)
- Multi-document RAG
- Query decomposition / parallel sub-agents (Approach C in brainstorming)
- Chat history / multi-turn context (rewriter is single-turn for now; designed to accept history later)
- Switching to `bge-reranker-v2-m3` (deferred; rerank interface allows it)
- Switching to true BM25 (Postgres FTS retained; could swap to ParadeDB later)
- ColBERT/PLAID retrieval

## 13. References

- Corrective RAG (CRAG): https://www.langchain.com/blog/agentic-rag-with-langgraph
- Reciprocal Rank Fusion: https://www.paradedb.com/learn/search-concepts/reciprocal-rank-fusion
- FlashRank: https://github.com/PrithivirajDamodaran/FlashRank
- bge-reranker-v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3
- RAGAS: https://docs.ragas.io/
- GiovanniPasq/agentic-rag-for-dummies: https://github.com/GiovanniPasq/agentic-rag-for-dummies (parent-child pattern reference)
- Predecessor spec: `docs/superpowers/specs/2026-05-04-agentic-rag-design.md`
