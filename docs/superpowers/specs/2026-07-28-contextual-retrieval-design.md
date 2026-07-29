# Contextual Retrieval for the RAG Pipeline

**Date:** 2026-07-28
**Status:** Approved, not yet implemented
**Source guideline:** `features_planning/7.enhancing_rag_pipeline/claude_contextual_embedding.md`
**Current-state reference:** `wiki/02-flows.md` (flows 2 and 3), `wiki/03-data-model.md`

## Problem

The retrieval pipeline embeds each ~300-token child chunk in isolation. A chunk
reading "the limit rises to 40% in the second year" carries no indication of
which limit, which agreement, or which section it belongs to, so its vector
sits nowhere near a query that names those things.

The [Anthropic contextual-embeddings
guide](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/contextual-embeddings/guide.ipynb)
addresses this by prepending an LLM-generated description that situates each
chunk within its source document before embedding. Reported effect: Pass@10
rises from 87% to 92% on contextual embeddings alone, ~93% adding hybrid
search, ~95% adding reranking.

### What already exists

Two of the guide's three techniques are already implemented, which narrows the
scope of this work considerably:

| Technique | Current state |
|---|---|
| Hybrid search + RRF | **Implemented.** `retrieval.py:22-77` — pgvector cosine (top 30) + Postgres FTS `ts_rank`/`plainto_tsquery` (top 30), fused by `rrf_fuse(k=60)`. |
| Reranking | **Implemented.** `reranker.py:34` — OpenRouter `/v1/rerank`, over-retrieve ~30-60 → top 6. |
| Contextual embeddings | **Missing.** `ingestion.py:376` `embed_chunks()` embeds bare child text. |

So the work is contextual embeddings, plus three changes that make the existing
hybrid search and reranking actually *contextual*:

1. The keyword arm searches `content` only. Without change it would remain
   non-contextual while the vector arm became contextual.
2. RRF is unweighted (50/50). The guide recommends 80/20 semantic/keyword.
3. The reranker scores bare `content`, so it cannot use the context either.

### Two constraints that shape the design

**`content` must not be mutated.** Migration `0009` added `document_chunks.bbox`,
normalized rects mapping each chunk back to the source page lines it came from,
derived from the chunk's character offsets within the page text. Citations are
built from child `content` and highlighted via those rects. Prepending generated
prose to `content` would break the offset mapping and would surface
LLM-generated text to the user as a quotation from their document. Context
therefore lives in its own column.

**Postgres `ts_rank` is not BM25.** The guide specifies BM25. Real BM25 in
Postgres requires the ParadeDB `pg_search` extension. Note that RRF consumes
*rank order* only, not scores, so the practical gap between `ts_rank` and BM25 is
likely smaller than the gap contextualization closes — but BM25 is in scope by
explicit decision (see Decisions).

## Goals

- Every child chunk carries a generated context string, used for embedding,
  keyword search, and reranking.
- The keyword arm is true BM25 over context + content.
- RRF is weighted and tunable.
- No regression to citation geometry or answer faithfulness.
- Every new behaviour is flag-controlled so it can be turned off or A/B'd.

## Non-goals

- Feeding generated context to the answer-generation LLM (see Decisions).
- Backfilling existing documents (see Decisions).
- An automated eval gate (see Accepted risks).

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Contextualization granularity | **Per child**, full document as situating context | Highest fidelity; matches the cookbook. Per-parent would be ~6× cheaper but gives every child of a page an identical context. |
| Large documents | **Token-threshold fallback** to doc summary + parent page | Every document ingests successfully; large ones degrade rather than failing or costing $20+. |
| Context storage | **Separate `context` column** + stored generated column for search | Protects `bbox`/citations; keeps `content` byte-identical to source text. |
| BM25 backend | **ParadeDB `pg_search`**, `ts_rank` retained as fallback | User decision. Fallback keeps the app working without the extension and allows A/B against `ts_rank`. |
| BM25 index shape | **Stored generated column** `search_text = context ‖ content`, single `bm25` index | Trivial query (`search_text @@@ :q`); no per-field boost tuning. Costs ~2-3 MB of duplicated text on a large document. |
| Database image | **Migrate to `paradedb/paradedb`** | Bundles `pg_search` *and* `pgvector`; no custom Dockerfile. Viable only because existing data is being wiped. |
| Context in answer prompt | **No — retrieval-only** | Generated prose in the grounding context lets the model cite a hallucinated summary as document fact, and `faithfulness_check` cannot distinguish the two. |
| Backfill of existing docs | **None** | User will wipe and re-ingest manually. `context IS NULL` is a safe, supported state. |
| Cache TTL | **1 hour**, not the 5-minute default | 2× write vs 1.25×, break-even at 3 reads, and we get 120+ reads per document. Eliminates mid-run cache expiry on large documents. |

### Cost and latency

Measured against Haiku 4.5 pricing ($1/MTok input, $5/MTok output) with a 1.25×
cache write, 0.1× cache read, and ~100 output tokens per context:

| Document | Children | Added ingestion time (8-way) | Cost |
|---|---|---|---|
| 20 pages (~14k tok) | ~120 | ~30s | ~$0.28 |
| 100 pages (~70k tok) | ~600 | ~2.5 min | ~$4.60 |
| > threshold | — | — | falls back to the cheap tier |

Ingestion is already a background Celery task reporting via SSE, so added
latency is largely invisible to the user.

**Haiku 4.5's minimum cacheable prefix is 4096 tokens.** Documents below that
get no cache discount at all. They are cheap regardless, but the savings above
only materialise on mid-size and larger documents.

## Architecture

### New module: `backend/app/services/contextualizer.py`

Public surface is one function:

```python
def contextualize(
    parsed: ParsedDocument,
    children_per_parent: list[list[ChildChunk]],
) -> list[list[str | None]]
```

Returns a context string per child, positionally matching the input, with `None`
wherever generation failed. Tier selection, cache warming, concurrency, and
failure handling are internal. `ingestion.py` is already 444 lines; this keeps
it from growing.

Internals:

- `_situate(doc_context: str, chunk: str) -> str | None` — one LLM call. The
  document goes in a content block carrying
  `cache_control: {"type": "ephemeral", "ttl": "1h"}`; the chunk goes in a
  second, uncached block *after* it. Caching is a prefix match, so the stable
  document must precede the volatile chunk.
- `_summarize_document(text: str) -> str` — one call, fallback tier only.
- `_count_tokens(text: str) -> int` — `tiktoken` `cl100k_base`.

Prompts follow the cookbook (`DOCUMENT_CONTEXT_PROMPT` /
`CHUNK_CONTEXT_PROMPT`): "give a short succinct context to situate this chunk
within the overall document for the purposes of improving search retrieval",
answering with the context and nothing else.

#### Cache warming is load-bearing, not an optimisation

A cache entry becomes readable only once the first response *begins streaming*.
N concurrent calls sharing a prefix therefore all pay full input price. The
module issues child #1 alone, awaits it, then fans out the remaining N−1 through
a `ThreadPoolExecutor(max_workers=contextualizer_max_workers)`.

Getting this wrong raises ingestion cost roughly 10× **and produces no error** —
so it is asserted in tests and checked once manually on the first real ingest.

#### Failure degradation

Per-chunk and non-fatal, matching how `reranker.py` and `ocr_client.py` already
behave. An exception on one chunk yields `None` for that chunk, logged and
counted. A document whose contextualization fails wholesale still ingests
successfully with today's retrieval quality.

### Tier selection

Uses the existing `ParsedDocument.text` property.

| Document size | Situating context supplied per child |
|---|---|
| ≤ `contextualizer_full_doc_token_limit` (default 100k) | The full document |
| > limit | Generated doc-level summary + the child's own parent page |

The tier used and a `contextualized_children` count are written to
`documents.doc_metadata`, so a document's retrieval quality is explainable
after the fact.

`tiktoken` `cl100k_base` is OpenAI's tokenizer and undercounts Claude tokens by
roughly 15-20%; because embeddings and chat both route through OpenRouter, the
Anthropic `count_tokens` endpoint is not available. The 100k default against a
200k context window leaves ample margin for that error rather than attempting
precision.

## Data model

Migration `0010_contextual_retrieval`:

```sql
CREATE EXTENSION IF NOT EXISTS pg_search;

ALTER TABLE document_chunks ADD COLUMN context text;

ALTER TABLE document_chunks ADD COLUMN search_text text
  GENERATED ALWAYS AS (coalesce(context, '') || ' ' || content) STORED;

CREATE INDEX chunks_bm25 ON document_chunks
  USING bm25 (id, search_text) WITH (key_field = 'id');
```

`context` is nullable — that is the degradation path, and `coalesce` keeps the
generated column correct for every row without one. The GIN index from
migration `0003` indexes `to_tsvector('english', content)`, which cannot serve
a query over `search_text` (a different expression) — so it does **not**
cover the `ts_rank` fallback. Migration `0011` adds a second GIN index over
`to_tsvector('english', search_text)` for that path.

`shared_preload_libraries` is **not** required: `pg_search` only needs it on
Postgres < 17, and the ParadeDB image ships Postgres 18.

Model change in `models.py`: `DocumentChunk.context` (nullable `Text`).
`search_text` is database-generated and is not mapped as a writable column.
`ChildChunk` in `ingestion.py` gains a `context: str | None = None` field.

## Ingestion flow

`tasks.py` and `scripts/reingest_all.py` both move from
`parse → chunk → embed → store` to
`parse → chunk → contextualize → embed → store`.

The embedding input is `f"{context}\n\n{content}"` when context is present and
bare `content` when it is not. The caller builds these strings, so
`embed_chunks()` keeps its current `list[str]` signature. `store_chunks()`
persists `child.context`.

When `contextual_embeddings_enabled` is `False` the contextualize step is
skipped and every context is `None`, reproducing the pre-contextual
**embedding** behaviour only (the embedding input degrades to bare
`content`, byte-identical to before). It does **not** reproduce the rest of
today's pipeline: pool sizes (30→75), fusion weights (50/50→0.8/0.2), and the
BM25 keyword arm all change unconditionally and are not gated on this flag.

## Retrieval flow

`retrieval.py` keeps its shape — two independent recall arms fused by RRF, then
rerank, then parent fetch — with three changes.

**Keyword arm → BM25.** Selected by `bm25_enabled`. When that setting is left at
its default, a module-level `@lru_cache`'d helper in `retrieval.py` resolves it
on first use by querying `pg_extension` for `pg_search` (one query per process,
not per request); an explicitly-set `False` always wins so the fallback can be
forced for A/B runs.

```sql
SELECT id, paradedb.score(id) AS rank
FROM   document_chunks
WHERE  document_id = :doc_id
  AND  search_text @@@ :q
  -- optional: AND page BETWEEN :lo AND :hi
ORDER BY rank DESC
LIMIT :k
```

The existing `ts_rank`/`plainto_tsquery` query is retained as the fallback
implementation. This is **not** a safety net for a developer who hasn't
rebuilt the db image — migration `0010` opens with `CREATE EXTENSION
pg_search`, which fails outright on a non-ParadeDB image, and
`backend/Dockerfile`'s CMD (`alembic upgrade head && uvicorn`) means that
container never starts in the first place. The real reason the fallback
exists is so the keyword arm can be A/B'd against BM25 on a ParadeDB host via
`bm25_enabled=False`.

**Weighted RRF.** `rrf_fuse()` takes weights:
`w_vec / (k + rank)` for vector hits, `w_keyword / (k + rank)` for keyword
hits. Defaults 0.8 / 0.2.

**Contextual reranking.** `reranker.rerank()` sends
`f"{context}\n\n{content}"` per candidate rather than bare `content`, so the
cross-encoder sees the same signal the retrieval arms did.

Candidate pool rises from 30 + 30 to 75 + 75 (~150 after fusion), feeding the
same `rerank_top_n = 6`.

**Deliberately unchanged:** `fetch_parents()` still returns parent chunks as the
LLM's grounding context, and `graph.py` still builds citations from child
`content`. The `bbox` highlighting from migration `0009` therefore keeps
working, and no generated text can reach the user as a document quotation.

## Infrastructure

`docker-compose.yml`, `db` service:

- Image: `pgvector/pgvector:pg16` → a pinned `paradedb/paradedb` Postgres 18
  tag. ParadeDB publishes both `<app-version>-pg18` and `18-v<app-version>`
  naming; resolve the exact tag at implementation time with
  `docker manifest inspect` and hard-code it. Do **not** use `latest` or
  `latest-pg18` — `docker compose pull` could then move the Postgres major
  version underneath the volume.
- Volume: `pgdata:/var/lib/postgresql/data` → `pgdata:/var/lib/postgresql/`.
  Postgres 18 changed the data directory layout, and the ParadeDB docs mount the
  parent. Getting this wrong yields a database that silently re-initialises
  empty on every boot.

`POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` and the `pg_isready`
healthcheck are unchanged — the image keeps the official Postgres entrypoint.

Postgres 16 → 18 is a major-version jump, acceptable only because existing data
is being discarded. `pgvector`'s HNSW index and the `Vector(1536)` column need a
smoke test on 18 before being trusted.

## Config

Added to `backend/app/config.py`:

```python
contextual_embeddings_enabled: bool = True
contextualizer_model: str = "anthropic/claude-haiku-4.5"
contextualizer_max_workers: int = 8
contextualizer_full_doc_token_limit: int = 100_000  # cost ceiling, not just a
    # context-window guard: cost is quadratic in document size in the full-doc
    # tier (every child re-reads the whole doc), roughly $4-5 at the threshold.
contextualizer_cache_ttl: str = "1h"

bm25_enabled: bool = True          # auto-disabled when pg_search is absent
rrf_weight_vector: float = 0.8
rrf_weight_keyword: float = 0.2
```

Changed defaults:

```python
vector_top_k: int = 75             # was 30
fts_top_k: int = 75                # was 30
```

`fts_top_k` keeps its name even though the arm it sizes is now BM25 rather than
Postgres FTS, because the setting is also what sizes the retained `ts_rank`
fallback, and renaming it would break any existing `.env`.

## Testing

Unit tests, no network calls:

- `contextualize()` returns results positionally aligned with its input, given a
  stubbed LLM.
- A stub that raises produces all-`None` contexts and does not fail the document.
- Tier selection flips at `contextualizer_full_doc_token_limit`.
- The cache-warm call is issued alone and awaited before the fan-out begins.
- Embedding input is `context + content` when context exists, bare `content`
  when it is `None`.
- `rrf_fuse()` honours the weights (a keyword-only hit ranks below a
  vector-only hit at the same rank under 0.8/0.2).
- Migration round-trip: `search_text` populates from `context` + `content`, and
  stays correct when `context` is NULL.

Manual check on the first real ingest: confirm the OpenRouter response reports
non-zero cached-read input tokens. If OpenRouter does not pass `cache_control`
through to Anthropic as expected, contextualization still produces correct
output but costs roughly 10× the estimate, and it fails silently — so this needs
an explicit look rather than a test.

## Rollout

1. Swap the `db` image and volume path; wipe the volume; `alembic upgrade head`.
2. Verify `pg_search` and `pgvector` both load; smoke-test the HNSW index and a
   `Vector(1536)` similarity query on Postgres 18.
3. Ingest one representative document with `contextual_embeddings_enabled=False`,
   then again with it enabled. Compare retrieved chunks on a handful of queries
   by hand.
4. Re-ingest the remaining documents.

## Accepted risks

**No automated eval gate.** `backend/evals/golden_set.yaml` is still the
one-entry stub shipped with the harness (`REPLACE_WITH_REAL_FILENAME.pdf`), so
`python -m evals.run_eval --compare` has nothing to run against. The 0.8/0.2 RRF
weights, the 75/75 pool sizes, the 100k fallback threshold, and whether BM25
actually outperforms the `ts_rank` implementation it replaces are all reasoned
defaults rather than measured ones. Rollout step 3 is a manual spot-check, not a
regression test.

This is an explicit decision to defer evaluation, not an oversight. The
`contextual_embeddings_enabled` and `bm25_enabled` flags exist specifically so
each change can be A/B'd independently once a golden set is populated. A
retrieval-only Pass@k harness — labels of `(document, question, expected_page)`,
no ground-truth answer prose and no RAGAS judge cost — is the cheapest way to
close this later.

**Silent-failure surface.** Two of this design's failure modes produce correct
output at wrong cost rather than an error: a broken cache warm-up, and
OpenRouter not honouring `cache_control`. Both are covered above (a test and a
manual check respectively) because neither would otherwise be noticed.
