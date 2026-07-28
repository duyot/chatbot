# Contextual Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every child chunk an LLM-generated context string that situates it in its source document, and make the existing hybrid search and reranking use that context.

**Architecture:** A new `contextualizer.py` service generates one context per child chunk at ingestion time, passing the whole document in a prompt-cached content block. Context is stored in a new nullable `document_chunks.context` column plus a Postgres generated column `search_text = context ‖ content`, which a ParadeDB `pg_search` BM25 index covers. Retrieval keeps its existing shape — two recall arms fused by RRF, then rerank, then parent fetch — but the keyword arm becomes BM25, the fusion becomes weighted 0.8/0.2, and the reranker sees context + content.

**Tech Stack:** Python 3.11, FastAPI, SQLAlchemy 2.x, Alembic, Celery, Postgres 18 via `paradedb/paradedb` (bundles `pg_search` + `pgvector`), OpenRouter (`anthropic/claude-haiku-4.5` for chat, `qwen/qwen3-embedding-8b` for embeddings), pytest + pytest-mock.

**Spec:** `docs/superpowers/specs/2026-07-28-contextual-retrieval-design.md`

## Global Constraints

- **Never mutate `document_chunks.content`.** Migration `0009` derives `bbox` citation rects from character offsets within `content`. Context goes in its own column. Citations in `graph.py` continue to be built from child `content`.
- **Generated context must never reach the answer-generation LLM.** `generate_answer` keeps using parent-chunk text only. Retrieval-only, by design.
- **`context` is nullable and NULL is a supported state.** No backfill is planned; existing rows have no context. Every read path must handle `None`.
- **Contextualization failure is never fatal to a document.** A failed chunk gets `None`; a wholly failed document still ingests with today's quality.
- Document status string is `"done"`, never `"ready"`.
- Test DB schema comes from `Base.metadata.create_all()` in `backend/tests/conftest.py`, **not** Alembic. Any column tests need must be declared on the SQLAlchemy model, not only in the migration.
- Tests run against `chatbot_test` on `localhost:5432`; Docker's ParadeDB is on `5434`. Assume `pg_search` is **absent** in the test DB and skip BM25-SQL tests conditionally.
- Run all commands from `backend/`. Test command: `pytest`. The `-m "not eval"` default in `pytest.ini` already excludes the golden-set test.

---

## File Structure

| File | Responsibility |
|---|---|
| `backend/app/services/contextualizer.py` | **New.** Generate per-child context. Owns prompts, tier selection, cache warming, concurrency, failure degradation. |
| `backend/app/config.py` | Modify. New contextualizer + BM25 + RRF-weight settings; raise pool sizes. |
| `backend/app/models.py` | Modify. `DocumentChunk.context`, `DocumentChunk.search_text` (computed). |
| `backend/alembic/versions/0010_contextual_retrieval.py` | **New.** Extension, columns, BM25 index. |
| `backend/app/services/ingestion.py` | Modify. `ChildChunk.context` field; `store_chunks()` persists it; `build_embedding_input()`. |
| `backend/app/workers/tasks.py` | Modify. Insert the contextualize step; record metadata. |
| `backend/scripts/reingest_all.py` | Modify. Same pipeline change. |
| `backend/app/services/rag/retrieval.py` | Modify. BM25 arm + fallback detection, weighted RRF, pool sizes. |
| `backend/app/services/rag/reranker.py` | Modify. Send context + content per candidate. |
| `docker-compose.yml` | Modify. ParadeDB image + volume path. |
| `backend/tests/test_contextualizer.py` | **New.** Contextualizer unit tests. |
| `backend/tests/test_retrieval.py` | Modify. Weighted-RRF, BM25, and contextual-rerank tests. |
| `backend/tests/test_ingestion.py` | Modify. Context persistence + embed-input tests. |
| `wiki/02-flows.md`, `wiki/03-data-model.md`, `CLAUDE.md` | Modify. Document the new step, column, and migration. |

Contextualization lives in its own module rather than in `ingestion.py` (already 444 lines, approaching the 800 ceiling in the project's coding rules) because it has a single cohesive responsibility and a network-call failure model that nothing else in ingestion shares.

---

## Task 1: Foundation — settings, model columns, migration, database image

Establishes the schema and configuration everything else depends on. A reviewer can accept or reject this whole unit on one question: does the schema come up correctly on ParadeDB and in the test DB?

**Files:**
- Modify: `backend/app/config.py:31-43` (retrieval tunables block)
- Modify: `backend/app/models.py:3` (imports), `:45-69` (`DocumentChunk`)
- Create: `backend/alembic/versions/0010_contextual_retrieval.py`
- Modify: `docker-compose.yml:79-95` (`db` service)
- Test: `backend/tests/test_ingestion.py` (append)

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `settings.contextual_embeddings_enabled: bool`, `settings.contextualizer_model: str`, `settings.contextualizer_max_workers: int`, `settings.contextualizer_full_doc_token_limit: int`, `settings.contextualizer_cache_ttl: str`
  - `settings.bm25_enabled: bool`, `settings.rrf_weight_vector: float`, `settings.rrf_weight_keyword: float`
  - `settings.vector_top_k == 75`, `settings.fts_top_k == 75`
  - `DocumentChunk.context: str | None`, `DocumentChunk.search_text: str` (read-only, DB-computed)

- [ ] **Step 1: Add the settings**

In `backend/app/config.py`, immediately after the existing `reranker_model` line (~line 43), insert:

```python
    # --- Contextual embeddings (see docs/superpowers/specs/2026-07-28-contextual-retrieval-design.md)
    # Each child chunk gets an LLM-generated context string situating it within
    # its source document; that context is embedded and indexed alongside the
    # chunk text. Disable to reproduce the pre-contextual pipeline exactly.
    contextual_embeddings_enabled: bool = True
    contextualizer_model: str = "anthropic/claude-haiku-4.5"
    # Concurrent context-generation calls. The first call is always issued alone
    # to warm the prompt cache before fanning out.
    contextualizer_max_workers: int = 8
    # Documents above this token count fall back to
    # (doc summary + the child's own page) instead of the full document.
    # Measured with tiktoken cl100k_base, which undercounts Claude tokens by
    # ~15-20%, so this sits well under the 200k context window on purpose.
    contextualizer_full_doc_token_limit: int = 100_000
    # "1h" costs 2x on the cache write vs 1.25x for the 5-minute default, but
    # break-even is 3 reads and we get 100+ per document — and it stops a long
    # document from re-paying the write when a 5-minute entry expires mid-run.
    contextualizer_cache_ttl: str = "1h"

    # --- BM25 keyword search (ParadeDB pg_search)
    # Auto-detected at runtime: if pg_search is not installed, retrieval falls
    # back to the Postgres ts_rank query. Set False to force the fallback.
    bm25_enabled: bool = True
    # Weighted Reciprocal Rank Fusion. The guideline recommends 80/20
    # semantic/keyword; both are tunable.
    rrf_weight_vector: float = 0.8
    rrf_weight_keyword: float = 0.2
```

Then change the two existing pool sizes (~lines 32-33) from `30` to `75`:

```python
    vector_top_k: int = 75
    # Name retained even though this arm is now BM25: it also sizes the
    # ts_rank fallback, and renaming would break existing .env files.
    fts_top_k: int = 75
```

- [ ] **Step 2: Write the failing model/schema test**

Append to `backend/tests/test_ingestion.py`:

```python
def test_search_text_is_generated_from_context_and_content(db):
    """search_text is a Postgres STORED generated column; it must concatenate
    context and content, and stay correct when context is NULL."""
    import uuid
    from app.models import Document, DocumentChunk

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    with_ctx = DocumentChunk(
        document_id=doc.id, chunk_index=0, content="the rate is 40%",
        context="Section 3 of the lease agreement.", embedding=[0.0] * 1536,
    )
    without_ctx = DocumentChunk(
        document_id=doc.id, chunk_index=1, content="bare chunk",
        embedding=[0.0] * 1536,
    )
    db.add_all([with_ctx, without_ctx])
    db.flush()
    db.refresh(with_ctx)
    db.refresh(without_ctx)

    assert with_ctx.search_text == "Section 3 of the lease agreement. the rate is 40%"
    # NULL context must not null out the whole column
    assert without_ctx.search_text == " bare chunk"
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_ingestion.py::test_search_text_is_generated_from_context_and_content -v`
Expected: FAIL — `TypeError`/`AttributeError` on the unknown `context` kwarg.

- [ ] **Step 4: Add the model columns**

In `backend/app/models.py`, extend the top-level import (line 3) to include `Computed`:

```python
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, ForeignKey, UniqueConstraint, Index, Computed
```

Then in `DocumentChunk`, immediately after the `content` column (line 60), add:

```python
    # LLM-generated description situating this chunk within its source document.
    # Embedded and indexed alongside content; NULL means "not contextualized"
    # (pre-0010 rows, or contextualization disabled/failed) and is fully supported.
    context = Column(Text, nullable=True)
    # Postgres STORED generated column: what BM25 indexes. Declared here (not
    # only in the migration) because tests build their schema from
    # Base.metadata.create_all(), not Alembic. Read-only — never assign to it.
    search_text = Column(
        Text,
        Computed("coalesce(context, '') || ' ' || content", persisted=True),
        nullable=True,
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pytest tests/test_ingestion.py::test_search_text_is_generated_from_context_and_content -v`
Expected: PASS

If it fails with "column search_text does not exist", the session-scoped
`setup_tables` fixture is reusing an old schema. Drop and recreate:

```bash
psql -h localhost -U chatbot -c 'DROP DATABASE chatbot_test' -c 'CREATE DATABASE chatbot_test'
```

- [ ] **Step 6: Write the migration**

Create `backend/alembic/versions/0010_contextual_retrieval.py`:

```python
"""contextual retrieval: chunk context + BM25 search index

Adds document_chunks.context (nullable text, the LLM-generated situating
description) and document_chunks.search_text (STORED generated column
concatenating context and content), then a ParadeDB pg_search BM25 index over
search_text.

Requires the pg_search extension — the paradedb/paradedb image bundles it
alongside pgvector. shared_preload_libraries is NOT needed: pg_search only
requires it on Postgres < 17, and that image ships Postgres 18.

context is nullable so pre-existing rows survive; there is no backfill by
design. Retrieval degrades to content-only search for those rows via the
coalesce in search_text.

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-28
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.add_column("document_chunks", sa.Column("context", sa.Text(), nullable=True))
    op.add_column(
        "document_chunks",
        sa.Column(
            "search_text",
            sa.Text(),
            sa.Computed("coalesce(context, '') || ' ' || content", persisted=True),
            nullable=True,
        ),
    )
    op.execute(
        """
        CREATE INDEX chunks_bm25 ON document_chunks
        USING bm25 (id, search_text) WITH (key_field = 'id')
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS chunks_bm25")
    op.drop_column("document_chunks", "search_text")
    op.drop_column("document_chunks", "context")
    # pg_search is left installed: other objects may depend on it, and
    # dropping an extension is not this migration's business to undo.
```

- [ ] **Step 7: Swap the database image**

Resolve the concrete tag first — ParadeDB publishes two naming schemes and the current one must be verified, not guessed:

```bash
docker manifest inspect paradedb/paradedb:latest-pg18 >/dev/null && echo OK
```

Find the pinned equivalent (`<app-version>-pg18` or `18-v<app-version>`) on Docker Hub and use that literal tag. Then in `docker-compose.yml`, `db` service:

```yaml
  db:
    image: paradedb/paradedb:0.24.3-pg18   # <-- replace with the tag you verified
    environment:
      POSTGRES_DB: chatbot
      POSTGRES_USER: chatbot
      POSTGRES_PASSWORD: chatbot
    volumes:
      - pgdata:/var/lib/postgresql/
    ports:
      - "5434:5432"
```

Two things that will silently ruin the day if changed carelessly:
- The volume path loses `/data`. Postgres 18 changed the layout and ParadeDB mounts the parent. Mounting the old path yields a database that re-initialises empty on every boot, with no error.
- Do **not** use `latest` or `latest-pg18`. A later `docker compose pull` would move the Postgres major version underneath the volume.

Leave `environment`, `ports`, and the `pg_isready` healthcheck exactly as they are — the image keeps the official Postgres entrypoint.

- [ ] **Step 8: Bring up the new database and verify both extensions**

```bash
cd /d/development/chatbot
docker compose down
docker volume rm chatbot_pgdata
docker compose up -d db
sleep 10
docker compose exec db psql -U chatbot -d chatbot \
  -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pg_search;" -c "\dx"
```

Expected: `\dx` lists both `vector` and `pg_search`.

- [ ] **Step 9: Run the migration and smoke-test pgvector on Postgres 18**

```bash
docker compose run --rm backend alembic upgrade head
docker compose exec db psql -U chatbot -d chatbot -c "\d document_chunks" -c "\di chunks_bm25"
docker compose exec db psql -U chatbot -d chatbot -c \
  "SELECT id FROM document_chunks ORDER BY embedding <=> '[$(python -c "print(','.join(['0']*1536))")]' LIMIT 1;"
```

Expected: `context` and `search_text` present, `search_text` shown as `generated always as (...) stored`, `chunks_bm25` listed, and the vector query returning without error (zero rows is fine — it proves the HNSW index and operator work on PG18).

- [ ] **Step 10: Run the full suite and commit**

```bash
cd backend && pytest
git add app/config.py app/models.py alembic/versions/0010_contextual_retrieval.py tests/test_ingestion.py ../docker-compose.yml
git commit -m "feat: add chunk context column, BM25 index, and ParadeDB image

Migration 0010 adds document_chunks.context (nullable) and a STORED
generated search_text column, plus a pg_search BM25 index over it.
search_text is declared on the model too, since tests build schema from
create_all rather than Alembic.

Database image moves to paradedb/paradedb (bundles pg_search + pgvector);
volume path drops /data for the Postgres 18 layout."
```

---

## Task 2: The contextualizer module

**Files:**
- Create: `backend/app/services/contextualizer.py`
- Test: `backend/tests/test_contextualizer.py`

**Interfaces:**
- Consumes: `settings.contextualizer_model`, `settings.contextualizer_max_workers`, `settings.contextualizer_full_doc_token_limit`, `settings.contextualizer_cache_ttl` (Task 1); `ParsedDocument`, `PageContent`, `ChildChunk`, `_openai_client` from `app.services.ingestion`.
- Produces:
  - `contextualize(parsed: ParsedDocument, children_per_parent: list[list[ChildChunk]]) -> list[list[str | None]]` — same nesting shape as its input.
  - `contextualize_with_stats(parsed, children_per_parent) -> tuple[list[list[str | None]], dict]` where the dict is `{"tier": str, "contextualized_children": int, "total_children": int}`.
  - `count_tokens(text: str) -> int`
  - `TIER_FULL_DOC = "full_doc"`, `TIER_SUMMARY = "summary_plus_page"`
  - Internals that tests patch: `_situate(doc_context, chunk_content) -> str | None`, `_call_model(blocks, max_tokens) -> str`, `_summarize_document(text) -> str`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_contextualizer.py`:

```python
"""Unit tests for the contextualizer. No network calls: _situate,
_call_model, and _summarize_document are always patched."""
import pytest

from app.services.ingestion import ParsedDocument, PageContent, ChildChunk


def _parsed(*page_texts: str) -> ParsedDocument:
    return ParsedDocument(
        pages=[
            PageContent(page=i + 1, text=t, source="native")
            for i, t in enumerate(page_texts)
        ],
        metadata={"page_count": len(page_texts)},
    )


def _children(*specs: tuple[int, str]) -> list[list[ChildChunk]]:
    """specs is (page, content) per child; one parent per child for simplicity."""
    return [
        [ChildChunk(content=content, page=page, source="native")]
        for page, content in specs
    ]


def test_returns_same_nesting_shape_as_input(mocker):
    from app.services import contextualizer

    mocker.patch.object(contextualizer, "_situate", side_effect=lambda d, c: f"ctx:{c}")

    parsed = _parsed("page one text", "page two text")
    children = [
        [ChildChunk(content="a", page=1, source="native"),
         ChildChunk(content="b", page=1, source="native")],
        [ChildChunk(content="c", page=2, source="native")],
    ]

    result = contextualizer.contextualize(parsed, children)

    assert result == [["ctx:a", "ctx:b"], ["ctx:c"]]


def test_failed_chunk_yields_none_and_does_not_raise(mocker):
    from app.services import contextualizer

    def flaky(blocks, max_tokens):
        # The chunk text lives in the second content block.
        if "b" == _chunk_of(blocks):
            raise RuntimeError("openrouter 500")
        return f"ctx:{_chunk_of(blocks)}"

    mocker.patch.object(contextualizer, "_call_model", side_effect=flaky)

    parsed = _parsed("doc text")
    children = _children((1, "a"), (1, "b"), (1, "c"))

    result = contextualizer.contextualize(parsed, children)

    assert result == [["ctx:a"], [None], ["ctx:c"]]


def _chunk_of(blocks) -> str:
    """Pull the chunk content back out of the prompt blocks _situate built."""
    import re
    m = re.search(r"<chunk>\n(.*?)\n</chunk>", blocks[1]["text"], re.DOTALL)
    return m.group(1) if m else ""


def test_all_calls_failing_still_returns_full_shape(mocker):
    from app.services import contextualizer

    mocker.patch.object(contextualizer, "_call_model", side_effect=RuntimeError("down"))

    parsed = _parsed("doc text")
    children = _children((1, "a"), (1, "b"))

    result = contextualizer.contextualize(parsed, children)

    assert result == [[None], [None]]


def test_cache_is_warmed_before_fanout(mocker):
    """The first call must complete alone before the pool is created: a cache
    entry is only readable once the first response starts streaming, so a
    concurrent fan-out would make every call pay full input price."""
    from app.services import contextualizer

    events = []

    def spy_situate(doc_ctx, chunk):
        events.append(("situate", chunk))
        return f"ctx:{chunk}"

    real_pool = contextualizer.ThreadPoolExecutor

    def spy_pool(*args, **kwargs):
        events.append(("pool",))
        return real_pool(*args, **kwargs)

    mocker.patch.object(contextualizer, "_situate", side_effect=spy_situate)
    mocker.patch.object(contextualizer, "ThreadPoolExecutor", spy_pool)

    parsed = _parsed("doc text")
    children = _children((1, "a"), (1, "b"), (1, "c"))

    contextualizer.contextualize(parsed, children)

    assert events[0] == ("situate", "a"), "first chunk must be situated alone"
    assert events[1] == ("pool",), "pool must not be created until the warm call returns"


def test_single_child_document_creates_no_pool(mocker):
    from app.services import contextualizer

    events = []
    mocker.patch.object(contextualizer, "_situate", side_effect=lambda d, c: "ctx")
    mocker.patch.object(
        contextualizer, "ThreadPoolExecutor",
        side_effect=lambda *a, **k: events.append("pool"),
    )

    result = contextualizer.contextualize(_parsed("t"), _children((1, "only")))

    assert result == [["ctx"]]
    assert events == []


def test_full_doc_tier_passes_whole_document(mocker):
    from app.services import contextualizer

    seen = []
    mocker.patch.object(
        contextualizer, "_situate",
        side_effect=lambda d, c: seen.append(d) or "ctx",
    )

    parsed = _parsed("alpha page", "beta page")
    contextualizer.contextualize(parsed, _children((1, "a")))

    assert "alpha page" in seen[0]
    assert "beta page" in seen[0]


def test_oversized_doc_falls_back_to_summary_plus_page(mocker):
    from app.services import contextualizer

    mocker.patch.object(contextualizer.settings, "contextualizer_full_doc_token_limit", 5)
    mocker.patch.object(contextualizer, "_summarize_document", return_value="SUMMARY")

    seen = []
    mocker.patch.object(
        contextualizer, "_situate",
        side_effect=lambda d, c: seen.append(d) or "ctx",
    )

    parsed = _parsed("alpha " * 50, "beta page two")
    contextualizer.contextualize(parsed, _children((2, "chunk on page two")))

    assert "SUMMARY" in seen[0]
    assert "beta page two" in seen[0]
    assert "alpha" not in seen[0], "fallback must not include unrelated pages"


def test_stats_report_tier_and_success_count(mocker):
    from app.services import contextualizer

    def flaky(blocks, max_tokens):
        if _chunk_of(blocks) == "b":
            raise RuntimeError("boom")
        return "ctx"

    mocker.patch.object(contextualizer, "_call_model", side_effect=flaky)

    contexts, stats = contextualizer.contextualize_with_stats(
        _parsed("doc"), _children((1, "a"), (1, "b"), (1, "c"))
    )

    assert stats["tier"] == contextualizer.TIER_FULL_DOC
    assert stats["contextualized_children"] == 2
    assert stats["total_children"] == 3


def test_empty_children_returns_empty(mocker):
    from app.services import contextualizer
    mocker.patch.object(contextualizer, "_situate", side_effect=AssertionError("no calls"))
    assert contextualizer.contextualize(_parsed("t"), []) == []


def test_situate_sends_cache_control_on_document_block_only(mocker):
    """The document block carries cache_control and must come first; the
    volatile chunk block must follow it uncached, or the prefix never matches."""
    from app.services import contextualizer

    fake_client = mocker.MagicMock()
    fake_client.chat.completions.create.return_value = mocker.MagicMock(
        choices=[mocker.MagicMock(message=mocker.MagicMock(content="  the context  "))]
    )
    mocker.patch.object(contextualizer, "_openai_client", return_value=fake_client)

    out = contextualizer._situate("WHOLE DOC", "THE CHUNK")

    assert out == "the context"
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    blocks = kwargs["messages"][0]["content"]
    assert "WHOLE DOC" in blocks[0]["text"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "THE CHUNK" in blocks[1]["text"]
    assert "cache_control" not in blocks[1]


def test_count_tokens_is_monotonic():
    from app.services.contextualizer import count_tokens
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
    assert count_tokens("hello world " * 100) > count_tokens("hello world")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_contextualizer.py -v`
Expected: all FAIL with `ModuleNotFoundError: No module named 'app.services.contextualizer'`

- [ ] **Step 3: Write the module**

Create `backend/app/services/contextualizer.py`:

```python
"""Generate a short context string situating each child chunk within its source
document, so the chunk embeds and indexes with the information it needs.

A chunk reading "the limit rises to 40% in the second year" is nearly useless in
isolation; prefixed with "Section 3 of the 2024 lease, on rent escalation" it is
findable. See the Anthropic contextual-embeddings cookbook and
docs/superpowers/specs/2026-07-28-contextual-retrieval-design.md.

Cost control rests on prompt caching: the document is sent once as a cached
block and re-read per chunk at ~10% of input price. Two properties of that are
load-bearing and easy to break silently:

1. The cached document block must come FIRST and the volatile chunk block
   SECOND. Caching is a prefix match.
2. The first call must complete BEFORE the rest fan out. A cache entry is only
   readable once the first response begins streaming, so a fully concurrent
   fan-out makes every call pay full input price — roughly 10x, with no error.

Failure is per-chunk and never fatal: a chunk that cannot be contextualized
gets None and is embedded on its content alone, matching how reranker.py and
ocr_client.py already degrade.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import tiktoken

from ..config import settings
from .ingestion import ChildChunk, ParsedDocument, _openai_client

logger = logging.getLogger(__name__)

TIER_FULL_DOC = "full_doc"
TIER_SUMMARY = "summary_plus_page"

DOCUMENT_CONTEXT_PROMPT = """<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

SUMMARY_PROMPT = """<document>
{doc_content}
</document>

Summarize this document in at most 200 words. Focus on what it is, who it
concerns, and how it is organized, so the summary can help situate excerpts
taken from it. Answer only with the summary and nothing else."""

# Truncation guard for the summary tier: a document over the full-doc limit can
# still be enormous, and the summary call itself must not blow the window.
_SUMMARY_INPUT_CHAR_CAP = 400_000


def count_tokens(text: str) -> int:
    """Approximate token count via cl100k_base.

    This is OpenAI's tokenizer, not Claude's, and undercounts Claude tokens by
    roughly 15-20%. We use it because chat routes through OpenRouter, where the
    Anthropic count_tokens endpoint is unavailable. The full-doc threshold is
    set well below the context window to absorb the error.
    """
    if not text:
        return 0
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _call_model(blocks: List[dict], max_tokens: int) -> str:
    """Single chat completion through OpenRouter. Raises on failure — callers
    decide whether that is fatal."""
    client = _openai_client()
    response = client.chat.completions.create(
        model=settings.contextualizer_model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": blocks}],
    )
    return (response.choices[0].message.content or "").strip()


def _situate(doc_context: str, chunk_content: str) -> Optional[str]:
    """Generate one chunk's context. Returns None on any failure.

    Block order matters: the stable document first (cached), the volatile chunk
    second (uncached). Reversing them defeats the prefix match entirely.
    """
    blocks = [
        {
            "type": "text",
            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc_context),
            "cache_control": {
                "type": "ephemeral",
                "ttl": settings.contextualizer_cache_ttl,
            },
        },
        {
            "type": "text",
            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content),
        },
    ]
    try:
        text = _call_model(blocks, max_tokens=256)
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the document
        logger.warning(
            "_situate: context generation failed, chunk will embed on content "
            "alone (%s)", exc,
        )
        return None
    return text or None


def _summarize_document(text: str) -> str:
    """Doc-level summary for the fallback tier. Returns "" on failure, which
    still leaves the child's own page as situating context."""
    blocks = [{
        "type": "text",
        "text": SUMMARY_PROMPT.format(doc_content=text[:_SUMMARY_INPUT_CHAR_CAP]),
    }]
    try:
        return _call_model(blocks, max_tokens=512)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_summarize_document: failed, falling back to page-only context (%s)", exc
        )
        return ""


def contextualize_with_stats(
    parsed: ParsedDocument,
    children_per_parent: List[List[ChildChunk]],
) -> Tuple[List[List[Optional[str]]], dict]:
    """Generate a context per child. Returns (contexts, stats).

    `contexts` mirrors the nesting of `children_per_parent` exactly, with None
    wherever generation failed. `stats` carries the tier used and how many
    children were successfully contextualized, for documents.doc_metadata.
    """
    total = sum(len(cs) for cs in children_per_parent)
    if total == 0:
        return [], {
            "tier": TIER_FULL_DOC,
            "contextualized_children": 0,
            "total_children": 0,
        }

    doc_text = parsed.text
    doc_tokens = count_tokens(doc_text)
    use_full_doc = doc_tokens <= settings.contextualizer_full_doc_token_limit
    tier = TIER_FULL_DOC if use_full_doc else TIER_SUMMARY

    if use_full_doc:
        page_text: dict = {}
        summary = ""
    else:
        page_text = {p.page: p.text for p in parsed.pages}
        summary = _summarize_document(doc_text)

    def doc_context_for(child: ChildChunk) -> str:
        if use_full_doc:
            return doc_text
        parts = []
        if summary:
            parts.append(f"Document summary:\n{summary}")
        page = page_text.get(child.page)
        if page:
            parts.append(f"Page {child.page} of the document:\n{page}")
        return "\n\n".join(parts)

    # Flatten to (parent_idx, child_idx, child) so results can be scattered back.
    flat = [
        (pi, ci, child)
        for pi, children in enumerate(children_per_parent)
        for ci, child in enumerate(children)
    ]

    logger.info(
        "contextualize: doc_tokens=%d tier=%s children=%d workers=%d",
        doc_tokens, tier, len(flat), settings.contextualizer_max_workers,
    )

    results: List[Optional[str]] = [None] * len(flat)

    # Warm the prompt cache with a single call before fanning out. Do NOT
    # collapse this into the pool: concurrent calls cannot read a cache entry
    # that is still being written, so all of them would pay full price.
    _, _, first_child = flat[0]
    results[0] = _situate(doc_context_for(first_child), first_child.content)

    if len(flat) > 1:
        with ThreadPoolExecutor(max_workers=settings.contextualizer_max_workers) as pool:
            futures = {
                pool.submit(_situate, doc_context_for(child), child.content): idx
                for idx, (_, _, child) in enumerate(flat)
                if idx > 0
            }
            for future, idx in futures.items():
                results[idx] = future.result()

    contexts: List[List[Optional[str]]] = [
        [None] * len(children) for children in children_per_parent
    ]
    for (pi, ci, _), ctx in zip(flat, results):
        contexts[pi][ci] = ctx

    succeeded = sum(1 for c in results if c)
    if succeeded < len(flat):
        logger.warning(
            "contextualize: %d/%d children failed contextualization",
            len(flat) - succeeded, len(flat),
        )
    logger.info("contextualize: done tier=%s ok=%d/%d", tier, succeeded, len(flat))

    return contexts, {
        "tier": tier,
        "contextualized_children": succeeded,
        "total_children": len(flat),
    }


def contextualize(
    parsed: ParsedDocument,
    children_per_parent: List[List[ChildChunk]],
) -> List[List[Optional[str]]]:
    """Convenience wrapper when the caller does not need the stats dict."""
    contexts, _ = contextualize_with_stats(parsed, children_per_parent)
    return contexts
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_contextualizer.py -v`
Expected: all 12 PASS

If `test_single_child_document_creates_no_pool` fails, the `len(flat) > 1` guard is missing — a one-child document must not construct a pool at all.

- [ ] **Step 5: Commit**

```bash
git add app/services/contextualizer.py tests/test_contextualizer.py
git commit -m "feat: add contextualizer service for per-chunk document context

Generates one situating context per child chunk via OpenRouter, with the
document sent as a 1h-cached prompt block. Warms the cache with a single
call before fanning out, since concurrent calls cannot read an entry that
is still being written. Per-chunk failures degrade to None."
```

---

## Task 3: Wire contextualization into ingestion

**Files:**
- Modify: `backend/app/services/ingestion.py:63-70` (`ChildChunk`), `:376` (add helper), `:398-443` (`store_chunks`)
- Modify: `backend/app/workers/tasks.py:1-10` (imports), `:28-48`
- Modify: `backend/scripts/reingest_all.py:18-25` (imports), `:45-58`
- Test: `backend/tests/test_ingestion.py` (append)

**Interfaces:**
- Consumes: `contextualize_with_stats` (Task 2); `DocumentChunk.context`, `settings.contextual_embeddings_enabled` (Task 1).
- Produces:
  - `ChildChunk.context: str | None` (defaults `None`)
  - `build_embedding_input(context: str | None, content: str) -> str` in `ingestion.py`
  - `store_chunks()` persists `child.context`
  - `documents.doc_metadata["contextualization"] = {"tier": ..., "contextualized_children": ..., "total_children": ...}`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_ingestion.py`:

```python
def test_build_embedding_input_prefixes_context_when_present():
    from app.services.ingestion import build_embedding_input
    assert build_embedding_input("Section 3.", "the rate is 40%") == (
        "Section 3.\n\nthe rate is 40%"
    )


def test_build_embedding_input_returns_bare_content_when_no_context():
    """Disabling contextual embeddings must be a true no-op, so the no-context
    path has to stay byte-identical to the pre-feature behaviour."""
    from app.services.ingestion import build_embedding_input
    assert build_embedding_input(None, "bare") == "bare"
    assert build_embedding_input("", "bare") == "bare"


def test_store_chunks_persists_context(db):
    import uuid
    from app.models import Document, DocumentChunk
    from app.services.ingestion import ParentChunk, ChildChunk, store_chunks

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.flush()

    parents = [ParentChunk(content="P", page_start=1, page_end=1, source="native")]
    children = [[
        ChildChunk(content="c1", page=1, source="native", context="CTX ONE"),
        ChildChunk(content="c2", page=1, source="native", context=None),
    ]]
    store_chunks(db, str(doc.id), parents, children, [[0.0] * 1536, [0.0] * 1536])

    rows = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == doc.id)
        .order_by(DocumentChunk.chunk_index)
        .all()
    )
    assert [r.context for r in rows] == ["CTX ONE", None]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_ingestion.py -k "embedding_input or persists_context" -v`
Expected: FAIL — `ImportError: cannot import name 'build_embedding_input'`, and `TypeError` on the unexpected `context` kwarg.

- [ ] **Step 3: Extend `ChildChunk` and add the embedding-input helper**

In `backend/app/services/ingestion.py`, replace the `ChildChunk` dataclass with:

```python
@dataclass
class ChildChunk:
    content: str
    page: int
    source: str
    ocr_confidence: Optional[float] = None
    bbox: Optional[List[List[float]]] = None  # normalized rects covering this chunk
    # LLM-generated context situating this chunk in its document. None when
    # contextualization is disabled or failed; the chunk then embeds on content
    # alone, exactly as before this feature existed.
    context: Optional[str] = None
```

Then add this function just above `embed_chunks` (near line 376):

```python
def build_embedding_input(context: Optional[str], content: str) -> str:
    """Text actually sent to the embedding model for a child chunk.

    Contextual embeddings prepend the generated context so the vector carries
    document-level meaning. A chunk with no context embeds on content alone —
    this must stay byte-identical to the pre-contextual behaviour so disabling
    the feature is a true no-op.
    """
    if not context:
        return content
    return f"{context}\n\n{content}"
```

- [ ] **Step 4: Persist context in `store_chunks`**

In `store_chunks`, add `context=child.context` to the `DocumentChunk(...)` construction, immediately after the `bbox` line (~line 435):

```python
                bbox=child.bbox or None,
                context=child.context,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_ingestion.py -k "embedding_input or persists_context" -v`
Expected: PASS

- [ ] **Step 6: Insert the contextualize step in the Celery task**

In `backend/app/workers/tasks.py`, replace the import block (lines 5-8) with:

```python
from .celery_app import celery_app
from ..config import settings
from ..database import SessionLocal
from ..models import Document
from ..services.contextualizer import contextualize_with_stats
from ..services.ingestion import (
    parse_document,
    chunk_document,
    embed_chunks,
    store_chunks,
    build_embedding_input,
)
```

Then replace the block from the `chunk_document` call through the `embed_chunks` call (lines 28-38) with:

```python
        parents, children_per_parent = chunk_document(parsed)
        if not parents:
            raise ValueError("No extractable text found in document (empty or OCR failed)")

        ctx_stats = None
        if settings.contextual_embeddings_enabled:
            contexts, ctx_stats = contextualize_with_stats(parsed, children_per_parent)
            for children, child_contexts in zip(children_per_parent, contexts):
                for child, ctx in zip(children, child_contexts):
                    child.context = ctx
            logger.info(
                "[task:%s] contextualized %d/%d children tier=%s",
                self.request.id, ctx_stats["contextualized_children"],
                ctx_stats["total_children"], ctx_stats["tier"],
            )
        else:
            logger.info("[task:%s] contextual embeddings disabled, skipping", self.request.id)

        flat_children = [
            build_embedding_input(c.context, c.content)
            for sub in children_per_parent for c in sub
        ]
        logger.info(
            "[task:%s] chunked text parents=%d children=%d",
            self.request.id, len(parents), len(flat_children),
        )

        embeddings = embed_chunks(flat_children)
```

Finally replace the `doc.doc_metadata` assignment (line 47) with:

```python
        metadata = dict(parsed.metadata)
        if ctx_stats is not None:
            metadata["contextualization"] = ctx_stats
        doc.doc_metadata = metadata
```

- [ ] **Step 7: Make the same change in the reingest script**

In `backend/scripts/reingest_all.py`, replace the import block (lines 18-25) with:

```python
from app.config import settings
from app.database import SessionLocal
from app.models import Document, DocumentChunk, DocumentParentChunk
from app.services.contextualizer import contextualize_with_stats
from app.services.ingestion import (
    parse_document,
    chunk_document,
    embed_chunks,
    store_chunks,
    build_embedding_input,
)
```

Then replace lines 46-51 with:

```python
        parents, children_per_parent = chunk_document(parsed)
        if not parents:
            raise ValueError("No extractable text found in document (empty or OCR failed)")

        ctx_stats = None
        if settings.contextual_embeddings_enabled:
            contexts, ctx_stats = contextualize_with_stats(parsed, children_per_parent)
            for children, child_contexts in zip(children_per_parent, contexts):
                for child, ctx in zip(children, child_contexts):
                    child.context = ctx
            logger.info(
                "contextualized %d/%d children tier=%s",
                ctx_stats["contextualized_children"], ctx_stats["total_children"],
                ctx_stats["tier"],
            )

        flat_children = [
            build_embedding_input(c.context, c.content)
            for sub in children_per_parent for c in sub
        ]
        embeddings = embed_chunks(flat_children)
        store_chunks(db, str(doc_id), parents, children_per_parent, embeddings)
```

And set metadata the same way as the task, replacing the `doc.doc_metadata` line (~line 57):

```python
        metadata = dict(parsed.metadata)
        if ctx_stats is not None:
            metadata["contextualization"] = ctx_stats
        doc.doc_metadata = metadata
```

- [ ] **Step 8: Run the full suite**

Run: `pytest`
Expected: PASS.

`tests/test_tasks.py` exercises `ingest_document`. If it patches `embed_chunks` or `chunk_document`, it will now also need `app.workers.tasks.contextualize_with_stats` patched — otherwise it attempts a real network call. Patch it there:

```python
    mocker.patch(
        "app.workers.tasks.contextualize_with_stats",
        return_value=([[None]], {"tier": "full_doc", "contextualized_children": 0, "total_children": 1}),
    )
```

Match the returned nesting to whatever `chunk_document` is stubbed to produce in that test. Do not weaken the task code to make a test pass.

- [ ] **Step 9: Commit**

```bash
git add app/services/ingestion.py app/workers/tasks.py scripts/reingest_all.py tests/test_ingestion.py tests/test_tasks.py
git commit -m "feat: contextualize chunks during ingestion

Inserts the contextualize step between chunking and embedding in both the
Celery task and the reingest script. Embedding input becomes
context + content when context exists, and stays byte-identical to the old
behaviour when it does not. Tier and success count land in doc_metadata."
```

---

## Task 4: BM25 keyword arm with ts_rank fallback

**Files:**
- Modify: `backend/app/services/rag/retrieval.py:19` (add detection), `:22-67` (`hybrid_search`)
- Test: `backend/tests/test_retrieval.py` (append)

**Interfaces:**
- Consumes: `settings.bm25_enabled`, `settings.fts_top_k` (Task 1); `DocumentChunk.search_text` (Task 1).
- Produces:
  - `bm25_available(db: Session) -> bool`
  - `reset_bm25_cache() -> None` (test hook)
  - `_keyword_search_bm25(db, document_id, query, page_range, k) -> list`
  - `_keyword_search_tsrank(db, document_id, query, page_range, k) -> list`
  - `hybrid_search()` signature unchanged: `(db, document_id, query, page_range=None) -> tuple[list[DocumentChunk], list]`

Note a deliberate deviation from the spec: detection is a module-level cached flag with an explicit reset, not `@lru_cache`, because the check needs a live `Session` and `lru_cache` would key on the unhashable session object.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_retrieval.py`:

```python
def test_bm25_available_false_when_setting_disabled(db, mocker):
    from app.services.rag import retrieval
    mocker.patch.object(retrieval.settings, "bm25_enabled", False)
    retrieval.reset_bm25_cache()
    assert retrieval.bm25_available(db) is False


def test_bm25_available_caches_after_first_probe(db, mocker):
    from app.services.rag import retrieval
    mocker.patch.object(retrieval.settings, "bm25_enabled", True)
    retrieval.reset_bm25_cache()

    spy = mocker.spy(db, "execute")
    first = retrieval.bm25_available(db)
    calls_after_first = spy.call_count
    second = retrieval.bm25_available(db)

    assert first == second
    assert spy.call_count == calls_after_first, "probe must run once per process"


def test_hybrid_search_uses_tsrank_when_bm25_unavailable(db, mocker):
    """The fallback must return real results, not an empty list — a dev without
    pg_search should still get working keyword recall."""
    from app.services.rag import retrieval
    from app.services.rag.retrieval import hybrid_search
    from app.models import Document, DocumentChunk

    mocker.patch.object(retrieval, "embed_text", return_value=[0.0] * 1536)
    mocker.patch.object(retrieval, "bm25_available", return_value=False)

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.add_all([
        DocumentChunk(document_id=doc.id, chunk_index=0, content="quarterly revenue grew",
                      embedding=[0.0] * 1536, page=1, source="native"),
        DocumentChunk(document_id=doc.id, chunk_index=1, content="unrelated boilerplate",
                      embedding=[0.0] * 1536, page=1, source="native"),
    ])
    db.flush()

    _, keyword_rows = hybrid_search(db, str(doc.id), "quarterly revenue")
    assert len(keyword_rows) >= 1


def test_tsrank_fallback_matches_on_context_too(db, mocker):
    """Even the fallback searches context + content, so contextual keyword
    recall does not depend on pg_search being installed."""
    from app.services.rag import retrieval
    from app.services.rag.retrieval import hybrid_search
    from app.models import Document, DocumentChunk

    mocker.patch.object(retrieval, "embed_text", return_value=[0.0] * 1536)
    mocker.patch.object(retrieval, "bm25_available", return_value=False)

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.add(DocumentChunk(
        document_id=doc.id, chunk_index=0, content="the rate is 40 percent",
        context="Section 3 on escalation clauses",
        embedding=[0.0] * 1536, page=1, source="native",
    ))
    db.flush()

    # "escalation" appears only in the context, never in content.
    _, keyword_rows = hybrid_search(db, str(doc.id), "escalation")
    assert len(keyword_rows) == 1


@pytest.mark.skipif(True, reason="requires pg_search; run against the ParadeDB container")
def test_bm25_search_matches_on_context(db, mocker):
    """Integration check for the real BM25 path. Flip the skipif to False and
    run with DATABASE_URL pointed at the ParadeDB container's chatbot_test DB."""
    from app.services.rag import retrieval
    from app.services.rag.retrieval import hybrid_search
    from app.models import Document, DocumentChunk

    mocker.patch.object(retrieval, "embed_text", return_value=[0.0] * 1536)
    retrieval.reset_bm25_cache()

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.add(DocumentChunk(
        document_id=doc.id, chunk_index=0, content="the rate is 40 percent",
        context="Section 3 on escalation clauses",
        embedding=[0.0] * 1536, page=1, source="native",
    ))
    db.flush()

    _, keyword_rows = hybrid_search(db, str(doc.id), "escalation")
    assert len(keyword_rows) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_retrieval.py -k "bm25 or tsrank" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'reset_bm25_cache'`. `test_bm25_search_matches_on_context` reports SKIPPED, which is correct.

- [ ] **Step 3: Implement detection**

In `backend/app/services/rag/retrieval.py`, add after the logger (line 19):

```python
# Cached pg_search probe. Module-level rather than @lru_cache because the check
# needs a live Session, which is unhashable. One query per process.
_BM25_AVAILABLE: bool | None = None


def reset_bm25_cache() -> None:
    """Clear the cached pg_search probe. Test hook only."""
    global _BM25_AVAILABLE
    _BM25_AVAILABLE = None


def bm25_available(db: Session) -> bool:
    """True when BM25 keyword search should be used.

    An explicit bm25_enabled=False always wins, so the ts_rank fallback can be
    forced for A/B comparison. Otherwise probe once for the pg_search
    extension: a developer who has not rebuilt the db image still gets a
    working app instead of a 500.
    """
    global _BM25_AVAILABLE
    if not settings.bm25_enabled:
        return False
    if _BM25_AVAILABLE is None:
        try:
            row = db.execute(
                sa_text("SELECT 1 FROM pg_extension WHERE extname = 'pg_search'")
            ).first()
            _BM25_AVAILABLE = bool(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25_available: probe failed, using ts_rank (%s)", exc)
            _BM25_AVAILABLE = False
        logger.info("bm25_available: pg_search detected=%s", _BM25_AVAILABLE)
    return _BM25_AVAILABLE
```

- [ ] **Step 4: Implement the two keyword arms**

Add these just above `hybrid_search`:

```python
def _keyword_search_bm25(
    db: Session, document_id: str, query: str,
    page_range: tuple[int, int] | None, k: int,
) -> list[Any]:
    """True BM25 over search_text (context || content) via ParadeDB pg_search."""
    page_clause = "AND page BETWEEN :lo AND :hi" if page_range is not None else ""
    sql = sa_text(
        f"""
        SELECT id, paradedb.score(id) AS rank
        FROM   document_chunks
        WHERE  document_id = :doc_id
          AND  search_text @@@ :q
          {page_clause}
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    params: dict[str, Any] = {"doc_id": document_id, "q": query, "k": k}
    if page_range is not None:
        params["lo"], params["hi"] = page_range[0], page_range[1]
    return db.execute(sql, params).fetchall()


def _keyword_search_tsrank(
    db: Session, document_id: str, query: str,
    page_range: tuple[int, int] | None, k: int,
) -> list[Any]:
    """Postgres FTS fallback when pg_search is unavailable.

    Searches search_text, not content, so contextual keyword recall works here
    too. ts_rank is not BM25 — but RRF consumes rank order, not scores, so the
    practical difference is smaller than it sounds.
    """
    page_clause = "AND page BETWEEN :lo AND :hi" if page_range is not None else ""
    sql = sa_text(
        f"""
        SELECT id,
               ts_rank(to_tsvector('english', search_text),
                       plainto_tsquery('english', :q)) AS rank
        FROM   document_chunks
        WHERE  document_id = :doc_id
          AND  to_tsvector('english', search_text) @@ plainto_tsquery('english', :q)
          {page_clause}
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    params: dict[str, Any] = {"doc_id": document_id, "q": query, "k": k}
    if page_range is not None:
        params["lo"], params["hi"] = page_range[0], page_range[1]
    return db.execute(sql, params).fetchall()
```

- [ ] **Step 5: Route `hybrid_search` through the chosen arm**

In `hybrid_search`, replace the inline FTS block (lines 44-61) with:

```python
    if bm25_available(db):
        fts_rows = _keyword_search_bm25(
            db, document_id, query, page_range, settings.fts_top_k
        )
        arm = "bm25"
    else:
        fts_rows = _keyword_search_tsrank(
            db, document_id, query, page_range, settings.fts_top_k
        )
        arm = "ts_rank"
```

and extend the existing log line to record which arm ran:

```python
    logger.info(
        "hybrid_search: query=%.80s vec=%d keyword=%d arm=%s page_range=%s",
        query, len(vec_hits), len(fts_rows), arm, page_range,
    )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval.py -v`
Expected: PASS, with `test_bm25_search_matches_on_context` SKIPPED.

The GIN index from migration `0003` covers `to_tsvector('english', content)`, not `search_text`, so the fallback query will do a sequential scan. That is acceptable: it is a fallback, and scans are scoped to a single `document_id`.

- [ ] **Step 7: Commit**

```bash
git add app/services/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: BM25 keyword arm over context+content with ts_rank fallback

Keyword recall now searches search_text (context || content) instead of
content alone. Uses ParadeDB pg_search BM25 when the extension is present,
detected once per process, and falls back to the existing ts_rank query
otherwise so the app works without a db image rebuild."
```

---

## Task 5: Weighted RRF and larger candidate pool

**Files:**
- Modify: `backend/app/services/rag/retrieval.py:70-77` (`rrf_fuse`), `:139-161` (`retrieve`)
- Test: `backend/tests/test_retrieval.py` (append)

**Interfaces:**
- Consumes: `settings.rrf_weight_vector`, `settings.rrf_weight_keyword`, `settings.vector_top_k`, `settings.fts_top_k` (Task 1).
- Produces: `rrf_fuse(vec_hits, fts_rows, k, w_vec=None, w_keyword=None) -> list[tuple[Any, float]]` — existing call sites keep working unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_retrieval.py`:

```python
def test_rrf_fuse_weights_vector_above_keyword():
    """At equal rank, a vector-only hit must outrank a keyword-only hit under
    the default 0.8/0.2 split."""
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    result = rrf_fuse([C("V")], [C("K")], k=60)
    ids = [cid for cid, _ in result]

    assert ids[0] == "V"
    scores = dict(result)
    assert scores["V"] == pytest.approx(0.8 / 60)
    assert scores["K"] == pytest.approx(0.2 / 60)


def test_rrf_fuse_explicit_weights_override_settings():
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    result = rrf_fuse([C("V")], [C("K")], k=60, w_vec=0.1, w_keyword=0.9)
    assert [cid for cid, _ in result][0] == "K"


def test_rrf_fuse_scores_are_additive_across_arms():
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    # BOTH is rank 1 in each arm; VEC_ONLY is rank 0 in the vector arm only.
    vec = [C("VEC_ONLY"), C("BOTH")]
    kw = [C("KW_ONLY"), C("BOTH")]

    scores = dict(rrf_fuse(vec, kw, k=1))
    # BOTH: 0.8/2 + 0.2/2 = 0.5 ; VEC_ONLY: 0.8/1 = 0.8 ; KW_ONLY: 0.2/1 = 0.2
    assert scores["BOTH"] == pytest.approx(0.5)
    assert scores["VEC_ONLY"] == pytest.approx(0.8)
    assert scores["KW_ONLY"] == pytest.approx(0.2)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_retrieval.py -k rrf -v`
Expected: the two new weight tests FAIL (unweighted fusion gives `V` and `K` identical scores, and `rrf_fuse` rejects the `w_vec` kwarg). The pre-existing `test_rrf_fuse_combines_two_ranked_lists` must still PASS — under 0.8/0.2 its assertions still hold.

- [ ] **Step 3: Add weights to `rrf_fuse`**

Replace `rrf_fuse` in `backend/app/services/rag/retrieval.py`:

```python
def rrf_fuse(
    vec_hits,
    fts_rows,
    k: int,
    w_vec: float | None = None,
    w_keyword: float | None = None,
) -> list[tuple[Any, float]]:
    """Weighted Reciprocal Rank Fusion. Returns (id, score) sorted descending.

    Semantic search understands paraphrase; keyword search catches exact terms.
    The default 0.8/0.2 split follows the guideline's recommendation and is
    tunable via settings. Weights default to settings when not passed, so
    existing call sites keep working.
    """
    wv = settings.rrf_weight_vector if w_vec is None else w_vec
    wk = settings.rrf_weight_keyword if w_keyword is None else w_keyword
    scores: dict[Any, float] = defaultdict(float)
    for rank, chunk in enumerate(vec_hits):
        scores[chunk.id] += wv / (k + rank)
    for rank, row in enumerate(fts_rows):
        scores[row.id] += wk / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval.py -k rrf -v`
Expected: all PASS, including the pre-existing fusion test.

- [ ] **Step 5: Widen the candidate slice in `retrieve`**

`retrieve` truncates the fused list with `max(settings.vector_top_k, settings.fts_top_k)`, which discards half the pool now that both arms return 75. Replace that line (~line 147) with:

```python
    # Both arms return up to top_k, so the fused set can hold their sum. Keep
    # all of it: the reranker is what narrows to rerank_top_n, and starving it
    # of candidates is the one thing that reliably degrades rerank quality.
    candidate_limit = settings.vector_top_k + settings.fts_top_k
    fused_ids = [cid for cid, _ in fused[:candidate_limit]]
```

- [ ] **Step 6: Run the full suite and commit**

```bash
cd backend && pytest
git add app/services/rag/retrieval.py tests/test_retrieval.py
git commit -m "feat: weighted RRF (0.8 semantic / 0.2 keyword) and wider pool

Fusion was 50/50 by construction; the guideline recommends favouring
semantic recall. Weights are settings-driven and overridable per call.
retrieve() now keeps vector_top_k + fts_top_k fused candidates instead of
max(), which was silently halving the pool."
```

---

## Task 6: Contextual reranking

**Files:**
- Modify: `backend/app/services/rag/reranker.py:34-96` (`rerank`, add `_rerank_text`)
- Test: `backend/tests/test_retrieval.py` (amend five existing tests, append two)

**Interfaces:**
- Consumes: `DocumentChunk.context` (Task 1). `build_embedding_input` is deliberately **not** reused here — the reranker builds its own string so the two modules stay independent.
- Produces: `_rerank_text(chunk) -> str`. `rerank(query, chunks, top_n)` signature unchanged; the documents it POSTs now carry context + content.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_retrieval.py`:

```python
def test_rerank_sends_context_with_content(mocker):
    """The cross-encoder must see the same contextual signal the retrieval arms
    did, or it re-ranks on strictly less information than recall used."""
    from app.services.rag.reranker import rerank

    patched = _mock_rerank_response(
        mocker, {"results": [{"index": 0, "relevance_score": 0.9}]}
    )

    with_ctx = MagicMock(id="A", content="the rate is 40%", context="Section 3.")
    without_ctx = MagicMock(id="B", content="bare", context=None)

    rerank("q", [with_ctx, without_ctx], top_n=2)

    body = patched.return_value.__enter__.return_value.post.call_args.kwargs["json"]
    assert body["documents"] == [
        {"text": "Section 3.\n\nthe rate is 40%"},
        {"text": "bare"},
    ]


def test_rerank_handles_chunks_without_context_attribute(mocker):
    """Defensive: rerank is also called with plain objects that only carry
    .id and .content."""
    from app.services.rag.reranker import rerank

    patched = _mock_rerank_response(
        mocker, {"results": [{"index": 0, "relevance_score": 0.5}]}
    )

    class Bare:
        id = "A"
        content = "text only"

    rerank("q", [Bare()], top_n=1)

    body = patched.return_value.__enter__.return_value.post.call_args.kwargs["json"]
    assert body["documents"] == [{"text": "text only"}]
```

- [ ] **Step 2: Pin `context=None` on the pre-existing rerank mocks**

Five existing tests build chunks with `MagicMock`, which auto-creates a truthy `.context` attribute — that would now be prepended and break assertions that mean to test ordering, not content. Add `context=None` to every `MagicMock` chunk in:

- `test_rerank_sorts_by_score` (lines 48-50):

```python
    chunk_a = MagicMock(id="A", content="cat", context=None)
    chunk_b = MagicMock(id="B", content="dog", context=None)
    chunk_c = MagicMock(id="C", content="fish", context=None)
```

- `test_rerank_handles_bare_list_response` (lines 77-78):

```python
    chunk_a = MagicMock(id="A", content="x", context=None)
    chunk_b = MagicMock(id="B", content="y", context=None)
```

- `test_rerank_skips_out_of_range_index` (line 95):

```python
    chunk_a = MagicMock(id="A", content="x", context=None)
```

- `test_rerank_truncates_to_top_n` (line 114) and `test_rerank_falls_back_to_input_order_on_api_error` (line 127), both of which use the same comprehension:

```python
    chunks = [MagicMock(id=i, content=str(i), context=None) for i in range(3)]
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_retrieval.py -k rerank -v`
Expected: `test_rerank_sends_context_with_content` FAILS (documents still carry bare content). Everything amended in Step 2 PASSES.

- [ ] **Step 4: Build contextual documents in `rerank`**

In `backend/app/services/rag/reranker.py`, add this helper just above `rerank`:

```python
def _rerank_text(chunk: Any) -> str:
    """Text sent to the cross-encoder for one candidate.

    Includes the generated context when present so the reranker scores on the
    same signal the retrieval arms used. Tolerates objects without a .context
    attribute — rerank() is called with plain chunk-likes in places.
    """
    content = (getattr(chunk, "content", "") or "").strip()
    context = (getattr(chunk, "context", None) or "").strip()
    if not context:
        return content
    return f"{context}\n\n{content}"
```

Then replace the `documents` construction (line 50) with:

```python
    documents = [{"text": _rerank_text(c)} for c in chunks]
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_retrieval.py -k rerank -v`
Expected: all PASS

- [ ] **Step 6: Run the full suite and commit**

```bash
cd backend && pytest
git add app/services/rag/reranker.py tests/test_retrieval.py
git commit -m "feat: rerank on context + content

The cross-encoder was scoring bare chunk text while both recall arms used
context, so reranking ran on strictly less information than retrieval.
Tolerates chunks with no context attribute."
```

---

## Task 7: Documentation

**Files:**
- Modify: `wiki/02-flows.md` (flow 2 diagram + step 4; flow 3 step 4)
- Modify: `wiki/03-data-model.md` (intro, ERD, `document_chunks` section, migration table)
- Modify: `CLAUDE.md` ("Backend — RAG pipeline" and "Reingestion after schema changes")

**Interfaces:**
- Consumes: everything from Tasks 1-6. Produces no code.

- [ ] **Step 1: Update the ingestion flow diagram**

In `wiki/02-flows.md`, flow 2's mermaid diagram, replace the chunk/embed lines with:

```
    W->>W: chunk_document() (parent ~1500 tok, child ~300 tok)
    W->>OR: contextualize children (doc cached 1h, first call warms cache)
    OR-->>W: one context string per child
    W->>OR: embed (context + child text), batched
```

- [ ] **Step 2: Document the new ingestion steps**

In `wiki/02-flows.md`, flow 2's step 4 bullet list, insert after the `chunk_document()` bullet:

```markdown
   - `backend/app/services/contextualizer.py:contextualize_with_stats()` — one
     LLM call per child chunk generating a context string that situates it in
     the document. The document is sent as a 1-hour prompt-cached block; the
     first call is issued alone to warm that cache before the remaining calls
     fan out across `contextualizer_max_workers` threads. Documents over
     `contextualizer_full_doc_token_limit` (100k) fall back to a generated doc
     summary plus the child's own page. Per-chunk failures yield `None` and are
     non-fatal. Skipped entirely when `contextual_embeddings_enabled=False`.
   - `ingestion.py:build_embedding_input()` — embeds `context + "\n\n" + content`
     when context exists, bare `content` otherwise.
```

- [ ] **Step 3: Update the retrieval description**

In `wiki/02-flows.md`, flow 3 step 4, replace the `retrieve` bullet with:

```markdown
   - `retrieve` (registered node id; function `retrieve_and_rerank`,
     `nodes.py:56-71`) — calls `retrieval.retrieve()` (`retrieval.py`):
     `hybrid_search` (pgvector cosine + a keyword arm over `search_text`
     = `context || content` — ParadeDB `pg_search` BM25 when the extension is
     present, else the `ts_rank` fallback, decided once per process by
     `bm25_available()`) → `rrf_fuse` (**weighted** RRF, `k=60`, 0.8 semantic /
     0.2 keyword) → `rerank()` via OpenRouter `/v1/rerank` on
     `context + content` (top 6) → `apply_metadata_boost` (no-op by default) →
     `fetch_parents` (dedup parent chunks). Both arms return `top_k=75`, so the
     reranker sees up to 150 candidates.
```

- [ ] **Step 4: Update the data model**

In `wiki/03-data-model.md`:

Replace the opening sentence (the database description) with:

```markdown
Single Postgres 18 database (`paradedb/paradedb`, which bundles `pg_search` and
`pgvector`, see `docker-compose.yml` — note the volume mounts
`/var/lib/postgresql/`, not `/var/lib/postgresql/data`, for the Postgres 18
layout), schema owned entirely by SQLAlchemy models in `backend/app/models.py`
and versioned via Alembic (`backend/alembic/versions/`). No other datastore holds
relational/vector data — Redis is Celery-only (see `wiki/04-integrations.md`).
```

Add two fields to the `DOCUMENT_CHUNKS` ERD block, after `text content`:

```
        text context "nullable, LLM-generated"
        text search_text "generated: context || content"
```

Replace the `document_chunks` prose section with:

```markdown
### `document_chunks` (`DocumentChunk`, `models.py:45-69`)
One row per "child" passage (~300 tokens, 50-token overlap), the retrieval unit.
Carries the `pgvector` `embedding` column (`Vector(1536)`, migration
`0005_embedding_to_1536.py`), OCR provenance (`source`, `ocr_confidence`),
citation geometry (`bbox`, migration `0009`), and — since migration `0010` — the
generated `context` plus a `search_text` STORED generated column
(`coalesce(context,'') || ' ' || content`) that the `chunks_bm25` ParadeDB index
covers. `context` is nullable and NULL is fully supported: those rows retrieve on
content alone. `parent_id` is nullable "until reingest completes" per the model
comment.
- **Writes**: `services/ingestion.py:store_chunks()` (including `context`);
  wiped and rebuilt per-document by `scripts/reingest_all.py`. `search_text` is
  computed by Postgres — never assigned.
- **Reads**: `services/rag/retrieval.py` — vector similarity
  (`embedding.cosine_distance`) and a keyword arm over `search_text` (BM25 via
  `pg_search`, or the `ts_rank` fallback) in `hybrid_search()`, then re-fetched
  by id after weighted RRF fusion in `fetch_chunks_by_ids()`.
```

Add a row to the migration history table:

```markdown
| 0010 | `0010_contextual_retrieval.py` | `pg_search` extension; `document_chunks.context` + `search_text` generated column; `chunks_bm25` BM25 index |
```

- [ ] **Step 5: Update CLAUDE.md**

In the "Backend — RAG pipeline" section, replace the retrieval sentence with:

```markdown
Retrieval is hybrid: pgvector cosine similarity plus a keyword arm over
`search_text` (`context || content`), fused with **weighted** RRF (0.8 semantic
/ 0.2 keyword, both in settings). The keyword arm is ParadeDB `pg_search` BM25
when the extension is available and falls back to Postgres `ts_rank` otherwise —
detected once per process by `retrieval.bm25_available()`. Results are reranked
via OpenRouter's `/v1/rerank` cross-encoder on context + content (default
`anthropic/claude-haiku-4.5`; set `nvidia/llama-nemotron-rerank-vl-1b-v2:free`
for a dedicated reranker), then we return parent chunks (~1500 tokens) to the
LLM while children (~300 tokens) are what gets retrieved.

At ingestion, `services/contextualizer.py` generates a context string per child
chunk situating it in its source document, sending the document as a 1-hour
prompt-cached block. **The first call must complete before the rest fan out** —
a cache entry is only readable once the first response starts streaming, so a
concurrent fan-out silently costs ~10x. Disable with
`contextual_embeddings_enabled=False`. See
`docs/superpowers/specs/2026-07-28-contextual-retrieval-design.md`.
```

Add to the "Reingestion after schema changes" section:

```markdown
Migration `0010` added `context`; there is **no backfill**. Documents ingested
before it retrieve on content alone until re-ingested. Run
`python -m scripts.reingest_all` to contextualize them.
```

- [ ] **Step 6: Commit**

```bash
cd /d/development/chatbot
git add wiki/02-flows.md wiki/03-data-model.md CLAUDE.md
git commit -m "docs: document contextual retrieval in wiki and CLAUDE.md"
```

---

## Post-implementation verification

Not a task — these are the manual checks the spec calls for, which no test can cover.

- [ ] **Confirm prompt caching is actually working.** Ingest one document of 15+ pages and inspect the worker log. If OpenRouter is not passing `cache_control` through to Anthropic, contextualization still produces correct output at roughly 10x the cost, and nothing errors.

```bash
docker compose logs worker | grep -E "contextualize|cache"
```

  If cached-token counts are absent, add a temporary `logger.info("usage=%s", response.usage)` inside `_call_model` and re-ingest. Non-zero cached/read input tokens on calls *after* the first is the signal you want. Zero means the cache is not being hit — check block order first (the document block must be index 0).

- [ ] **A/B one document by hand.** The spec accepts having no eval harness, so this is the only quality signal before rollout:

```bash
cd backend
CONTEXTUAL_EMBEDDINGS_ENABLED=false python -m scripts.reingest_all --doc-id <uuid>
# ask 3-5 questions through the UI, record the citations
CONTEXTUAL_EMBEDDINGS_ENABLED=true python -m scripts.reingest_all --doc-id <uuid>
# ask the same questions, compare
```

- [ ] **Re-ingest everything else:** `python -m scripts.reingest_all`

- [ ] **Known gap, deliberately accepted:** `backend/evals/golden_set.yaml` is still the one-entry stub, so `run_eval --compare` cannot gate any of this. The 0.8/0.2 weights, the 75/75 pool sizes, the 100k threshold, and whether BM25 beats `ts_rank` are reasoned defaults, not measured ones. The `contextual_embeddings_enabled` and `bm25_enabled` flags exist so each can be A/B'd independently once a golden set is populated.
