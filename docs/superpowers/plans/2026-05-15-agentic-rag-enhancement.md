# Agentic RAG Enhancement Implementation Plan

> **Historical note (2026-06-28):** The stack moved off self-hosted Ollama/TEI.
> Chat LLM is now `anthropic/claude-haiku-4.5` via OpenRouter, embeddings are
> OpenAI `text-embedding-3-small` (1536d), and the reranker is LLM-as-reranker
> via OpenRouter (no more bge-reranker-v2-m3 / FlashRank). Content below
> describes the original 2026-05-15 plan and is kept for historical context.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle regex-preprocessor + 3-round `bind_tools` loop in `backend/app/services/rag.py` with a LangGraph CRAG-lite state machine, parent-child chunking, RRF hybrid-fusion + FlashRank reranker, and a RAGAS golden-set evaluation harness — fixing the "answer is in the doc but agent misses it" failure mode.

**Architecture:** Phased rollout. Phase 0 builds a baseline eval. Phase 1 introduces a parent/child chunk schema and reingests. Phase 2 swaps in RRF fusion + cross-encoder reranking and pipes parents into the prompt. Phase 3 rewrites the agent as a LangGraph state machine with an LLM query rewriter, grading, retry, and faithfulness check. Phase 4 polishes logs and docs.

**Tech Stack:** Python 3 / FastAPI / SQLAlchemy 2 / Postgres + pgvector / Alembic / LangChain / LangGraph / Ollama / RAGAS / FlashRank.

**Design spec:** `docs/superpowers/specs/2026-05-15-agentic-rag-enhancement-design.md`

---

## File structure

### Files to create

| Path | Purpose |
|---|---|
| `backend/evals/__init__.py` | Marker |
| `backend/evals/golden_set.yaml` | Q/A pairs grouped by document |
| `backend/evals/metrics.py` | RAGAS wrapper + custom checks |
| `backend/evals/run_eval.py` | CLI runner; `python -m evals.run_eval` |
| `backend/evals/results/.gitkeep` | Keep dir tracked |
| `backend/alembic/versions/0004_parent_child_chunks.py` | Migration |
| `backend/scripts/__init__.py` | Marker |
| `backend/scripts/reingest_all.py` | One-shot reingest |
| `backend/app/services/rag/__init__.py` | Re-export `agentic_rag_stream` |
| `backend/app/services/rag/state.py` | `AgentState` TypedDict + helpers |
| `backend/app/services/rag/prompts.py` | All prompt constants |
| `backend/app/services/rag/reranker.py` | FlashRank singleton |
| `backend/app/services/rag/retrieval.py` | Hybrid + RRF + rerank + fetch parents |
| `backend/app/services/rag/nodes.py` | Six LangGraph nodes |
| `backend/app/services/rag/graph.py` | State machine wiring + entry point |
| `backend/tests/test_retrieval.py` | Unit tests for `retrieval.py` |
| `backend/tests/test_rag_nodes.py` | Unit tests for each node |
| `backend/tests/test_rag_graph.py` | Wiring tests with mocked nodes |

### Files to modify

| Path | Change |
|---|---|
| `backend/requirements.txt` | Add `ragas`, `datasets`, `langgraph`, `flashrank`, `pyyaml` |
| `backend/app/models.py` | Add `DocumentParentChunk` model, add `parent_id` FK on `DocumentChunk` |
| `backend/app/services/ingestion.py` | Produce parents + children; write both tables |
| `backend/app/services/rag.py` | **Deleted at end of Phase 3** (becomes the new package) |
| `backend/tests/test_rag.py` | Updated for new shape, old tests deleted/rewritten |
| `backend/tests/test_ingestion.py` | Updated for parent/child output |
| `backend/Dockerfile` | Cache FlashRank model in image (Phase 2) |
| `backend/app/config.py` | Add retrieval tunables |
| `features/chat_with_doc/rag_enhancement.md` | Replaced by Phase 4 with final architecture summary |
| `CLAUDE.md` | Phase 4: point to new `rag/` module layout |

---

# Phase 0 — Baseline evaluation

**Goal:** Build a golden set + RAGAS runner, capture a baseline score against today's `rag.py`. Without this, "the reranker helped" is just a feeling.

### Task 0.1: Add evaluation dependencies

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Add dependencies**

Append to `backend/requirements.txt`:

```
ragas==0.2.6
datasets==3.0.1
pyyaml==6.0.2
```

(Pin to specific versions to keep evals reproducible. `ragas` pulls in `pandas`/`datasets` transitively but we list `datasets` explicitly because we use `Dataset.from_list` directly.)

- [ ] **Step 2: Rebuild and verify**

Run: `cd backend && pip install -r requirements.txt` (or `docker compose build backend`).
Expected: install completes; `python -c "import ragas; import datasets; import yaml"` exits 0.

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt
git commit -m "deps: add ragas + datasets + pyyaml for eval harness"
```

---

### Task 0.2: Create evals package skeleton

**Files:**
- Create: `backend/evals/__init__.py`
- Create: `backend/evals/results/.gitkeep`

- [ ] **Step 1: Create package marker**

`backend/evals/__init__.py` — empty file.

- [ ] **Step 2: Create results dir**

```bash
mkdir -p backend/evals/results
touch backend/evals/results/.gitkeep
```

- [ ] **Step 3: Update .gitignore so result JSONs are tracked deliberately**

Append to root `.gitignore`:

```
# Eval results: track baselines only, ignore ad-hoc local runs
backend/evals/results/*.json
!backend/evals/results/baseline_*.json
```

- [ ] **Step 4: Commit**

```bash
git add backend/evals/__init__.py backend/evals/results/.gitkeep .gitignore
git commit -m "feat(evals): add evals package skeleton"
```

---

### Task 0.3: Build the golden Q/A set

**Files:**
- Create: `backend/evals/golden_set.yaml`

**Prerequisite:** A copy of each referenced document must exist in `backend/uploads/` (or wherever `UPLOAD_DIR` points) AND be ingested with `status='ready'`. If you don't have suitable docs, ingest 3-4 short PDFs first via the existing upload UI.

- [ ] **Step 1: Inspect uploaded documents**

Run: `psql -d chatbot -c "SELECT id, file_name, status FROM documents WHERE status='ready' ORDER BY uploaded_at DESC LIMIT 10;"`

Pick 3-4 documents that have a mix of structured fields and prose.

- [ ] **Step 2: Write `backend/evals/golden_set.yaml`**

Open each document, pick ~5 Q/A pairs per document. Categories (cover all five):

```yaml
# backend/evals/golden_set.yaml
# Aim for 15-20 entries across 3-4 documents.
# Categories: named_field_lookup, fact_lookup, summarization, not_in_doc, pronoun_dependent

- document_file_name: "ACME_Corp_Registration.pdf"
  question: "What is the Corporate Name?"
  expected_answer: "ACME Corporation Ltd."
  expected_substring: "ACME Corporation"  # used for cheap substring check
  category: "named_field_lookup"

- document_file_name: "ACME_Corp_Registration.pdf"
  question: "When was the company registered?"
  expected_answer: "March 15, 2018"
  expected_substring: "March 15, 2018"
  category: "fact_lookup"

- document_file_name: "ACME_Corp_Registration.pdf"
  question: "Summarize the company purpose."
  expected_answer: "Software development and consulting services."
  expected_substring: null  # no exact substring expected for summaries
  category: "summarization"

- document_file_name: "ACME_Corp_Registration.pdf"
  question: "What is the CEO's home address?"
  expected_answer: "Not stated in the document."
  expected_substring: null
  category: "not_in_doc"

- document_file_name: "ACME_Corp_Registration.pdf"
  question: "And when was it last amended?"  # pronoun "it" -> "the registration"
  expected_answer: "January 4, 2024"
  expected_substring: "January 4, 2024"
  category: "pronoun_dependent"

# Repeat for documents 2, 3, 4 with their own field lookups, fact lookups, summaries, not-in-doc, pronoun questions.
```

Aim for at least 3 entries per category across all docs. Be realistic — short fact answers in `expected_substring`; null for summaries and not-in-doc.

- [ ] **Step 3: Validate YAML parses**

Run: `python -c "import yaml; d = yaml.safe_load(open('backend/evals/golden_set.yaml')); print(len(d), 'entries'); print({e['category'] for e in d})"`

Expected: prints entry count + set of categories.

- [ ] **Step 4: Commit**

```bash
git add backend/evals/golden_set.yaml
git commit -m "feat(evals): add golden Q/A set"
```

---

### Task 0.4: Implement metrics.py

**Files:**
- Create: `backend/evals/metrics.py`

- [ ] **Step 1: Write `backend/evals/metrics.py`**

```python
"""RAGAS metric wrappers + cheap custom checks.

Judge model = ollama_chat_model from app.config.settings, wrapped via
langchain_ollama.ChatOllama so it can be passed to ragas.evaluate(llm=...).
"""
from __future__ import annotations
import logging
from typing import Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings

logger = logging.getLogger(__name__)


def _build_judge() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=0.0,
    )


def _build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )


def run_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, list[float]]:
    """Run RAGAS metrics. Each input list must have the same length."""
    n = len(questions)
    assert len(answers) == len(contexts) == len(ground_truths) == n, (
        "metrics inputs must be same length"
    )
    ds = Dataset.from_list([
        {
            "question": questions[i],
            "answer": answers[i],
            "contexts": contexts[i],
            "ground_truth": ground_truths[i],
        }
        for i in range(n)
    ])
    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_build_judge(),
        embeddings=_build_embeddings(),
    )
    df = result.to_pandas()
    return {
        "faithfulness": df["faithfulness"].fillna(0.0).tolist(),
        "answer_relevancy": df["answer_relevancy"].fillna(0.0).tolist(),
        "context_precision": df["context_precision"].fillna(0.0).tolist(),
        "context_recall": df["context_recall"].fillna(0.0).tolist(),
    }


def expected_substring_match(answer: str, expected_substring: str | None) -> bool:
    """Case-insensitive substring check. Returns True if expected_substring is None
    (i.e. not applicable, like summaries and not_in_doc)."""
    if not expected_substring:
        return True
    return expected_substring.lower() in (answer or "").lower()


def answered(answer: str) -> bool:
    """True if the agent produced a non-empty, non-disclaimer answer."""
    if not answer or len(answer.strip()) < 3:
        return False
    return True


def summarize(per_question: list[dict[str, Any]]) -> dict[str, float]:
    """Mean-of-floats summary across all questions, plus boolean rates."""
    def mean(key: str) -> float:
        vals = [q["metrics"].get(key) for q in per_question if q["metrics"].get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    n = len(per_question)
    return {
        "n": n,
        "faithfulness_mean": mean("faithfulness"),
        "answer_relevancy_mean": mean("answer_relevancy"),
        "context_precision_mean": mean("context_precision"),
        "context_recall_mean": mean("context_recall"),
        "answered_rate": sum(1 for q in per_question if q["answered"]) / n if n else 0.0,
        "expected_substring_match_rate": (
            sum(1 for q in per_question if q["expected_substring_match"]) / n if n else 0.0
        ),
    }
```

- [ ] **Step 2: Smoke-import**

Run: `cd backend && python -c "from evals.metrics import expected_substring_match, answered, summarize; print(expected_substring_match('Hello World', 'world'))"`

Expected: `True`.

- [ ] **Step 3: Commit**

```bash
git add backend/evals/metrics.py
git commit -m "feat(evals): metrics module (RAGAS + custom checks)"
```

---

### Task 0.5: Implement run_eval.py

**Files:**
- Create: `backend/evals/run_eval.py`

- [ ] **Step 1: Write `backend/evals/run_eval.py`**

```python
"""CLI runner for the golden Q/A set.

Usage:
  python -m evals.run_eval --name baseline
  python -m evals.run_eval --name after_rerank
  python -m evals.run_eval --compare baseline after_rerank
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import yaml

from app.database import SessionLocal
from app.models import Document
from app.services.rag import agentic_rag_stream
from evals.metrics import (
    answered,
    expected_substring_match,
    run_ragas_metrics,
    summarize,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_set.yaml"
RESULTS_DIR = Path(__file__).parent / "results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_golden() -> list[dict]:
    with open(GOLDEN_PATH) as f:
        return yaml.safe_load(f)


def _resolve_document_id(db, file_name: str) -> str:
    doc = (
        db.query(Document)
        .filter(Document.file_name == file_name, Document.status == "ready")
        .first()
    )
    if not doc:
        raise RuntimeError(
            f"golden_set.yaml references {file_name!r} but no ready document with that "
            f"name exists in the database. Ingest it first."
        )
    return str(doc.id)


async def _run_one(db, document_id: str, question: str) -> tuple[str, list[str]]:
    """Stream the RAG pipeline and collect final answer + citation texts."""
    answer_parts: list[str] = []
    contexts: list[str] = []
    async for event in agentic_rag_stream(document_id, question, db):
        if event["type"] == "token":
            answer_parts.append(event["content"])
        elif event["type"] == "citations":
            contexts = [c["content"] for c in event["chunks"]]
    return "".join(answer_parts).strip(), contexts


async def run_eval(name: str) -> dict:
    golden = _load_golden()
    db = SessionLocal()
    per_question: list[dict] = []
    try:
        # First pass: produce answers + contexts
        questions, answers, ctx_lists, gts = [], [], [], []
        for entry in golden:
            doc_id = _resolve_document_id(db, entry["document_file_name"])
            logger.info("Running question: %s", entry["question"][:80])
            answer, contexts = await _run_one(db, doc_id, entry["question"])
            questions.append(entry["question"])
            answers.append(answer)
            ctx_lists.append(contexts or [""])  # ragas dislikes empty contexts
            gts.append(entry["expected_answer"])

        # RAGAS metrics in batch
        logger.info("Running RAGAS judge over %d questions (this is slow)...", len(questions))
        metrics_by_key = run_ragas_metrics(questions, answers, ctx_lists, gts)

        # Per-question record
        for i, entry in enumerate(golden):
            per_question.append({
                "category": entry["category"],
                "question": entry["question"],
                "expected_answer": entry["expected_answer"],
                "agent_answer": answers[i],
                "answered": answered(answers[i]),
                "expected_substring_match": expected_substring_match(
                    answers[i], entry.get("expected_substring")
                ),
                "metrics": {
                    "faithfulness": metrics_by_key["faithfulness"][i],
                    "answer_relevancy": metrics_by_key["answer_relevancy"][i],
                    "context_precision": metrics_by_key["context_precision"][i],
                    "context_recall": metrics_by_key["context_recall"][i],
                },
            })
    finally:
        db.close()

    payload = {
        "run_name": name,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "git_sha": _git_sha(),
        "config_snapshot": {
            "ollama_chat_model": os.environ.get("OLLAMA_CHAT_MODEL", "from-settings"),
        },
        "per_question": per_question,
        "summary": summarize(per_question),
    }
    return payload


def _save(payload: dict, name: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = RESULTS_DIR / f"{name}_{date}.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def _compare(name_a: str, name_b: str) -> None:
    def _latest(name: str) -> dict:
        candidates = sorted(RESULTS_DIR.glob(f"{name}_*.json"))
        if not candidates:
            raise RuntimeError(f"no results file matching {name}_*.json")
        return json.loads(candidates[-1].read_text())

    a, b = _latest(name_a), _latest(name_b)
    print(f"\nComparing {name_a} vs {name_b}")
    print("-" * 60)
    for key in ["faithfulness_mean", "answer_relevancy_mean",
                "context_precision_mean", "context_recall_mean",
                "answered_rate", "expected_substring_match_rate"]:
        va, vb = a["summary"].get(key, 0.0), b["summary"].get(key, 0.0)
        delta = vb - va
        arrow = "up" if delta > 0 else ("down" if delta < 0 else "=")
        print(f"  {key:38s}  {va:.3f} -> {vb:.3f}  {arrow}{delta:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="Run name (e.g. 'baseline', 'after_rerank')")
    parser.add_argument(
        "--compare", nargs=2, metavar=("A", "B"), help="Compare two existing runs"
    )
    args = parser.parse_args()

    if args.compare:
        _compare(*args.compare)
        return
    if not args.name:
        parser.error("provide --name or --compare")
    payload = asyncio.run(run_eval(args.name))
    out = _save(payload, args.name)
    print(f"\nWrote {out}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses**

Run: `cd backend && python -m evals.run_eval --help`
Expected: prints argparse help with `--name` and `--compare`.

- [ ] **Step 3: Commit**

```bash
git add backend/evals/run_eval.py
git commit -m "feat(evals): CLI runner for golden Q/A set"
```

---

### Task 0.6: Capture baseline run

**Files:**
- Create: `backend/evals/results/baseline_<date>.json` (output)

**Prerequisite:** Backend services running locally (`docker compose up postgres ollama`). Ollama has both `ollama_chat_model` and `ollama_embedding_model` pulled.

- [ ] **Step 1: Run baseline**

Run: `cd backend && python -m evals.run_eval --name baseline`

Expected: runs through all golden questions (slow — ~5-15 min). Prints summary JSON. Writes `backend/evals/results/baseline_<date>.json`.

- [ ] **Step 2: Inspect baseline**

Run: `cat backend/evals/results/baseline_*.json | python -m json.tool | tail -20`

Expected: see the `summary` block. Make a mental note of `expected_substring_match_rate` and `context_recall_mean` — these are the headline numbers Phase 2 must beat.

- [ ] **Step 3: Commit baseline**

```bash
git add backend/evals/results/baseline_*.json
git commit -m "evals: capture baseline against current rag.py"
```

This is the regression target for the rest of the project.

---

### Task 0.7: Register `eval` pytest marker (so `pytest -m eval` works without polluting normal runs)

**Files:**
- Create: `backend/pytest.ini` (or update `pyproject.toml` if one exists)
- Create: `backend/tests/test_eval_golden.py`

- [ ] **Step 1: Add marker config**

Create `backend/pytest.ini` if it doesn't exist:

```ini
[pytest]
markers =
    eval: slow golden-set evaluation (excluded from default runs)
addopts = -m "not eval"
```

If `backend/pyproject.toml` exists, add the equivalent `[tool.pytest.ini_options]` block instead.

- [ ] **Step 2: Add a pytest entrypoint to the golden runner**

Create `backend/tests/test_eval_golden.py`:

```python
"""pytest entrypoint for the golden set. Runs only when `-m eval` is passed.

Usage:
  pytest -m eval -s backend/tests/test_eval_golden.py
"""
import asyncio
import pytest

from evals.run_eval import run_eval, _save


@pytest.mark.eval
def test_golden_set_runs_and_summary_meets_thresholds():
    payload = asyncio.run(run_eval("pytest_marker_run"))
    _save(payload, "pytest_marker_run")
    s = payload["summary"]
    # Thresholds from spec §8.3 — relaxed slightly because this can run pre-Phase-2
    assert s["faithfulness_mean"] >= 0.70, f"faithfulness too low: {s}"
    assert s["answered_rate"] >= 0.80, f"answered_rate too low: {s}"
```

- [ ] **Step 3: Verify default pytest run still skips it**

Run: `cd backend && pytest tests/ -v`
Expected: `test_golden_set_runs_and_summary_meets_thresholds` is collected and **deselected** because of the default `-m "not eval"`.

- [ ] **Step 4: Verify the eval marker works**

Run: `cd backend && pytest -m eval -v -s` (only when you actually want to run the golden set)
Expected: the single test runs (slow). Skip this step if Ollama isn't running.

- [ ] **Step 5: Commit**

```bash
git add backend/pytest.ini backend/tests/test_eval_golden.py
git commit -m "test(evals): register eval pytest marker; gate golden run behind -m eval"
```

---

# Phase 1 — Parent-child chunking

**Goal:** Add a parent chunk table, make `document_chunks` the child layer, switch ingestion to produce both layers, and reingest existing documents.

### Task 1.1: Add DocumentParentChunk model + parent_id column

**Files:**
- Modify: `backend/app/models.py`

- [ ] **Step 1: Write test first**

Append to `backend/tests/test_ingestion.py`:

```python
def test_document_parent_chunk_model_round_trips(db):
    from app.models import Document, DocumentParentChunk, DocumentChunk
    import uuid

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="ready")
    db.add(doc)
    parent = DocumentParentChunk(
        document_id=doc.id,
        parent_index=0,
        content="Parent body of text...",
    )
    db.add(parent)
    db.flush()

    child = DocumentChunk(
        document_id=doc.id,
        parent_id=parent.id,
        chunk_index=0,
        content="Child snippet",
        embedding=[0.0] * 2560,
    )
    db.add(child)
    db.flush()

    fetched = db.query(DocumentChunk).filter_by(id=child.id).one()
    assert fetched.parent_id == parent.id
```

- [ ] **Step 2: Run test (should fail — model missing)**

Run: `cd backend && pytest tests/test_ingestion.py::test_document_parent_chunk_model_round_trips -v`
Expected: FAIL `AttributeError: module 'app.models' has no attribute 'DocumentParentChunk'`.

- [ ] **Step 3: Update `backend/app/models.py`**

Replace the file with:

```python
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey, UniqueConstraint
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


class DocumentParentChunk(Base):
    __tablename__ = "document_parent_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("document_id", "parent_index", name="uq_dpc_doc_idx"),
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    parent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_parent_chunks.id", ondelete="CASCADE"),
        nullable=True,  # nullable until reingest completes
    )
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(2560))
```

`parent_id` is nullable so existing rows (pre-reingest) remain valid until they're swapped out.

- [ ] **Step 4: Run test (should pass)**

Run: `cd backend && pytest tests/test_ingestion.py::test_document_parent_chunk_model_round_trips -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/models.py backend/tests/test_ingestion.py
git commit -m "feat(rag): add DocumentParentChunk model + parent_id FK"
```

---

### Task 1.2: Alembic migration 0004

**Files:**
- Create: `backend/alembic/versions/0004_parent_child_chunks.py`

- [ ] **Step 1: Write the migration**

```python
"""parent-child chunk schema: add document_parent_chunks + parent_id FK

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-15
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_parent_chunks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("document_id", UUID(as_uuid=True), nullable=False),
        sa.Column("parent_index", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("document_id", "parent_index", name="uq_dpc_doc_idx"),
    )
    op.create_index("ix_dpc_doc", "document_parent_chunks", ["document_id"])

    op.add_column(
        "document_chunks",
        sa.Column("parent_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_dc_parent",
        "document_chunks",
        "document_parent_chunks",
        ["parent_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_index("ix_dc_parent", "document_chunks", ["parent_id"])


def downgrade() -> None:
    op.drop_index("ix_dc_parent", table_name="document_chunks")
    op.drop_constraint("fk_dc_parent", "document_chunks", type_="foreignkey")
    op.drop_column("document_chunks", "parent_id")

    op.drop_index("ix_dpc_doc", table_name="document_parent_chunks")
    op.drop_table("document_parent_chunks")
```

- [ ] **Step 2: Run migration**

Run: `cd backend && alembic upgrade head`
Expected: `Running upgrade 0003 -> 0004, parent-child chunk schema...`.

- [ ] **Step 3: Verify schema**

Run: `psql -d chatbot -c "\d document_parent_chunks" -c "\d document_chunks"`
Expected: `document_parent_chunks` table exists; `document_chunks` has `parent_id uuid` column with FK + index.

- [ ] **Step 4: Verify downgrade reversibility (then re-upgrade)**

Run:
```bash
alembic downgrade 0003
alembic upgrade head
```
Expected: both succeed without error.

- [ ] **Step 5: Commit**

```bash
git add backend/alembic/versions/0004_parent_child_chunks.py
git commit -m "feat(rag): alembic 0004 — parent-child chunk schema"
```

---

### Task 1.3: Update ingestion to produce parents + children

**Files:**
- Modify: `backend/app/services/ingestion.py`
- Modify: `backend/tests/test_ingestion.py`

- [ ] **Step 1: Write test first**

Append to `backend/tests/test_ingestion.py`:

```python
def test_chunk_text_produces_parents_and_children():
    from app.services.ingestion import chunk_text

    # Produce text large enough to split into multiple parents
    text = ("This is a sentence. " * 800)  # ~16k characters → multiple 1500-token parents
    parents, children_by_parent = chunk_text(text)

    assert len(parents) >= 2, "should split into multiple parents"
    # children_by_parent is a list aligned with parents — each entry is a list of child strings
    assert len(children_by_parent) == len(parents)
    for parent_text, children in zip(parents, children_by_parent):
        assert children, "every parent must have at least one child"
        # children should be shorter than the parent
        joined = " ".join(children)
        # Tokens are smaller than chars, but length sanity: children re-joined should
        # roughly cover the parent (allow slack for overlap and whitespace)
        assert len(joined) >= len(parent_text) * 0.7
```

- [ ] **Step 2: Run test (should fail — current chunk_text returns flat list)**

Run: `cd backend && pytest tests/test_ingestion.py::test_chunk_text_produces_parents_and_children -v`
Expected: FAIL — tuple-unpack error or attribute error.

- [ ] **Step 3: Rewrite `chunk_text` in `backend/app/services/ingestion.py`**

Replace the file with:

```python
import logging
import os
from typing import List, Tuple
import fitz  # PyMuPDF
import httpx
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy.orm import Session

from ..config import settings
from ..models import DocumentChunk, DocumentParentChunk

logger = logging.getLogger(__name__)


def parse_file(file_path: str, file_name: str) -> str:
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".pdf":
        doc = fitz.open(file_path)
        text = "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        doc = DocxDocument(file_path)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    else:
        text = f"[image: {file_name}]"
    logger.info("parse_file: file=%s type=%s text_len=%d", file_name, ext or "image", len(text))
    return text


def _parent_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=1500,
        chunk_overlap=0,
    )


def _child_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=300,
        chunk_overlap=50,
    )


def chunk_text(text: str) -> Tuple[List[str], List[List[str]]]:
    """Returns (parents, children_per_parent). children_per_parent[i] are the child
    chunks derived from parents[i]."""
    parents = _parent_splitter().split_text(text)
    child_splitter = _child_splitter()
    children_per_parent = [child_splitter.split_text(p) for p in parents]
    n_children = sum(len(c) for c in children_per_parent)
    logger.info(
        "chunk_text: input_len=%d parents=%d children=%d",
        len(text), len(parents), n_children,
    )
    return parents, children_per_parent


def embed_text(text: str) -> List[float]:
    with httpx.Client() as client:
        response = client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.ollama_embedding_model, "input": [text]},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    embeddings: List[List[float]] = []
    batch_size = 100
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    logger.info("embed_chunks: total=%d batches=%d", len(chunks), total_batches)
    with httpx.Client() as client:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            logger.debug("embed_chunks: batch=%d/%d size=%d", batch_num, total_batches, len(batch))
            response = client.post(
                f"{settings.ollama_base_url}/api/embed",
                json={"model": settings.ollama_embedding_model, "input": batch},
            )
            response.raise_for_status()
            embeddings.extend(response.json()["embeddings"])
    logger.info("embed_chunks: done embeddings=%d", len(embeddings))
    return embeddings


def store_chunks(
    db: Session,
    document_id: str,
    parents: List[str],
    children_per_parent: List[List[str]],
    child_embeddings: List[List[float]],
) -> None:
    """Insert parents then children. Children carry parent_id."""
    parent_rows = [
        DocumentParentChunk(
            document_id=document_id,
            parent_index=i,
            content=parent_text,
        )
        for i, parent_text in enumerate(parents)
    ]
    db.add_all(parent_rows)
    db.flush()  # populate parent_rows[*].id

    child_rows: List[DocumentChunk] = []
    embed_iter = iter(child_embeddings)
    global_idx = 0
    for parent_row, children in zip(parent_rows, children_per_parent):
        for child_text in children:
            child_rows.append(DocumentChunk(
                document_id=document_id,
                parent_id=parent_row.id,
                chunk_index=global_idx,
                content=child_text,
                embedding=next(embed_iter),
            ))
            global_idx += 1
    db.bulk_save_objects(child_rows)
    db.commit()
    logger.info(
        "store_chunks: parents=%d children=%d document_id=%s",
        len(parent_rows), len(child_rows), document_id,
    )
```

- [ ] **Step 4: Run test (should pass)**

Run: `cd backend && pytest tests/test_ingestion.py::test_chunk_text_produces_parents_and_children -v`
Expected: PASS.

- [ ] **Step 5: Update the Celery task that calls these**

Inspect `backend/app/workers/tasks.py`. Find the place that calls `chunk_text`, `embed_chunks`, `store_chunks` and update the call sites to match the new signatures.

The flow becomes:
```python
parents, children_per_parent = chunk_text(text)
flat_children = [c for sub in children_per_parent for c in sub]
embeddings = embed_chunks(flat_children)
store_chunks(db, document_id, parents, children_per_parent, embeddings)
```

- [ ] **Step 6: Run all ingestion tests**

Run: `cd backend && pytest tests/test_ingestion.py tests/test_tasks.py -v`
Expected: all pass. Fix any failures by adjusting test mocks to the new shape.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/ingestion.py backend/app/workers/tasks.py backend/tests/test_ingestion.py
git commit -m "feat(rag): ingestion produces parent+child chunks"
```

---

### Task 1.4: Reingest script

**Files:**
- Create: `backend/scripts/__init__.py`
- Create: `backend/scripts/reingest_all.py`

- [ ] **Step 1: Create marker**

`backend/scripts/__init__.py` — empty.

- [ ] **Step 2: Write `backend/scripts/reingest_all.py`**

```python
"""One-shot reingest: for every Document with status='ready' (or 'failed'),
delete its existing chunks and re-run the ingestion pipeline using the
new parent-child chunker. Per-document transaction; on crash the document
is left in status='failed' for the user to retry from the UI.

Usage:
  python -m scripts.reingest_all                  # all ready docs
  python -m scripts.reingest_all --include-failed # also retry previously failed
  python -m scripts.reingest_all --doc-id <uuid>  # single doc
"""
from __future__ import annotations
import argparse
import logging
from uuid import UUID

from sqlalchemy import select, delete

from app.database import SessionLocal
from app.models import Document, DocumentChunk, DocumentParentChunk
from app.services.ingestion import (
    parse_file,
    chunk_text,
    embed_chunks,
    store_chunks,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def reingest_one(doc_id: UUID) -> None:
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            logger.error("doc not found: %s", doc_id)
            return
        logger.info("reingest start: %s (%s)", doc_id, doc.file_name)

        # Delete in dependency order; cascades handle children rows
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc_id))
        db.execute(delete(DocumentParentChunk).where(DocumentParentChunk.document_id == doc_id))
        db.commit()

        text = parse_file(doc.file_path, doc.file_name)
        parents, children_per_parent = chunk_text(text)
        flat_children = [c for sub in children_per_parent for c in sub]
        embeddings = embed_chunks(flat_children)
        store_chunks(db, str(doc_id), parents, children_per_parent, embeddings)

        doc.status = "ready"
        doc.error_msg = None
        db.commit()
        logger.info("reingest done: %s", doc_id)
    except Exception as e:
        db.rollback()
        doc = db.get(Document, doc_id)
        if doc:
            doc.status = "failed"
            doc.error_msg = f"reingest: {e}"
            db.commit()
        logger.exception("reingest failed: %s", doc_id)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", help="UUID of a single document")
    parser.add_argument("--include-failed", action="store_true",
                        help="Also reingest documents currently in status='failed'")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        if args.doc_id:
            ids = [UUID(args.doc_id)]
        else:
            statuses = ["ready"] + (["failed"] if args.include_failed else [])
            rows = db.execute(
                select(Document.id).where(Document.status.in_(statuses))
            ).scalars().all()
            ids = list(rows)
    finally:
        db.close()

    logger.info("reingesting %d documents", len(ids))
    for doc_id in ids:
        reingest_one(doc_id)
    logger.info("all done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Dry-run with a single doc**

Pick a doc with `status='ready'`:
```bash
psql -d chatbot -tA -c "SELECT id FROM documents WHERE status='ready' LIMIT 1;"
```
Then:
```bash
cd backend && python -m scripts.reingest_all --doc-id <uuid>
```
Expected: logs `reingest start ... reingest done`.

- [ ] **Step 4: Verify the doc has parents now**

```bash
psql -d chatbot -c "SELECT count(*) AS parents FROM document_parent_chunks WHERE document_id='<uuid>';"
psql -d chatbot -c "SELECT count(*) FILTER (WHERE parent_id IS NOT NULL) AS children FROM document_chunks WHERE document_id='<uuid>';"
```
Expected: parents > 0; children > parents; every child has `parent_id` set.

- [ ] **Step 5: Reingest all**

Run: `cd backend && python -m scripts.reingest_all`
Expected: cycles through every ready doc. Slow (embedding bottleneck).

- [ ] **Step 6: Commit**

```bash
git add backend/scripts/__init__.py backend/scripts/reingest_all.py
git commit -m "feat(rag): reingest_all script for parent-child migration"
```

---

### Task 1.5: Re-run baseline against reingested data

This proves Phase 1 didn't regress anything by itself (chunks just got smaller).

- [ ] **Step 1: Run eval**

Run: `cd backend && python -m evals.run_eval --name after_chunking`

- [ ] **Step 2: Compare**

Run: `cd backend && python -m evals.run_eval --compare baseline after_chunking`
Expected: most metrics flat or slightly improved (smaller chunks help context_precision; FTS recall may dip). If `expected_substring_match_rate` collapses, investigate before continuing — likely a bug in `store_chunks` or `chunk_text`.

- [ ] **Step 3: Commit result**

```bash
git add backend/evals/results/after_chunking_*.json
git commit -m "evals: capture after-chunking results"
```

---

# Phase 2 — RRF + FlashRank reranker

**Goal:** Replace the current vector+FTS merge with RRF fusion, add a cross-encoder reranker, and start feeding parents (not children) to the LLM.

### Task 2.1: Add flashrank + retrieval tunables

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/config.py`

- [ ] **Step 1: Add flashrank to requirements**

Append:
```
flashrank==0.2.10
langgraph==0.2.50
```

(Adding `langgraph` here too even though it's used in Phase 3 — saves a second image rebuild.)

- [ ] **Step 2: Install**

Run: `cd backend && pip install -r requirements.txt`
Expected: succeeds.

- [ ] **Step 3: Add retrieval tunables to `backend/app/config.py`**

Replace the file with:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/"
    ollama_embedding_model: str = "qwen3-embedding:4b"
    ollama_chat_model: str = "qwen3:4b-instruct-2507-q8_0"
    upload_dir: str = "./uploads"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    # Retrieval tunables
    vector_top_k: int = 30
    fts_top_k: int = 30
    rrf_k: int = 60
    rerank_top_n: int = 6
    rerank_score_floor: float = 0.05
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = "/tmp/flashrank"

    # Agent loop tunables
    max_retrieval_retries: int = 2
    strict_grader: bool = False  # if True, use LLM grader instead of rule-based

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
```

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt backend/app/config.py
git commit -m "deps: flashrank + langgraph; add retrieval/agent tunables"
```

---

### Task 2.2: FlashRank singleton

**Files:**
- Create: `backend/app/services/rag/__init__.py`
- Create: `backend/app/services/rag/reranker.py`

We start migrating `rag.py` into `rag/`. To keep imports working while Phase 2 lands, the new `rag/__init__.py` will re-export from the *old* `rag.py` (moved aside) for now; in Phase 3 we replace the contents entirely.

- [ ] **Step 1: Move `backend/app/services/rag.py` aside**

```bash
git mv backend/app/services/rag.py backend/app/services/_rag_legacy.py
mkdir -p backend/app/services/rag
```

- [ ] **Step 2: Create `backend/app/services/rag/__init__.py`**

```python
"""rag package — public API surface. Phases 2-3 progressively replace the legacy
implementation. Until Phase 3 lands, agentic_rag_stream still comes from the
legacy module."""
from .._rag_legacy import agentic_rag_stream  # noqa: F401
```

- [ ] **Step 3: Verify imports still work**

Run: `cd backend && python -c "from app.services.rag import agentic_rag_stream; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Run test suite — should still pass**

Run: `cd backend && pytest tests/ -v`
Expected: all tests pass (legacy file is unchanged behaviorally).

- [ ] **Step 5: Write reranker test first**

Create `backend/tests/test_retrieval.py`:

```python
import pytest
from unittest.mock import MagicMock, patch


def test_reranker_singleton_returns_same_instance():
    from app.services.rag.reranker import get_reranker
    a = get_reranker()
    b = get_reranker()
    assert a is b


def test_rerank_returns_top_n_sorted(mocker):
    from app.services.rag.reranker import rerank

    fake_ranker = MagicMock()
    fake_ranker.rerank.return_value = [
        {"id": "B", "score": 0.9},
        {"id": "A", "score": 0.5},
        {"id": "C", "score": 0.1},
    ]
    mocker.patch("app.services.rag.reranker.get_reranker", return_value=fake_ranker)

    chunk_a = MagicMock(id="A", content="cat")
    chunk_b = MagicMock(id="B", content="dog")
    chunk_c = MagicMock(id="C", content="fish")

    result = rerank("pets", [chunk_a, chunk_b, chunk_c], top_n=2)

    assert [c.id for c, s in result] == ["B", "A"]
    assert result[0][1] == 0.9


def test_rerank_empty_chunks_returns_empty():
    from app.services.rag.reranker import rerank
    assert rerank("q", [], top_n=5) == []
```

- [ ] **Step 6: Run test (should fail — module not exist)**

Run: `cd backend && pytest tests/test_retrieval.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 7: Write `backend/app/services/rag/reranker.py`**

```python
"""FlashRank cross-encoder reranker. Singleton, lazy-loaded — the model is
downloaded the first time get_reranker() is called.
"""
from __future__ import annotations
import logging
from typing import Any
from threading import Lock

from ...config import settings

logger = logging.getLogger(__name__)

_RERANKER: Any | None = None
_LOCK = Lock()


def get_reranker():
    """Return the process-wide FlashRank Ranker instance, building it on first call."""
    global _RERANKER
    if _RERANKER is None:
        with _LOCK:
            if _RERANKER is None:
                from flashrank import Ranker
                logger.info(
                    "loading FlashRank model=%s cache_dir=%s",
                    settings.reranker_model, settings.flashrank_cache_dir,
                )
                _RERANKER = Ranker(
                    model_name=settings.reranker_model,
                    cache_dir=settings.flashrank_cache_dir,
                )
    return _RERANKER


def rerank(query: str, chunks: list, top_n: int) -> list[tuple[Any, float]]:
    """Run the cross-encoder. Returns list of (chunk, score) sorted desc, len <= top_n.

    `chunks` is a list with .id and .content attributes (DocumentChunk works directly)."""
    if not chunks:
        return []
    from flashrank import RerankRequest

    passages = [{"id": str(c.id), "text": c.content} for c in chunks]
    request = RerankRequest(query=query, passages=passages)
    results = get_reranker().rerank(request)

    by_id = {str(c.id): c for c in chunks}
    return [(by_id[r["id"]], float(r["score"])) for r in results[:top_n]]
```

- [ ] **Step 8: Run tests (should pass)**

Run: `cd backend && pytest tests/test_retrieval.py -v`
Expected: all three tests pass.

- [ ] **Step 9: Smoke test real model load (slow first time)**

Run: `cd backend && python -c "from app.services.rag.reranker import get_reranker, rerank; from types import SimpleNamespace; c = [SimpleNamespace(id='1', content='cats meow'), SimpleNamespace(id='2', content='dogs bark')]; print(rerank('what sound does a cat make', c, 2))"`

Expected: model downloads (~120MB on first run), then prints the reranked tuples with chunk 1 ranked above chunk 2.

- [ ] **Step 10: Commit**

```bash
git add backend/app/services/rag/__init__.py backend/app/services/rag/reranker.py backend/app/services/_rag_legacy.py backend/tests/test_retrieval.py
git commit -m "feat(rag): FlashRank reranker singleton + rag/ package skeleton"
```

---

### Task 2.3: retrieval.py — hybrid search + RRF + fetch parents

**Files:**
- Create: `backend/app/services/rag/retrieval.py`
- Modify: `backend/tests/test_retrieval.py`

- [ ] **Step 1: Write tests**

Append to `backend/tests/test_retrieval.py`:

```python
import uuid


def test_rrf_fuse_combines_two_ranked_lists():
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    vec = [C("A"), C("B"), C("C")]  # ranks 0,1,2
    fts = [C("B"), C("A"), C("D")]  # ranks 0,1,2

    result = rrf_fuse(vec, fts, k=60)
    ids = [cid for cid, _ in result]

    # A and B both have RRF score 1/60 + 1/61; they should be above C and D.
    assert ids.index("A") < ids.index("C")
    assert ids.index("B") < ids.index("D")


def test_rrf_fuse_empty_legs():
    from app.services.rag.retrieval import rrf_fuse
    assert rrf_fuse([], [], k=60) == []


def test_fetch_parents_dedups_and_preserves_first_appearance(db):
    from app.services.rag.retrieval import fetch_parents
    from app.models import Document, DocumentParentChunk, DocumentChunk

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="ready")
    db.add(doc)
    p1 = DocumentParentChunk(document_id=doc.id, parent_index=0, content="P1")
    p2 = DocumentParentChunk(document_id=doc.id, parent_index=1, content="P2")
    db.add_all([p1, p2])
    db.flush()

    c1 = DocumentChunk(document_id=doc.id, parent_id=p1.id, chunk_index=0, content="C1", embedding=[0.0]*2560)
    c2 = DocumentChunk(document_id=doc.id, parent_id=p2.id, chunk_index=1, content="C2", embedding=[0.0]*2560)
    c3 = DocumentChunk(document_id=doc.id, parent_id=p1.id, chunk_index=2, content="C3", embedding=[0.0]*2560)
    db.add_all([c1, c2, c3])
    db.flush()

    # Children ordered [c1 (p1), c2 (p2), c3 (p1)] => expect parents [p1, p2]
    parents = fetch_parents(db, [c1, c2, c3])
    assert [p.id for p in parents] == [p1.id, p2.id]
```

- [ ] **Step 2: Run tests (should fail — module not exist)**

Run: `cd backend && pytest tests/test_retrieval.py::test_rrf_fuse_combines_two_ranked_lists -v`
Expected: FAIL.

- [ ] **Step 3: Write `backend/app/services/rag/retrieval.py`**

```python
"""Retrieval pipeline: hybrid_search -> rrf_fuse -> rerank -> fetch_parents.

Stages are plain functions, no LLM calls. Consumed by nodes.retrieve_and_rerank.
"""
from __future__ import annotations
import logging
from collections import defaultdict
from typing import Any
from uuid import UUID

from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from ...config import settings
from ...models import DocumentChunk, DocumentParentChunk
from ..ingestion import embed_text
from .reranker import rerank

logger = logging.getLogger(__name__)


def hybrid_search(
    db: Session,
    document_id: str,
    query: str,
) -> tuple[list[DocumentChunk], list[Any]]:
    """Return (vector_hits, fts_rows). Each ordered by relevance, length <= top_k."""
    vector = embed_text(query)

    vec_hits = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.embedding.cosine_distance(vector))
        .limit(settings.vector_top_k)
        .all()
    )

    fts_sql = sa_text(
        """
        SELECT id,
               ts_rank(to_tsvector('english', content),
                       plainto_tsquery('english', :q)) AS rank
        FROM   document_chunks
        WHERE  document_id = :doc_id
          AND  to_tsvector('english', content) @@ plainto_tsquery('english', :q)
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    fts_rows = db.execute(
        fts_sql, {"doc_id": document_id, "q": query, "k": settings.fts_top_k}
    ).fetchall()

    logger.info(
        "hybrid_search: query=%.80s vec=%d fts=%d",
        query, len(vec_hits), len(fts_rows),
    )
    return vec_hits, fts_rows


def rrf_fuse(vec_hits, fts_rows, k: int) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion. Returns (id, score) sorted descending by score."""
    scores: dict[Any, float] = defaultdict(float)
    for rank, chunk in enumerate(vec_hits):
        scores[chunk.id] += 1.0 / (k + rank)
    for rank, row in enumerate(fts_rows):
        scores[row.id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


def fetch_chunks_by_ids(db: Session, ids: list[UUID]) -> list[DocumentChunk]:
    if not ids:
        return []
    rows = db.query(DocumentChunk).filter(DocumentChunk.id.in_(ids)).all()
    by_id = {r.id: r for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def fetch_parents(
    db: Session, child_chunks: list[DocumentChunk]
) -> list[DocumentParentChunk]:
    """Dedup parent_ids preserving first-appearance order."""
    parent_ids: list[UUID] = []
    seen: set = set()
    for c in child_chunks:
        if c.parent_id and c.parent_id not in seen:
            seen.add(c.parent_id)
            parent_ids.append(c.parent_id)
    if not parent_ids:
        return []
    rows = (
        db.query(DocumentParentChunk)
        .filter(DocumentParentChunk.id.in_(parent_ids))
        .all()
    )
    by_id = {p.id: p for p in rows}
    return [by_id[pid] for pid in parent_ids if pid in by_id]


def retrieve(
    db: Session, document_id: str, query: str
) -> tuple[list[DocumentChunk], list[DocumentParentChunk], list[float]]:
    """End-to-end: hybrid -> RRF -> rerank -> fetch parents.
    Returns (reranked_children, parents, rerank_scores)."""
    vec_hits, fts_rows = hybrid_search(db, document_id, query)

    fused = rrf_fuse(vec_hits, fts_rows, settings.rrf_k)
    fused_ids = [cid for cid, _ in fused[:max(settings.vector_top_k, settings.fts_top_k)]]
    candidates = fetch_chunks_by_ids(db, fused_ids)

    reranked = rerank(query, candidates, settings.rerank_top_n)
    children = [c for c, _ in reranked]
    scores = [s for _, s in reranked]

    parents = fetch_parents(db, children)
    logger.info(
        "retrieve: candidates=%d reranked=%d parents=%d top_score=%.3f",
        len(candidates), len(children), len(parents),
        scores[0] if scores else 0.0,
    )
    return children, parents, scores
```

- [ ] **Step 4: Run tests**

Run: `cd backend && pytest tests/test_retrieval.py -v`
Expected: all pass (the new tests + the existing reranker tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/rag/retrieval.py backend/tests/test_retrieval.py
git commit -m "feat(rag): retrieval.py — hybrid search + RRF + fetch parents"
```

---

### Task 2.4: Swap retrieval into legacy `_rag_legacy.py`

We keep the bind_tools loop for now but route it through `retrieval.retrieve()` and feed parents to the LLM. This isolates the retrieval change so it can be measured independently.

**Files:**
- Modify: `backend/app/services/_rag_legacy.py`
- Modify: `backend/tests/test_rag.py`

- [ ] **Step 1: Rewrite `_rag_legacy.py`**

Replace the file with:

```python
import logging
from typing import AsyncGenerator
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from ..config import settings
from .rag.retrieval import retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided "
    "document. Use the search_document tool to retrieve relevant context before answering. "
    "You may search up to 3 times with different query phrasings if the first results "
    "are insufficient. Quote relevant passages verbatim when they directly answer the "
    "question. If the answer is not in the retrieved context, say so."
)


def make_search_tool(document_id: str, db: Session, collected_children: list,
                     collected_parents: list, collected_scores: list):
    @tool
    def search_document(query: str) -> str:
        """Search the document for passages relevant to the query.
        Call with different phrasings if first results are insufficient."""
        logger.info("search_document query=%.120s", query)
        children, parents, scores = retrieve(db, document_id, query)
        collected_children.extend(children)
        collected_parents.extend(parents)
        collected_scores.extend(scores)
        if not parents:
            return "NO_RELEVANT_CHUNKS"
        return "\n\n---\n\n".join(p.content for p in parents)

    return search_document


async def agentic_rag_stream(
    document_id: str, message: str, db: Session,
) -> AsyncGenerator[dict, None]:
    logger.info("agentic_rag_stream: doc=%s q=%.120s", document_id, message)
    collected_children, collected_parents, collected_scores = [], [], []
    search_tool = make_search_tool(
        document_id, db, collected_children, collected_parents, collected_scores
    )

    llm = ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url)
    llm_with_tools = llm.bind_tools([search_tool])

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(message)]

    for round_num in range(3):
        response = await llm_with_tools.ainvoke(messages)
        if not response.tool_calls:
            logger.info("agentic_rag_stream: no more tool calls after round=%d", round_num)
            break
        messages.append(response)
        for tc in response.tool_calls:
            result = search_tool.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    # Citations: emit reranked children (best signal for "what backs the answer")
    seen: set = set()
    uniq = []
    for c in collected_children:
        if c.chunk_index not in seen:
            seen.add(c.chunk_index)
            uniq.append({"chunk_index": c.chunk_index, "content": c.content[:400]})

    logger.info("agentic_rag_stream: done citations=%d", len(uniq))
    yield {"type": "citations", "chunks": uniq}
    yield {"type": "done"}
```

- [ ] **Step 2: Update `backend/tests/test_rag.py`**

The old tests patched `app.services.rag.embed_text` and `app.services.rag.ChatOllama`. Those module paths have moved. Replace `backend/tests/test_rag.py` with:

```python
import asyncio
import uuid
from unittest.mock import MagicMock


def test_make_search_tool_returns_parents(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk
    from app.services._rag_legacy import make_search_tool

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="ready")
    db.add(doc)
    parent = DocumentParentChunk(document_id=doc_id, parent_index=0,
                                 content="The revenue was $100M in Q3.")
    db.add(parent)
    db.flush()
    child = DocumentChunk(document_id=doc_id, parent_id=parent.id, chunk_index=0,
                          content="Revenue $100M Q3", embedding=[0.1] * 2560)
    db.add(child)
    db.flush()

    fake_children = [child]
    fake_parents = [parent]
    fake_scores = [0.9]
    mocker.patch(
        "app.services._rag_legacy.retrieve",
        return_value=(fake_children, fake_parents, fake_scores),
    )

    collected_c, collected_p, collected_s = [], [], []
    tool = make_search_tool(str(doc_id), db, collected_c, collected_p, collected_s)
    result = tool.invoke({"query": "Q3 revenue"})

    assert "The revenue was $100M in Q3." in result
    assert collected_p == [parent]


def test_make_search_tool_no_results_sentinel(mocker, db):
    from app.services._rag_legacy import make_search_tool

    mocker.patch(
        "app.services._rag_legacy.retrieve",
        return_value=([], [], []),
    )
    collected_c, collected_p, collected_s = [], [], []
    tool = make_search_tool("doc-id", db, collected_c, collected_p, collected_s)
    assert tool.invoke({"query": "x"}) == "NO_RELEVANT_CHUNKS"


def test_agentic_rag_stream_yields_tokens_and_citations(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="ready")
    db.add(doc)
    parent = DocumentParentChunk(document_id=doc_id, parent_index=0, content="Q3 revenue $100M")
    db.add(parent)
    db.flush()
    child = DocumentChunk(document_id=doc_id, parent_id=parent.id, chunk_index=0,
                          content="Q3 revenue $100M", embedding=[0.1] * 2560)
    db.add(child)
    db.flush()

    mocker.patch(
        "app.services._rag_legacy.retrieve",
        return_value=([child], [parent], [0.9]),
    )

    mock_ai_with_tc = MagicMock()
    mock_ai_with_tc.tool_calls = [{"id": "tc1", "args": {"query": "Q3 revenue"}}]
    mock_ai_no_tc = MagicMock()
    mock_ai_no_tc.tool_calls = []

    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = mocker.AsyncMock(side_effect=[mock_ai_with_tc, mock_ai_no_tc])

    mock_token = MagicMock()
    mock_token.content = "Revenue $100M."

    async def mock_astream(_messages):
        yield mock_token

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm.astream = mock_astream
    mocker.patch("app.services._rag_legacy.ChatOllama", return_value=mock_llm)

    from app.services._rag_legacy import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "What was Q3 revenue?", db)]

    events = asyncio.run(run())
    token_events = [e for e in events if e["type"] == "token"]
    citations = next(e for e in events if e["type"] == "citations")
    assert token_events[0]["content"] == "Revenue $100M."
    assert len(citations["chunks"]) == 1
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/test_rag.py -v`
Expected: all three pass.

- [ ] **Step 4: Run full test suite**

Run: `cd backend && pytest tests/ -v`
Expected: green.

- [ ] **Step 5: Manual smoke**

Start backend, hit the chat endpoint via the UI with a known-answer question. Verify a sensible streamed response and citations.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/_rag_legacy.py backend/tests/test_rag.py
git commit -m "feat(rag): route legacy loop through retrieval.retrieve(); feed parents to LLM"
```

---

### Task 2.5: Cache reranker model in Docker image (optional but recommended)

**Files:**
- Modify: `backend/Dockerfile`

- [ ] **Step 1: Inspect current Dockerfile**

```bash
cat backend/Dockerfile
```

- [ ] **Step 2: Add a model pre-fetch step**

Add after `RUN pip install -r requirements.txt`:

```dockerfile
# Pre-download the FlashRank reranker model so first request isn't slow
RUN python -c "from flashrank import Ranker; Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/tmp/flashrank')"
```

- [ ] **Step 3: Rebuild image**

Run: `docker compose build backend`
Expected: pulls the model into the image (~120MB).

- [ ] **Step 4: Commit**

```bash
git add backend/Dockerfile
git commit -m "build(backend): pre-cache FlashRank reranker model"
```

---

### Task 2.6: Run eval — Phase 2 checkpoint

This is **the moment of truth** for the original failure mode.

- [ ] **Step 1: Run eval**

Run: `cd backend && python -m evals.run_eval --name after_rerank`

- [ ] **Step 2: Compare against baseline AND after_chunking**

Run:
```bash
python -m evals.run_eval --compare baseline after_rerank
python -m evals.run_eval --compare after_chunking after_rerank
```

Expected: `expected_substring_match_rate` and `context_precision_mean` should jump materially. `faithfulness` and `answer_relevancy` should be flat or up. If `context_recall` drops, the rerank `top_n` may be too low — bump `rerank_top_n` to 8 and re-run.

- [ ] **Step 3: Manually verify the original failing question**

If you had a specific failing question in mind (a named-field lookup), run it through the chat UI. It should now answer correctly.

- [ ] **Step 4: Commit results**

```bash
git add backend/evals/results/after_rerank_*.json
git commit -m "evals: capture results after retrieval upgrade (RRF + reranker)"
```

---

# Phase 3 — LangGraph CRAG-lite

**Goal:** Replace the `bind_tools` loop with a LangGraph state machine. Add an LLM query rewriter (kills the regex). Wire grading, retry, faithfulness check. Delete `_rag_legacy.py` and `_preprocess_query`.

### Task 3.1: state.py + prompts.py

**Files:**
- Create: `backend/app/services/rag/state.py`
- Create: `backend/app/services/rag/prompts.py`

- [ ] **Step 1: Write `state.py`**

```python
"""LangGraph state for the agentic RAG pipeline."""
from __future__ import annotations
from typing import TypedDict, Literal


class AgentState(TypedDict, total=False):
    # Inputs
    document_id: str
    question: str

    # Routing / query
    rewritten_query: str
    intent: Literal["lookup", "summary", "reasoning", "unclear"]
    attempted_queries: list[str]
    retry_count: int

    # Retrieval results
    retrieved_children: list   # list[DocumentChunk]
    parents: list              # list[DocumentParentChunk]
    rerank_scores: list[float]

    # Grading
    graded_useful: bool

    # Generation
    answer: str
    answer_chunks: list[str]   # streamed token buffer for faithfulness check

    # Output
    citations: list[dict]
    warnings: list[dict]
    notes: list[str]


def initial_state(document_id: str, question: str) -> AgentState:
    return {
        "document_id": document_id,
        "question": question,
        "rewritten_query": "",
        "intent": "lookup",
        "attempted_queries": [],
        "retry_count": 0,
        "retrieved_children": [],
        "parents": [],
        "rerank_scores": [],
        "graded_useful": False,
        "answer": "",
        "answer_chunks": [],
        "citations": [],
        "warnings": [],
        "notes": [],
    }
```

- [ ] **Step 2: Write `prompts.py`**

```python
"""All system / instruction prompts for the agentic RAG pipeline."""

REWRITE_QUERY_SYSTEM = (
    "You are a query rewriter for a document search system. Rewrite the user's "
    "question into a concise, self-contained search query. Rules:\n"
    "1. Strip question framing ('what is', 'tell me', 'show me', 'find', 'please').\n"
    "2. For named-field lookups (e.g. 'what is the Corporate Name?'), return just "
    "the field name ('Corporate Name'). This produces better retrieval on structured docs.\n"
    "3. Preserve proper nouns, codes, dates, and exact field names verbatim — never paraphrase them.\n"
    "4. If the question is ambiguous or context-dependent (uses 'it', 'that', 'this' "
    "without a clear antecedent), set intent='unclear'.\n"
    "5. Choose intent from: 'lookup' (a specific fact), 'summary' (synthesis), "
    "'reasoning' (multi-step), 'unclear' (cannot resolve).\n"
    "Output only the JSON schema requested."
)

RETRY_QUERY_PROMPT = (
    "Previous queries returned no useful results: {attempted}. Propose ONE alternative "
    "query for the same intent. Use synonyms or different framing — do NOT repeat any "
    "previous query. Output just the query string, nothing else."
)

GRADE_CHUNKS_PROMPT = (
    "Question: {question}\n\nRetrieved passages:\n{passages}\n\n"
    "Is at least one of these passages sufficient to answer the question? "
    "Reply with exactly one word: YES or NO."
)

ANSWER_SYSTEM_GROUNDED = (
    "Answer the user's question using ONLY the document context below. "
    "When a passage directly answers the question, quote it. "
    "Do not invent details, do not draw on prior knowledge, do not speculate. "
    "If the context is insufficient, say so plainly."
)

ANSWER_SYSTEM_NOT_FOUND = (
    "The document does not appear to contain information that answers the user's question. "
    "Briefly state what the document does cover (based on the context below) and tell the "
    "user the question wasn't answered. Do not invent an answer."
)

FAITHFULNESS_PROMPT = (
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Draft answer:\n{answer}\n\n"
    "Is every factual claim in the draft answer supported by the context? "
    "Reply with exactly one word: YES or NO."
)
```

- [ ] **Step 3: Smoke import**

Run: `cd backend && python -c "from app.services.rag.state import initial_state; from app.services.rag.prompts import REWRITE_QUERY_SYSTEM; print(initial_state('d', 'q')['retry_count'])"`
Expected: prints `0`.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/rag/state.py backend/app/services/rag/prompts.py
git commit -m "feat(rag): state + prompts modules"
```

---

### Task 3.2: nodes.py — rewrite_query

**Files:**
- Create: `backend/app/services/rag/nodes.py`
- Create: `backend/tests/test_rag_nodes.py`

- [ ] **Step 1: Write test**

Create `backend/tests/test_rag_nodes.py`:

```python
from unittest.mock import MagicMock, AsyncMock


def test_rewrite_query_strips_framing_and_sets_intent(mocker):
    from app.services.rag.nodes import rewrite_query
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=MagicMock(rewritten_query="Corporate Name", intent="lookup")
    )
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    import asyncio
    state = initial_state("doc1", "What is the Corporate Name?")
    out = asyncio.run(rewrite_query(state))

    assert out["rewritten_query"] == "Corporate Name"
    assert out["intent"] == "lookup"
```

- [ ] **Step 2: Run test (should fail — module missing)**

Run: `cd backend && pytest tests/test_rag_nodes.py::test_rewrite_query_strips_framing_and_sets_intent -v`
Expected: FAIL.

- [ ] **Step 3: Write `backend/app/services/rag/nodes.py`** (first node only)

```python
"""LangGraph nodes for the agentic RAG pipeline.

Each node is `async def node(state: AgentState) -> dict` returning a partial state.
LangGraph merges the partial back into the full state.
"""
from __future__ import annotations
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from ...config import settings
from . import prompts
from .state import AgentState

logger = logging.getLogger(__name__)


def _chat_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


class QueryRewrite(BaseModel):
    rewritten_query: str = Field(..., description="The cleaned search query.")
    intent: Literal["lookup", "summary", "reasoning", "unclear"]


async def rewrite_query(state: AgentState) -> dict:
    llm = _chat_llm()
    structured = llm.with_structured_output(QueryRewrite)
    response: QueryRewrite = await structured.ainvoke([
        SystemMessage(prompts.REWRITE_QUERY_SYSTEM),
        HumanMessage(state["question"]),
    ])
    logger.info(
        "rewrite_query: q=%.80s -> rewritten=%.80s intent=%s",
        state["question"], response.rewritten_query, response.intent,
    )
    return {
        "rewritten_query": response.rewritten_query,
        "intent": response.intent,
        "notes": state.get("notes", []) + [f"rewrite: {response.intent}"],
    }
```

- [ ] **Step 4: Run test (should pass)**

Run: `cd backend && pytest tests/test_rag_nodes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/rag/nodes.py backend/tests/test_rag_nodes.py
git commit -m "feat(rag): nodes.py — rewrite_query node"
```

---

### Task 3.3: nodes.py — retrieve_and_rerank + grade_chunks

**Files:**
- Modify: `backend/app/services/rag/nodes.py`
- Modify: `backend/tests/test_rag_nodes.py`

- [ ] **Step 1: Write tests**

Append to `backend/tests/test_rag_nodes.py`:

```python
import asyncio
from unittest.mock import MagicMock


def test_retrieve_and_rerank_writes_children_parents_scores(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk
    from app.services.rag.nodes import retrieve_and_rerank
    from app.services.rag.state import initial_state
    import uuid

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="ready")
    db.add(doc)
    p = DocumentParentChunk(document_id=doc_id, parent_index=0, content="P")
    db.add(p); db.flush()
    c = DocumentChunk(document_id=doc_id, parent_id=p.id, chunk_index=0, content="C",
                      embedding=[0.0] * 2560)
    db.add(c); db.flush()

    mocker.patch(
        "app.services.rag.nodes.retrieve",
        return_value=([c], [p], [0.8]),
    )
    state = initial_state(str(doc_id), "q")
    state["rewritten_query"] = "rewritten"
    out = asyncio.run(retrieve_and_rerank(state, db))

    assert out["retrieved_children"] == [c]
    assert out["parents"] == [p]
    assert out["rerank_scores"] == [0.8]
    assert "rewritten" in out["attempted_queries"]


def test_grade_chunks_fast_path_yes(mocker):
    from app.services.rag.nodes import grade_chunks
    from app.services.rag.state import initial_state

    state = initial_state("d", "q")
    state["retrieved_children"] = [MagicMock()]
    state["rerank_scores"] = [0.5]  # above default 0.05

    out = asyncio.run(grade_chunks(state))
    assert out["graded_useful"] is True


def test_grade_chunks_fast_path_no(mocker):
    from app.services.rag.nodes import grade_chunks
    from app.services.rag.state import initial_state

    state = initial_state("d", "q")
    state["retrieved_children"] = [MagicMock()]
    state["rerank_scores"] = [0.001]  # below floor

    out = asyncio.run(grade_chunks(state))
    assert out["graded_useful"] is False
```

- [ ] **Step 2: Append to `nodes.py`**

```python
from sqlalchemy.orm import Session
from .retrieval import retrieve


async def retrieve_and_rerank(state: AgentState, db: Session) -> dict:
    query = state["rewritten_query"] or state["question"]
    children, parents, scores = retrieve(db, state["document_id"], query)
    attempted = state.get("attempted_queries", []) + [query]
    note = (
        f"retrieve: children={len(children)} parents={len(parents)} "
        f"top_score={scores[0] if scores else 0:.3f}"
    )
    logger.info(note)
    return {
        "retrieved_children": children,
        "parents": parents,
        "rerank_scores": scores,
        "attempted_queries": attempted,
        "notes": state.get("notes", []) + [note],
    }


async def grade_chunks(state: AgentState) -> dict:
    """Fast path by default. Strict LLM path if settings.strict_grader=True."""
    children = state.get("retrieved_children") or []
    scores = state.get("rerank_scores") or []

    if not children:
        return {"graded_useful": False,
                "notes": state.get("notes", []) + ["grade: no_chunks"]}

    if not settings.strict_grader:
        useful = max(scores, default=0.0) >= settings.rerank_score_floor
        return {"graded_useful": useful,
                "notes": state.get("notes", []) + [f"grade(fast): useful={useful}"]}

    # Strict path: LLM judge
    passages = "\n\n---\n\n".join(c.content for c in children[:5])
    llm = _chat_llm()
    response = await llm.ainvoke([
        SystemMessage("You judge whether retrieved passages contain the answer."),
        HumanMessage(prompts.GRADE_CHUNKS_PROMPT.format(
            question=state["question"], passages=passages,
        )),
    ])
    verdict = (response.content or "").strip().upper().startswith("YES")
    return {"graded_useful": verdict,
            "notes": state.get("notes", []) + [f"grade(strict): {verdict}"]}
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/test_rag_nodes.py -v`
Expected: all four tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/rag/nodes.py backend/tests/test_rag_nodes.py
git commit -m "feat(rag): retrieve_and_rerank + grade_chunks nodes"
```

---

### Task 3.4: nodes.py — rewrite_and_retry + generate_answer + faithfulness_check

**Files:**
- Modify: `backend/app/services/rag/nodes.py`
- Modify: `backend/tests/test_rag_nodes.py`

- [ ] **Step 1: Write tests**

Append to `backend/tests/test_rag_nodes.py`:

```python
def test_rewrite_and_retry_produces_new_query(mocker):
    from app.services.rag.nodes import rewrite_and_retry
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    msg = MagicMock(); msg.content = "alternative phrasing"
    fake_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    state = initial_state("d", "q")
    state["attempted_queries"] = ["first", "second"]
    state["retry_count"] = 0

    out = asyncio.run(rewrite_and_retry(state))
    assert out["rewritten_query"] == "alternative phrasing"
    assert out["retry_count"] == 1


def test_faithfulness_check_yes_emits_no_warning(mocker):
    from app.services.rag.nodes import faithfulness_check
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    msg = MagicMock(); msg.content = "YES"
    fake_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    p = MagicMock(); p.content = "context"
    state = initial_state("d", "q")
    state["parents"] = [p]
    state["answer"] = "answer"
    out = asyncio.run(faithfulness_check(state))
    assert out["warnings"] == []


def test_faithfulness_check_no_appends_warning(mocker):
    from app.services.rag.nodes import faithfulness_check
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    msg = MagicMock(); msg.content = "NO"
    fake_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    p = MagicMock(); p.content = "context"
    state = initial_state("d", "q")
    state["parents"] = [p]
    state["answer"] = "answer"
    out = asyncio.run(faithfulness_check(state))
    assert len(out["warnings"]) == 1
    assert "warning" in out["warnings"][0]["type"]
```

- [ ] **Step 2: Append to `nodes.py`**

```python
async def rewrite_and_retry(state: AgentState) -> dict:
    attempted = state.get("attempted_queries", [])
    llm = _chat_llm(temperature=0.3)  # a little creativity for alt phrasing
    response = await llm.ainvoke([
        HumanMessage(prompts.RETRY_QUERY_PROMPT.format(attempted=attempted)),
    ])
    new_query = (response.content or "").strip().strip('"').strip("'")
    return {
        "rewritten_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "notes": state.get("notes", []) + [f"retry({state.get('retry_count', 0)+1}): {new_query[:60]}"],
    }


async def generate_answer(state: AgentState) -> dict:
    """NOTE: streaming is handled at the graph level via astream_events.
    This node still calls the LLM and the streamed tokens are picked up
    by the graph layer. We accumulate the full answer here for the
    faithfulness_check node."""
    system_prompt = (
        prompts.ANSWER_SYSTEM_GROUNDED
        if state.get("graded_useful")
        else prompts.ANSWER_SYSTEM_NOT_FOUND
    )
    context = "\n\n---\n\n".join(p.content for p in state.get("parents", []))
    llm = _chat_llm()
    response = await llm.ainvoke([
        SystemMessage(system_prompt),
        HumanMessage(f"Document context:\n{context}\n\nQuestion: {state['question']}"),
    ])
    answer = response.content or ""
    return {"answer": answer,
            "notes": state.get("notes", []) + [f"answer: len={len(answer)}"]}


async def faithfulness_check(state: AgentState) -> dict:
    if not state.get("answer"):
        return {}
    context = "\n\n---\n\n".join(p.content for p in state.get("parents", []))
    llm = _chat_llm()
    response = await llm.ainvoke([
        HumanMessage(prompts.FAITHFULNESS_PROMPT.format(
            question=state["question"],
            context=context,
            answer=state["answer"],
        )),
    ])
    verdict = (response.content or "").strip().upper().startswith("YES")
    warnings = list(state.get("warnings", []))
    if not verdict:
        warnings.append({
            "type": "warning",
            "message": "Some claims may not be fully supported by the document.",
        })
    return {"warnings": warnings,
            "notes": state.get("notes", []) + [f"faithfulness: {verdict}"]}
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/test_rag_nodes.py -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/rag/nodes.py backend/tests/test_rag_nodes.py
git commit -m "feat(rag): rewrite_and_retry + generate_answer + faithfulness_check nodes"
```

---

### Task 3.5: graph.py — wire the state machine + streaming entry point

**Files:**
- Create: `backend/app/services/rag/graph.py`
- Create: `backend/tests/test_rag_graph.py`
- Modify: `backend/app/services/rag/__init__.py`

- [ ] **Step 1: Write test for the routing function**

Create `backend/tests/test_rag_graph.py`:

```python
from app.services.rag.state import initial_state


def test_route_after_grade_answer_when_useful():
    from app.services.rag.graph import route_after_grade
    state = initial_state("d", "q")
    state["graded_useful"] = True
    assert route_after_grade(state) == "answer"


def test_route_after_grade_retry_when_not_useful_under_budget():
    from app.services.rag.graph import route_after_grade
    state = initial_state("d", "q")
    state["graded_useful"] = False
    state["retry_count"] = 0
    assert route_after_grade(state) == "retry"


def test_route_after_grade_give_up_when_budget_exhausted():
    from app.services.rag.graph import route_after_grade
    state = initial_state("d", "q")
    state["graded_useful"] = False
    state["retry_count"] = 2
    assert route_after_grade(state) == "give_up"
```

- [ ] **Step 2: Run test (should fail)**

Run: `cd backend && pytest tests/test_rag_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.rag.graph'`.

- [ ] **Step 3: Write `backend/app/services/rag/graph.py`**

```python
"""LangGraph state machine for agentic RAG.

Wiring:
    START -> rewrite_query -> retrieve -> grade
        grade --useful--> answer -> check -> END
        grade --retry-->  retry  -> retrieve  (back-edge, loops)
        grade --give_up--> answer -> check -> END (with NOT_FOUND framing)

Streaming: agentic_rag_stream() runs the graph with astream_events and yields
SSE-shaped dicts compatible with the existing chat router.
"""
from __future__ import annotations
import logging
from functools import partial
from typing import AsyncGenerator, Literal
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, END

from ...config import settings
from .state import AgentState, initial_state
from . import nodes

logger = logging.getLogger(__name__)


def route_after_grade(state: AgentState) -> Literal["answer", "retry", "give_up"]:
    if state.get("graded_useful"):
        return "answer"
    if state.get("retry_count", 0) < settings.max_retrieval_retries:
        return "retry"
    return "give_up"


def build_graph(db: Session):
    """Compile a StateGraph. db is closed-over via partial because the retrieve
    node needs a Session."""
    g = StateGraph(AgentState)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("retrieve", partial(nodes.retrieve_and_rerank, db=db))
    g.add_node("grade", nodes.grade_chunks)
    g.add_node("retry", nodes.rewrite_and_retry)
    g.add_node("answer", nodes.generate_answer)
    g.add_node("check", nodes.faithfulness_check)

    g.set_entry_point("rewrite_query")
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "answer": "answer",
        "retry": "retry",
        "give_up": "answer",
    })
    g.add_edge("retry", "retrieve")
    g.add_edge("answer", "check")
    g.add_edge("check", END)

    return g.compile()


def _build_citations(state: AgentState) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for c in state.get("retrieved_children", []):
        if c.chunk_index in seen:
            continue
        seen.add(c.chunk_index)
        out.append({"chunk_index": c.chunk_index, "content": (c.content or "")[:400]})
    return out


async def agentic_rag_stream(
    document_id: str, message: str, db: Session,
) -> AsyncGenerator[dict, None]:
    """Run the graph and yield SSE events: token / citations / warning / done."""
    logger.info("graph stream: doc=%s q=%.120s", document_id, message)
    graph = build_graph(db)
    state_in = initial_state(document_id, message)

    final_state: AgentState = state_in
    answer_chunks: list[str] = []

    async for event in graph.astream_events(state_in, version="v2"):
        kind = event.get("event")
        # Token streaming from the 'answer' node only
        if kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node")
            if node == "answer":
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", "") if chunk else ""
                if content:
                    answer_chunks.append(content)
                    yield {"type": "token", "content": content}
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            final_state = event.get("data", {}).get("output", final_state)

    # Update answer in case astream_events didn't surface it via tokens (e.g. mocked)
    if not final_state.get("answer"):
        final_state["answer"] = "".join(answer_chunks)

    for w in final_state.get("warnings", []):
        yield w

    yield {"type": "citations", "chunks": _build_citations(final_state)}
    yield {"type": "done"}
```

- [ ] **Step 4: Run routing tests**

Run: `cd backend && pytest tests/test_rag_graph.py -v`
Expected: PASS.

- [ ] **Step 5: Wire graph as the public entry point**

Replace `backend/app/services/rag/__init__.py` with:

```python
"""rag package — public API."""
from .graph import agentic_rag_stream  # noqa: F401
```

- [ ] **Step 6: Run all tests**

Run: `cd backend && pytest tests/ -v`
Expected: most tests pass. Tests in `test_rag.py` (the ones that import `_rag_legacy`) still pass — that file still exists. We delete it in the next task.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/rag/graph.py backend/app/services/rag/__init__.py backend/tests/test_rag_graph.py
git commit -m "feat(rag): LangGraph state machine + streaming entry point"
```

---

### Task 3.6: Delete legacy code

**Files:**
- Delete: `backend/app/services/_rag_legacy.py`
- Modify: `backend/tests/test_rag.py`

- [ ] **Step 1: Replace `backend/tests/test_rag.py` with an integration test**

Replace the file with:

```python
"""Integration test for the public agentic_rag_stream entry point (via graph)."""
import asyncio
import uuid
from unittest.mock import MagicMock


def test_agentic_rag_stream_emits_token_citation_done_in_order(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="ready")
    db.add(doc)
    parent = DocumentParentChunk(document_id=doc_id, parent_index=0, content="Q3 rev $100M")
    db.add(parent); db.flush()
    child = DocumentChunk(document_id=doc_id, parent_id=parent.id, chunk_index=0,
                          content="Q3 rev $100M", embedding=[0.1] * 2560)
    db.add(child); db.flush()

    # Mock retrieval
    mocker.patch(
        "app.services.rag.nodes.retrieve",
        return_value=([child], [parent], [0.8]),
    )

    # Mock all LLM calls deterministically
    def fake_chat_llm(temperature=0.0):
        llm = MagicMock()
        # structured output for rewrite_query
        structured = MagicMock()
        structured.ainvoke = mocker.AsyncMock(
            return_value=MagicMock(rewritten_query="Q3 revenue", intent="lookup")
        )
        llm.with_structured_output.return_value = structured
        # plain ainvoke for grade (strict path off — won't be called)
        msg = MagicMock(); msg.content = "Q3 revenue was $100M."
        llm.ainvoke = mocker.AsyncMock(return_value=msg)
        return llm
    mocker.patch("app.services.rag.nodes._chat_llm", side_effect=fake_chat_llm)

    from app.services.rag import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "What was Q3 revenue?", db)]

    events = asyncio.run(run())
    # We expect at least: a citations event, a done event. Token events may be empty
    # if the mocked LLM doesn't stream — that's fine for this wiring test.
    types = [e["type"] for e in events]
    assert "citations" in types
    assert "done" in types
    assert types[-1] == "done"
    citation = next(e for e in events if e["type"] == "citations")
    assert citation["chunks"][0]["chunk_index"] == 0
```

- [ ] **Step 2: Delete legacy file**

```bash
git rm backend/app/services/_rag_legacy.py
```

- [ ] **Step 3: Run tests**

Run: `cd backend && pytest tests/ -v`
Expected: all tests pass. If `test_rag.py` fails because the wiring test depends on real LLM streaming, accept that the token assertion is loose — the integration is exercised by the manual smoke test below.

- [ ] **Step 4: Manual smoke test**

Start backend. Use the chat UI on a real document. Verify:
- Tokens stream
- Citations appear
- A previously-failing named-field question returns a correct answer (or, if the doc doesn't contain it, an honest "not in document" reply)

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_rag.py
git rm backend/app/services/_rag_legacy.py 2>/dev/null || true
git commit -m "refactor(rag): delete legacy bind_tools loop; LangGraph is the entry point"
```

---

### Task 3.7: Run eval — Phase 3 checkpoint

- [ ] **Step 1: Run eval**

Run: `cd backend && python -m evals.run_eval --name after_langgraph`

- [ ] **Step 2: Compare to after_rerank**

Run: `python -m evals.run_eval --compare after_rerank after_langgraph`

Expected:
- `answered_rate` on the `not_in_doc` category should improve (the give_up path correctly says "not found").
- `faithfulness_mean` should improve (faithfulness_check + better prompts).
- Overall metrics should not regress.

If `expected_substring_match_rate` drops, the rewrite_query node is over-aggressively stripping content. Inspect the new queries via `notes` logs and tune the prompt.

- [ ] **Step 3: Commit**

```bash
git add backend/evals/results/after_langgraph_*.json
git commit -m "evals: capture results after LangGraph CRAG-lite migration"
```

---

# Phase 4 — Polish

**Goal:** Make the system observable in production and update docs.

### Task 4.1: Surface notes in the streamed `done` event (debug mode)

**Files:**
- Modify: `backend/app/services/rag/graph.py`

- [ ] **Step 1: Add debug payload**

Update the `agentic_rag_stream` function in `graph.py` so the `done` event optionally carries debug info. Replace the last block (`yield {"type": "citations", ...}; yield {"type": "done"}`) with:

```python
    yield {"type": "citations", "chunks": _build_citations(final_state)}

    done_payload: dict = {"type": "done"}
    if settings.log_level.upper() == "DEBUG":
        done_payload["debug"] = {
            "attempted_queries": final_state.get("attempted_queries", []),
            "retry_count": final_state.get("retry_count", 0),
            "intent": final_state.get("intent"),
            "notes": final_state.get("notes", []),
        }
    yield done_payload
```

- [ ] **Step 2: Commit**

```bash
git add backend/app/services/rag/graph.py
git commit -m "feat(rag): include debug info in done event when LOG_LEVEL=DEBUG"
```

---

### Task 4.2: Replace rag_enhancement.md with final summary

**Files:**
- Modify: `features/chat_with_doc/rag_enhancement.md`

- [ ] **Step 1: Replace contents**

Replace `features/chat_with_doc/rag_enhancement.md` with:

```markdown
# Agentic RAG — enhanced architecture (post-2026-05-15)

The chat-with-doc pipeline is implemented as a LangGraph state machine in
`backend/app/services/rag/`. See the design spec at
`docs/superpowers/specs/2026-05-15-agentic-rag-enhancement-design.md` and the
implementation plan at `docs/superpowers/plans/2026-05-15-agentic-rag-enhancement.md`.

## Flow

```
rewrite_query  ->  retrieve_and_rerank  ->  grade_chunks
                                              |
                       +----------------------+----------------------+
                       v useful               v retry (<=2)         v give_up
                  generate_answer       rewrite_and_retry       generate_answer
                       |                       |                   (NOT_FOUND)
                       |                       ^                       |
                       |                       +- back to retrieve     |
                       v                                                v
                  faithfulness_check  --------------------------------- END
```

## Module layout

| File | Responsibility |
|---|---|
| `rag/graph.py` | LangGraph wiring + `agentic_rag_stream()` entry point |
| `rag/nodes.py` | Six nodes (rewrite_query, retrieve_and_rerank, grade_chunks, rewrite_and_retry, generate_answer, faithfulness_check) |
| `rag/state.py` | `AgentState` TypedDict |
| `rag/prompts.py` | All prompts |
| `rag/retrieval.py` | hybrid_search -> rrf_fuse -> rerank -> fetch_parents |
| `rag/reranker.py` | FlashRank singleton |

## Data model

`document_parent_chunks` stores large (~1500 token) parents fed to the LLM;
`document_chunks` are 300-token children that get embedded + FTS-indexed and
carry `parent_id`. See migration `0004_parent_child_chunks.py`.

## Evaluation

Golden Q/A set at `backend/evals/golden_set.yaml`. Run with:

`cd backend && python -m evals.run_eval --name <run_name>`
`python -m evals.run_eval --compare <run_a> <run_b>`

Baseline lives at `backend/evals/results/baseline_*.json`. Run an eval before
shipping any retrieval/agent change.

## Tunables

All in `backend/app/config.py`: `vector_top_k`, `fts_top_k`, `rrf_k`,
`rerank_top_n`, `rerank_score_floor`, `max_retrieval_retries`, `strict_grader`.
```

- [ ] **Step 2: Commit**

```bash
git add features_planning/chat_with_doc/rag_enhancement.md
git commit -m "docs(rag): update feature note with final architecture summary"
```

---

### Task 4.3: Update CLAUDE.md to point at the new module layout

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Find the architecture section in CLAUDE.md**

Open `CLAUDE.md` and locate the section that describes the backend layout (currently the file focuses on frontend scaffold; the backend was added after).

- [ ] **Step 2: Add a backend RAG section**

Append:

```markdown
## Backend — RAG pipeline

The agentic RAG service lives in `backend/app/services/rag/` and is a LangGraph
state machine. Entry point: `agentic_rag_stream(document_id, message, db)`.
Six nodes (rewrite_query, retrieve_and_rerank, grade_chunks, rewrite_and_retry,
generate_answer, faithfulness_check). Retrieval is hybrid (pgvector + Postgres
FTS) fused with RRF, reranked with FlashRank, then we return parent chunks
(~1500 tokens) to the LLM while children (~300 tokens) are what gets retrieved.

See `features/chat_with_doc/rag_enhancement.md` for the flow diagram and
`docs/superpowers/specs/2026-05-15-agentic-rag-enhancement-design.md` for the
design rationale.

### Evaluation

Before changing retrieval or agent code, run the eval harness:

`cd backend`
`python -m evals.run_eval --name <name>`
`python -m evals.run_eval --compare <baseline_name> <name>`

Golden set: `backend/evals/golden_set.yaml`.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: point CLAUDE.md at new RAG module + eval workflow"
```

---

# Final verification

- [ ] **Step 1: Full test suite**

Run: `cd backend && pytest tests/ -v`
Expected: all green.

- [ ] **Step 2: Final eval**

Run: `cd backend && python -m evals.run_eval --name final`
Compare against baseline:
```bash
python -m evals.run_eval --compare baseline final
```
Expected: `expected_substring_match_rate` materially higher than baseline; `faithfulness_mean` >= 0.85; `context_recall_mean` >= 0.80.

- [ ] **Step 3: End-to-end manual smoke**

- Upload a fresh document via the UI
- Wait for ingestion to complete (status="ready")
- Ask a named-field question — verify correct answer with citation
- Ask a fact lookup — verify correct answer
- Ask a question NOT in the document — verify the agent says so (no hallucination)
- Inspect `backend/logs/` (or stdout) — look for `notes:` breadcrumbs showing the full graph path

- [ ] **Step 4: Tag**

```bash
git tag -a v2-agentic-rag -m "Agentic RAG enhancement (LangGraph + reranker + parent-child + eval)"
```

Done.
