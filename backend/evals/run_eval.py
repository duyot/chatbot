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
        .filter(Document.file_name == file_name, Document.status == "done")
        .first()
    )
    if not doc:
        raise RuntimeError(
            f"golden_set.yaml references {file_name!r} but no done document with that "
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
