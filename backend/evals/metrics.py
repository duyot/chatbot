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
