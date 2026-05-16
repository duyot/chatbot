"""Ollama-hosted cross-encoder reranker (qllama/bge-reranker-v2-m3 by default).

The qllama BGE reranker is exposed via Ollama's /api/embed endpoint: send the
concatenated "query\\n\\npassage" as the input and the model returns a 1-element
"embedding" whose value is the relevance logit (higher = more relevant).

Batch: we send all (query, passage) pairs in a single /api/embed call.

There is no singleton state to hold; get_reranker() is kept for API
compatibility with the previous FlashRank version and just returns the model name.
"""
from __future__ import annotations
import logging
from typing import Any

import httpx

from ...config import settings

logger = logging.getLogger(__name__)


def get_reranker() -> str:
    """Return the configured reranker model name. Kept for API compatibility."""
    return settings.reranker_model


def _format_pair(query: str, passage: str) -> str:
    """How qllama/bge-reranker expects (query, passage) input — join with a
    blank line so the model's tokenizer treats them as separate segments."""
    return f"{query}\n\n{passage}"


def rerank(query: str, chunks: list, top_n: int) -> list[tuple[Any, float]]:
    """Score each chunk via Ollama, return top_n (chunk, score) sorted desc.

    `chunks` is a list with .id and .content attributes (DocumentChunk works directly).
    """
    if not chunks:
        return []

    inputs = [_format_pair(query, c.content or "") for c in chunks]

    url = f"{settings.ollama_base_url.rstrip('/')}/api/embed"
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            url,
            json={"model": settings.reranker_model, "input": inputs},
        )
        response.raise_for_status()
        body = response.json()

    embeddings = body.get("embeddings") or []
    if len(embeddings) != len(chunks):
        logger.warning(
            "rerank: model returned %d embeddings for %d chunks; aligning by index",
            len(embeddings), len(chunks),
        )

    scored: list[tuple[Any, float]] = []
    for chunk, emb in zip(chunks, embeddings):
        # bge-reranker via Ollama returns a 1-element vector with the logit.
        # Be defensive in case a larger vector is returned: take element 0.
        score = float(emb[0]) if emb else 0.0
        scored.append((chunk, score))

    scored.sort(key=lambda x: -x[1])
    logger.info(
        "rerank: model=%s n_chunks=%d top_score=%.3f bottom_score=%.3f",
        settings.reranker_model, len(scored),
        scored[0][1] if scored else 0.0,
        scored[-1][1] if scored else 0.0,
    )
    return scored[:top_n]
