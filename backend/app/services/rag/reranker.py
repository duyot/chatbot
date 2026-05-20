"""Cross-encoder reranker backed by HuggingFace Text Embeddings Inference (TEI).

TEI exposes a first-class /rerank endpoint for cross-encoder models such as
BAAI/bge-reranker-v2-m3. Given a query and a list of texts it returns a list of
{index, score} records sorted by score descending; scores are raw logits
(roughly -10..+10 for bge-reranker-v2-m3), higher = more relevant.

History: we previously called Ollama's /api/embed for qllama/bge-reranker-v2-m3,
but that endpoint returns the model's hidden-state embedding rather than the
classifier-head logit, so the "score" was just embedding[0] — effectively
random. TEI exposes the actual rerank head.

get_reranker() is kept for API compatibility and just returns the model name.
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


def rerank(query: str, chunks: list, top_n: int) -> list[tuple[Any, float]]:
    """Score each chunk via TEI's /rerank, return top_n (chunk, score) sorted desc.

    `chunks` is a list with .id and .content attributes (DocumentChunk works directly).
    """
    if not chunks:
        return []

    texts = [c.content or "" for c in chunks]
    url = f"{settings.reranker_base_url.rstrip('/')}/rerank"
    payload = {
        "query": query,
        "texts": texts,
        "raw_scores": True,
        "return_text": False,
        "truncate": True,
    }

    logger.info(
        "rerank: request query=%.80s n_texts=%d url=%s",
        query, len(texts), url,
    )

    with httpx.Client(timeout=60.0) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()

    # TEI usually returns a bare list of {index, score}; some versions wrap it.
    results = body if isinstance(body, list) else body.get("results") or []

    if len(results) != len(chunks):
        logger.warning(
            "rerank: TEI returned %d results for %d chunks; aligning by index",
            len(results), len(chunks),
        )

    scored: list[tuple[Any, float]] = []
    for item in results:
        idx = item.get("index")
        if idx is None or idx < 0 or idx >= len(chunks):
            logger.warning("rerank: skipping result with out-of-range index=%r", idx)
            continue
        scored.append((chunks[idx], float(item.get("score", 0.0))))

    scored.sort(key=lambda x: -x[1])
    logger.info(
        "rerank: model=%s n_chunks=%d top_score=%.3f bottom_score=%.3f",
        settings.reranker_model, len(scored),
        scored[0][1] if scored else 0.0,
        scored[-1][1] if scored else 0.0,
    )
    return scored[:top_n]
