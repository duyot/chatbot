"""Cross-encoder reranker backed by OpenRouter's /v1/rerank endpoint.

We POST query + candidate documents to OpenRouter, which forwards to the
configured reranker model (e.g. nvidia/llama-nemotron-rerank-vl-1b-v2:free).
The response is a sorted list of {index, relevance_score, document}.

History: an earlier draft tried to call ChatOpenAI.with_structured_output
to score passages via /v1/chat/completions, which 404s for dedicated rerank
models (they expose /rerank only, not chat). Before that, this module
called HuggingFace TEI's /rerank in-cluster; the request shape is similar.
Before *that*, it misused Ollama's /api/embed and read embedding[0] as a
score — effectively random.

`rerank_score_floor` (in settings) can be set above the typical noise floor
of the chosen model to drop weak hits — Nemotron rerank scores are roughly
in [0, 1]; default of -1e9 lets everything through.
"""
from __future__ import annotations
import logging
from typing import Any

import httpx

from ...config import settings

logger = logging.getLogger(__name__)

# Returned in place of a real relevance score when rerank() degrades to input
# order. Real relevance scores from the configured models are always >= 0, so
# this sentinel is distinguishable from a genuine (if low) score — a degraded
# rerank must be detectable, not silently indistinguishable from "the reranker
# ran and found everything irrelevant."
RERANK_FAILED_SCORE = -1.0


def get_reranker() -> str:
    """Return the configured reranker model name. Kept for API compatibility."""
    return settings.reranker_model


def _rerank_text(chunk: Any) -> str:
    """Text sent to the cross-encoder for one candidate.

    Includes the generated context when present so the reranker scores on the
    same signal the retrieval arms used. Tolerates objects without a .context
    attribute — rerank() is called with plain chunk-likes in places.

    Format counterpart: app/services/ingestion.py:build_embedding_input must
    produce the identical string, or retrieval and reranking see different
    text.
    """
    content = (getattr(chunk, "content", "") or "").strip()
    context = (getattr(chunk, "context", None) or "").strip()
    if not context:
        return content
    return f"{context}\n\n{content}"


def rerank(query: str, chunks: list, top_n: int) -> list[tuple[Any, float]]:
    """Score each chunk via OpenRouter /v1/rerank, return top_n (chunk, score)
    sorted descending.

    `chunks` is a list with .id and .content attributes (DocumentChunk works
    directly). On any error returns [(c, RERANK_FAILED_SCORE) for c in
    chunks[:top_n]] so the retrieval pipeline degrades gracefully to RRF
    order, while still leaving a signal that the rerank did not actually run.
    """
    if not chunks:
        return []

    url = f"{settings.openrouter_base_url.rstrip('/')}/rerank"
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    documents = [{"text": _rerank_text(c)} for c in chunks]
    payload = {
        "model": settings.reranker_model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(chunks)),
    }

    logger.info(
        "rerank: request query=%.80s n_docs=%d model=%s",
        query, len(chunks), settings.reranker_model,
    )

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            body = response.json()
    except Exception as exc:
        logger.error(
            "rerank: API call failed (%s); falling back to input order", exc,
        )
        return [(c, RERANK_FAILED_SCORE) for c in chunks[:top_n]]

    results = body.get("results") if isinstance(body, dict) else body
    if not isinstance(results, list):
        logger.error("rerank: unexpected response shape: %r", body)
        return [(c, RERANK_FAILED_SCORE) for c in chunks[:top_n]]

    scored: list[tuple[Any, float]] = []
    for item in results:
        idx = item.get("index")
        if idx is None or idx < 0 or idx >= len(chunks):
            logger.warning("rerank: skipping result with invalid index=%r", idx)
            continue
        score = item.get("relevance_score", item.get("score", 0.0))
        scored.append((chunks[idx], float(score)))

    # API returns sorted; defensive sort in case a future model variant doesn't.
    scored.sort(key=lambda x: -x[1])
    logger.info(
        "rerank: model=%s n_chunks=%d top_score=%.3f bottom_score=%.3f",
        settings.reranker_model, len(scored),
        scored[0][1] if scored else 0.0,
        scored[-1][1] if scored else 0.0,
    )
    return scored[:top_n]
