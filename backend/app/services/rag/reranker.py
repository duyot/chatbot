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
