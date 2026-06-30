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
    page_range: tuple[int, int] | None = None,
) -> tuple[list[DocumentChunk], list[Any]]:
    """Return (vector_hits, fts_rows). Each ordered by relevance, length <= top_k.

    If page_range=(lo, hi) is given, only chunks whose page is in [lo, hi]
    (inclusive) are considered — a metadata filter for page-scoped queries.
    """
    vector = embed_text(query)

    vec_q = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id)
    if page_range is not None:
        vec_q = vec_q.filter(DocumentChunk.page.between(page_range[0], page_range[1]))
    vec_hits = (
        vec_q.order_by(DocumentChunk.embedding.cosine_distance(vector))
        .limit(settings.vector_top_k)
        .all()
    )

    page_clause = "AND page BETWEEN :lo AND :hi" if page_range is not None else ""
    fts_sql = sa_text(
        f"""
        SELECT id,
               ts_rank(to_tsvector('english', content),
                       plainto_tsquery('english', :q)) AS rank
        FROM   document_chunks
        WHERE  document_id = :doc_id
          AND  to_tsvector('english', content) @@ plainto_tsquery('english', :q)
          {page_clause}
        ORDER BY rank DESC
        LIMIT :k
        """
    )
    params: dict[str, Any] = {"doc_id": document_id, "q": query, "k": settings.fts_top_k}
    if page_range is not None:
        params["lo"], params["hi"] = page_range[0], page_range[1]
    fts_rows = db.execute(fts_sql, params).fetchall()

    logger.info(
        "hybrid_search: query=%.80s vec=%d fts=%d page_range=%s",
        query, len(vec_hits), len(fts_rows), page_range,
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


def apply_metadata_boost(
    reranked: list[tuple[DocumentChunk, float]]
) -> list[tuple[DocumentChunk, float]]:
    """Re-score reranked candidates using chunk metadata, then re-sort.

    Native-source chunks get +rerank_native_boost; low-confidence OCR chunks
    (confidence < rerank_lowconf_threshold) get -rerank_lowconf_penalty. With
    the default weights of 0.0 this is a no-op (stable sort preserves order), so
    retrieval behaviour is unchanged until the weights are tuned via evals.
    """
    nb = settings.rerank_native_boost
    pen = settings.rerank_lowconf_penalty
    if not nb and not pen:
        return reranked
    thr = settings.rerank_lowconf_threshold
    adjusted: list[tuple[DocumentChunk, float]] = []
    for chunk, score in reranked:
        s = score
        source = getattr(chunk, "source", None)
        if nb and source == "native":
            s += nb
        if pen and source == "ocr":
            conf = getattr(chunk, "ocr_confidence", None)
            if conf is not None and conf < thr:
                s -= pen
        adjusted.append((chunk, s))
    adjusted.sort(key=lambda x: -x[1])
    return adjusted


def retrieve(
    db: Session, document_id: str, query: str, page_range: tuple[int, int] | None = None
) -> tuple[list[DocumentChunk], list[DocumentParentChunk], list[float]]:
    """End-to-end: hybrid -> RRF -> rerank -> metadata boost -> fetch parents.
    Returns (reranked_children, parents, rerank_scores)."""
    vec_hits, fts_rows = hybrid_search(db, document_id, query, page_range)

    fused = rrf_fuse(vec_hits, fts_rows, settings.rrf_k)
    fused_ids = [cid for cid, _ in fused[:max(settings.vector_top_k, settings.fts_top_k)]]
    candidates = fetch_chunks_by_ids(db, fused_ids)

    reranked = rerank(query, candidates, settings.rerank_top_n)
    reranked = apply_metadata_boost(reranked)
    children = [c for c, _ in reranked]
    scores = [s for _, s in reranked]

    parents = fetch_parents(db, children)
    logger.info(
        "retrieve: candidates=%d reranked=%d parents=%d top_score=%.3f",
        len(candidates), len(children), len(parents),
        scores[0] if scores else 0.0,
    )
    return children, parents, scores
