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
