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
from ...observability import chunk_summary, emit, full_payloads, timed, trunc
from ..ingestion import embed_text
from .reranker import rerank

logger = logging.getLogger(__name__)

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
            logger.info("bm25_available: pg_search detected=%s", _BM25_AVAILABLE)
        except Exception as exc:  # noqa: BLE001
            # Probe errored (e.g. transient DB blip) — distinct from probing
            # successfully and finding pg_search genuinely absent. Don't latch
            # False into the module cache, or one bad connection makes this
            # process run unindexed FTS for its whole lifetime; leave the
            # cache unset so the next call retries.
            logger.warning("bm25_available: probe failed, using ts_rank (%s)", exc)
            return False
    return _BM25_AVAILABLE


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
    with timed() as embed_ms:
        vector = embed_text(query)
    emit(
        "retrieval.embed_query",
        query=trunc(query),
        model=settings.openai_embedding_model,
        dims=len(vector),
        ms=embed_ms(),
    )

    vec_q = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id)
    if page_range is not None:
        vec_q = vec_q.filter(DocumentChunk.page.between(page_range[0], page_range[1]))
    with timed() as vec_ms:
        vec_hits = (
            vec_q.order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(settings.vector_top_k)
            .all()
        )

    with timed() as kw_ms:
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

    logger.info(
        "hybrid_search: query=%.80s vec=%d keyword=%d arm=%s page_range=%s",
        query, len(vec_hits), len(fts_rows), arm, page_range,
    )
    emit(
        "retrieval.arm",
        arm="vector",
        query=trunc(query),
        top_k=settings.vector_top_k,
        page_range=page_range,
        n=len(vec_hits),
        ms=vec_ms(),
        hits=[chunk_summary(c, rank=i) for i, c in enumerate(vec_hits)],
    )
    emit(
        "retrieval.arm",
        arm=arm,
        query=trunc(query),
        top_k=settings.fts_top_k,
        page_range=page_range,
        n=len(fts_rows),
        ms=kw_ms(),
        # Keyword rows are (id, rank) tuples, not ORM chunks — the row's `rank`
        # is the arm's own relevance score, distinct from its position.
        hits=[
            {"chunk_id": str(r.id), "rank": i, "score": float(r.rank)}
            for i, r in enumerate(fts_rows)
        ],
    )
    return vec_hits, fts_rows


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
    moved: list[dict] = []
    for chunk, score in reranked:
        s = score
        source = getattr(chunk, "source", None)
        if nb and source == "native":
            s += nb
        if pen and source == "ocr":
            conf = getattr(chunk, "ocr_confidence", None)
            if conf is not None and conf < thr:
                s -= pen
        if s != score:
            moved.append({
                **chunk_summary(chunk),
                "rerank_score": round(float(score), 6),
                "boosted_score": round(float(s), 6),
            })
        adjusted.append((chunk, s))
    adjusted.sort(key=lambda x: -x[1])
    if moved:
        emit(
            "retrieval.boost",
            native_boost=nb,
            lowconf_penalty=pen,
            lowconf_threshold=thr,
            n_adjusted=len(moved),
            adjusted=moved,
        )
    return adjusted


def retrieve(
    db: Session, document_id: str, query: str, page_range: tuple[int, int] | None = None
) -> tuple[list[DocumentChunk], list[DocumentParentChunk], list[float]]:
    """End-to-end: hybrid -> RRF -> rerank -> metadata boost -> fetch parents.
    Returns (reranked_children, parents, rerank_scores)."""
    with timed() as total_ms:
        return _retrieve(db, document_id, query, page_range, total_ms)


def _retrieve(
    db: Session,
    document_id: str,
    query: str,
    page_range: tuple[int, int] | None,
    total_ms,
) -> tuple[list[DocumentChunk], list[DocumentParentChunk], list[float]]:
    """Body of retrieve(), split out only so the timer wrapping it stays a
    context manager rather than a try/finally around every return path."""
    vec_hits, fts_rows = hybrid_search(db, document_id, query, page_range)

    fused = rrf_fuse(vec_hits, fts_rows, settings.rrf_k)
    # RRF order must actually SELECT candidates, or the weights are inert:
    # rerank() re-sorts whatever it receives, so if every fused id survives,
    # fusion cannot influence the final result. Both arms return up to top_k
    # and the fused set holds their union, so truncating at the larger arm's
    # top_k is what makes the weighting load-bearing. Using the SUM here
    # (an earlier revision did) can never truncate and silently disables it.
    candidate_limit = max(settings.vector_top_k, settings.fts_top_k)
    fused_ids = [cid for cid, _ in fused[:candidate_limit]]
    candidates = fetch_chunks_by_ids(db, fused_ids)
    emit(
        "retrieval.fuse",
        rrf_k=settings.rrf_k,
        w_vector=settings.rrf_weight_vector,
        w_keyword=settings.rrf_weight_keyword,
        fused=len(fused),
        candidate_limit=candidate_limit,
        kept=len(fused_ids),
        # Non-zero means the weights are actually load-bearing; a persistent 0
        # here says fusion cannot influence the final ranking. See the comment
        # above candidate_limit.
        dropped=max(0, len(fused) - len(fused_ids)),
        fetched=len(candidates),
        top=[
            {"chunk_id": str(cid), "rank": i, "score": round(score, 6)}
            for i, (cid, score) in enumerate(fused[:candidate_limit])
        ],
    )

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
    emit(
        "retrieval.result",
        query=trunc(query),
        candidates=len(candidates),
        n_children=len(children),
        n_parents=len(parents),
        top_score=round(scores[0], 6) if scores else None,
        ms=total_ms(),
        children=[
            chunk_summary(c, score=s, rank=i)
            for i, (c, s) in enumerate(zip(children, scores))
        ],
        parents=[
            {
                "parent_id": str(p.id),
                "parent_index": getattr(p, "parent_index", None),
                "page_start": getattr(p, "page_start", None),
                "page_end": getattr(p, "page_end", None),
                "source": getattr(p, "source", None),
                "chars": len(getattr(p, "content", "") or ""),
                **({"content": p.content} if full_payloads() else {}),
            }
            for p in parents
        ],
    )
    return children, parents, scores
