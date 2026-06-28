"""LLM-as-reranker backed by OpenRouter chat completions.

We send the query + numbered candidate passages to an OpenRouter-hosted chat
model and ask for a JSON list of {index, score} scored 0..10 by relevance.

History: this used to call HuggingFace TEI's /rerank endpoint with a
bge-reranker-v2-m3 cross-encoder. Before that, it called Ollama's /api/embed
and read embedding[0] as a "score" — which was effectively random. The
OpenRouter LLM path costs tokens and adds ~1-3s latency per call but removes
the TEI dependency from the stack.

Score range is 0..10. Set settings.rerank_score_floor above 0 to filter weak
hits; default (-1e9) means "pass everything through, top_n decides".
"""
from __future__ import annotations
import logging
from typing import Any, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from ...config import settings

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a relevance grader. Given a user query and a numbered list of "
    "passages, score each passage from 0.0 (irrelevant) to 10.0 (perfectly "
    "answers the query). Be strict: most passages should score below 5.0. "
    "Return a JSON object with key 'scores' containing one entry per passage. "
    "Every input index must appear exactly once."
)


class _RankedItem(BaseModel):
    index: int = Field(..., description="0-based index of the passage")
    score: float = Field(..., description="Relevance score in 0..10")


class _RankResult(BaseModel):
    scores: List[_RankedItem]


def get_reranker() -> str:
    """Return the configured reranker model name. Kept for API compatibility."""
    return settings.reranker_model


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.reranker_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=0.0,
    )


def rerank(query: str, chunks: list, top_n: int) -> list[tuple[Any, float]]:
    """Score each chunk via OpenRouter LLM, return top_n (chunk, score) sorted desc.

    `chunks` is a list with .id and .content attributes (DocumentChunk works directly).
    """
    if not chunks:
        return []

    passages = "\n\n".join(
        f"[{i}] {(c.content or '').strip()}" for i, c in enumerate(chunks)
    )
    user_msg = (
        f"Query:\n{query}\n\n"
        f"Passages (n={len(chunks)}):\n{passages}\n\n"
        f"Score every passage. Return only the JSON."
    )

    logger.info(
        "rerank: request query=%.80s n_passages=%d model=%s",
        query, len(chunks), settings.reranker_model,
    )

    llm = _build_llm().with_structured_output(_RankResult)
    try:
        result: _RankResult = llm.invoke([
            SystemMessage(_SYSTEM_PROMPT),
            HumanMessage(user_msg),
        ])
    except Exception as exc:
        logger.error("rerank: LLM call failed (%s); falling back to original order", exc)
        return [(c, 0.0) for c in chunks[:top_n]]

    seen: set[int] = set()
    scored: list[tuple[Any, float]] = []
    for item in result.scores:
        if item.index in seen or item.index < 0 or item.index >= len(chunks):
            logger.warning("rerank: skipping result with invalid index=%r", item.index)
            continue
        seen.add(item.index)
        scored.append((chunks[item.index], float(item.score)))

    missing = [i for i in range(len(chunks)) if i not in seen]
    if missing:
        logger.warning("rerank: LLM omitted %d indices; appending with score 0.0", len(missing))
        for i in missing:
            scored.append((chunks[i], 0.0))

    scored.sort(key=lambda x: -x[1])
    logger.info(
        "rerank: model=%s n_chunks=%d top_score=%.3f bottom_score=%.3f",
        settings.reranker_model, len(scored),
        scored[0][1] if scored else 0.0,
        scored[-1][1] if scored else 0.0,
    )
    return scored[:top_n]
