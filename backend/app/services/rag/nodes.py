"""LangGraph nodes for the agentic RAG pipeline.

Each node is `async def node(state: AgentState) -> dict` returning a partial state.
LangGraph merges the partial back into the full state.
"""
from __future__ import annotations
import logging
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from ...config import settings
from . import prompts
from .state import AgentState

logger = logging.getLogger(__name__)


def _chat_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_chat_model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
    )


class QueryRewrite(BaseModel):
    rewritten_query: str = Field(..., description="The cleaned search query.")
    intent: Literal["lookup", "summary", "reasoning", "unclear"]


async def rewrite_query(state: AgentState) -> dict:
    llm = _chat_llm()
    structured = llm.with_structured_output(QueryRewrite)
    response: QueryRewrite = await structured.ainvoke([
        SystemMessage(prompts.REWRITE_QUERY_SYSTEM),
        HumanMessage(state["question"]),
    ])
    logger.info(
        "rewrite_query: q=%.80s -> rewritten=%.80s intent=%s",
        state["question"], response.rewritten_query, response.intent,
    )
    return {
        "rewritten_query": response.rewritten_query,
        "intent": response.intent,
        "notes": state.get("notes", []) + [f"rewrite: {response.intent}"],
    }


from sqlalchemy.orm import Session
from .retrieval import retrieve


async def retrieve_and_rerank(state: AgentState, db: Session) -> dict:
    query = state["rewritten_query"] or state["question"]
    children, parents, scores = retrieve(db, state["document_id"], query)
    attempted = state.get("attempted_queries", []) + [query]
    note = (
        f"retrieve: children={len(children)} parents={len(parents)} "
        f"top_score={scores[0] if scores else 0:.3f}"
    )
    logger.info(note)
    return {
        "retrieved_children": children,
        "parents": parents,
        "rerank_scores": scores,
        "attempted_queries": attempted,
        "notes": state.get("notes", []) + [note],
    }


async def grade_chunks(state: AgentState) -> dict:
    """Fast path by default. Strict LLM path if settings.strict_grader=True."""
    children = state.get("retrieved_children") or []
    scores = state.get("rerank_scores") or []

    if not children:
        return {"graded_useful": False,
                "notes": state.get("notes", []) + ["grade: no_chunks"]}

    if not settings.strict_grader:
        useful = max(scores, default=0.0) >= settings.rerank_score_floor
        return {"graded_useful": useful,
                "notes": state.get("notes", []) + [f"grade(fast): useful={useful}"]}

    # Strict path: LLM judge
    passages = "\n\n---\n\n".join(c.content for c in children[:5])
    llm = _chat_llm()
    response = await llm.ainvoke([
        SystemMessage("You judge whether retrieved passages contain the answer."),
        HumanMessage(prompts.GRADE_CHUNKS_PROMPT.format(
            question=state["question"], passages=passages,
        )),
    ])
    verdict = (response.content or "").strip().upper().startswith("YES")
    return {"graded_useful": verdict,
            "notes": state.get("notes", []) + [f"grade(strict): {verdict}"]}
