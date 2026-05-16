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


async def rewrite_and_retry(state: AgentState) -> dict:
    attempted = state.get("attempted_queries", [])
    llm = _chat_llm(temperature=0.3)  # a little creativity for alt phrasing
    response = await llm.ainvoke([
        HumanMessage(prompts.RETRY_QUERY_PROMPT.format(attempted=attempted)),
    ])
    new_query = (response.content or "").strip().strip('"').strip("'")
    return {
        "rewritten_query": new_query,
        "retry_count": state.get("retry_count", 0) + 1,
        "notes": state.get("notes", []) + [f"retry({state.get('retry_count', 0)+1}): {new_query[:60]}"],
    }


async def generate_answer(state: AgentState) -> dict:
    """NOTE: streaming is handled at the graph level via astream_events.
    This node still calls the LLM and the streamed tokens are picked up
    by the graph layer. We accumulate the full answer here for the
    faithfulness_check node."""
    system_prompt = (
        prompts.ANSWER_SYSTEM_GROUNDED
        if state.get("graded_useful")
        else prompts.ANSWER_SYSTEM_NOT_FOUND
    )
    context = "\n\n---\n\n".join(p.content for p in state.get("parents", []))
    llm = _chat_llm()
    response = await llm.ainvoke([
        SystemMessage(system_prompt),
        HumanMessage(f"Document context:\n{context}\n\nQuestion: {state['question']}"),
    ])
    answer = response.content or ""
    return {"answer": answer,
            "notes": state.get("notes", []) + [f"answer: len={len(answer)}"]}


async def faithfulness_check(state: AgentState) -> dict:
    if not state.get("answer"):
        return {}
    context = "\n\n---\n\n".join(p.content for p in state.get("parents", []))
    llm = _chat_llm()
    response = await llm.ainvoke([
        HumanMessage(prompts.FAITHFULNESS_PROMPT.format(
            question=state["question"],
            context=context,
            answer=state["answer"],
        )),
    ])
    verdict = (response.content or "").strip().upper().startswith("YES")
    warnings = list(state.get("warnings", []))
    if not verdict:
        warnings.append({
            "type": "warning",
            "message": "Some claims may not be fully supported by the document.",
        })
    return {"warnings": warnings,
            "notes": state.get("notes", []) + [f"faithfulness: {verdict}"]}
