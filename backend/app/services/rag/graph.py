"""LangGraph state machine for agentic RAG.

Wiring:
    START -> rewrite_query -> retrieve -> grade
        grade --useful--> generate -> check -> END
        grade --retry-->  retry  -> retrieve  (back-edge, loops)
        grade --give_up--> generate -> check -> END (with NOT_FOUND framing)

Note: the answer-generating node is named 'generate' (not 'answer') because
LangGraph 0.2.x forbids node names that collide with AgentState field names —
'answer' is already a state field.

Streaming: agentic_rag_stream() runs the graph with astream_events and yields
SSE-shaped dicts compatible with the existing chat router.
"""
from __future__ import annotations
import logging
from functools import partial
from typing import AsyncGenerator, Literal
from sqlalchemy.orm import Session
from langgraph.graph import StateGraph, END

from ...config import settings
from .state import AgentState, initial_state
from . import nodes

logger = logging.getLogger(__name__)


def route_after_grade(state: AgentState) -> Literal["generate", "retry", "give_up"]:
    if state.get("graded_useful"):
        return "generate"
    if state.get("retry_count", 0) < settings.max_retrieval_retries:
        return "retry"
    return "give_up"


def build_graph(db: Session):
    """Compile a StateGraph. db is closed-over via partial because the retrieve
    node needs a Session."""
    g = StateGraph(AgentState)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("retrieve", partial(nodes.retrieve_and_rerank, db=db))
    g.add_node("grade", nodes.grade_chunks)
    g.add_node("retry", nodes.rewrite_and_retry)
    g.add_node("generate", nodes.generate_answer)
    g.add_node("check", nodes.faithfulness_check)

    g.set_entry_point("rewrite_query")
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate",
        "retry": "retry",
        "give_up": "generate",
    })
    g.add_edge("retry", "retrieve")
    g.add_edge("generate", "check")
    g.add_edge("check", END)

    return g.compile()


def _build_citations(state: AgentState) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for c in state.get("retrieved_children", []):
        if c.chunk_index in seen:
            continue
        seen.add(c.chunk_index)
        out.append({
            "chunk_index": c.chunk_index,
            "page": getattr(c, "page", None),
            "source": getattr(c, "source", None),
            "content": (c.content or "")[:400],
        })
    return out


async def agentic_rag_stream(
    document_id: str, message: str, db: Session,
) -> AsyncGenerator[dict, None]:
    """Run the graph and yield SSE events: token / citations / warning / done."""
    logger.info("graph stream: doc=%s q=%.120s", document_id, message)
    graph = build_graph(db)
    state_in = initial_state(document_id, message)

    final_state: AgentState = state_in
    answer_chunks: list[str] = []

    async for event in graph.astream_events(state_in, version="v2"):
        kind = event.get("event")
        # Token streaming from the 'generate' node only
        if kind == "on_chat_model_stream":
            node = event.get("metadata", {}).get("langgraph_node")
            if node == "generate":
                chunk = event.get("data", {}).get("chunk")
                content = getattr(chunk, "content", "") if chunk else ""
                if content:
                    answer_chunks.append(content)
                    yield {"type": "token", "content": content}
        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            final_state = event.get("data", {}).get("output", final_state)

    # Update answer in case astream_events didn't surface it via tokens (e.g. mocked)
    if not final_state.get("answer"):
        final_state["answer"] = "".join(answer_chunks)

    for w in final_state.get("warnings", []):
        yield w

    yield {"type": "citations", "chunks": _build_citations(final_state)}

    done_payload: dict = {"type": "done"}
    if settings.log_level.upper() == "DEBUG":
        done_payload["debug"] = {
            "attempted_queries": final_state.get("attempted_queries", []),
            "retry_count": final_state.get("retry_count", 0),
            "intent": final_state.get("intent"),
            "notes": final_state.get("notes", []),
        }
    yield done_payload
