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
