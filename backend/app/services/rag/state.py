"""LangGraph state for the agentic RAG pipeline."""
from __future__ import annotations
from typing import TypedDict, Literal


class AgentState(TypedDict, total=False):
    # Inputs
    document_id: str
    question: str

    # Routing / query
    rewritten_query: str
    intent: Literal["lookup", "summary", "reasoning", "unclear"]
    attempted_queries: list[str]
    retry_count: int

    # Retrieval results
    retrieved_children: list   # list[DocumentChunk]
    parents: list              # list[DocumentParentChunk]
    rerank_scores: list[float]

    # Grading
    graded_useful: bool

    # Generation
    answer: str
    answer_chunks: list[str]

    # Output
    citations: list[dict]
    warnings: list[dict]
    notes: list[str]


def initial_state(document_id: str, question: str) -> AgentState:
    return {
        "document_id": document_id,
        "question": question,
        "rewritten_query": "",
        "intent": "lookup",
        "attempted_queries": [],
        "retry_count": 0,
        "retrieved_children": [],
        "parents": [],
        "rerank_scores": [],
        "graded_useful": False,
        "answer": "",
        "answer_chunks": [],
        "citations": [],
        "warnings": [],
        "notes": [],
    }
