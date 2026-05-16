import logging
from typing import AsyncGenerator
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from ..config import settings
from .rag.retrieval import retrieve

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided "
    "document. Use the search_document tool to retrieve relevant context before answering. "
    "You may search up to 3 times with different query phrasings if the first results "
    "are insufficient. Quote relevant passages verbatim when they directly answer the "
    "question. If the answer is not in the retrieved context, say so."
)


def make_search_tool(document_id: str, db: Session, collected_children: list,
                     collected_parents: list, collected_scores: list):
    @tool
    def search_document(query: str) -> str:
        """Search the document for passages relevant to the query.
        Call with different phrasings if first results are insufficient."""
        logger.info("search_document query=%.120s", query)
        children, parents, scores = retrieve(db, document_id, query)
        collected_children.extend(children)
        collected_parents.extend(parents)
        collected_scores.extend(scores)
        if not parents:
            return "NO_RELEVANT_CHUNKS"
        return "\n\n---\n\n".join(p.content for p in parents)

    return search_document


async def agentic_rag_stream(
    document_id: str, message: str, db: Session,
) -> AsyncGenerator[dict, None]:
    logger.info("agentic_rag_stream: doc=%s q=%.120s", document_id, message)
    collected_children, collected_parents, collected_scores = [], [], []
    search_tool = make_search_tool(
        document_id, db, collected_children, collected_parents, collected_scores
    )

    llm = ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url)
    llm_with_tools = llm.bind_tools([search_tool])

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(message)]

    for round_num in range(3):
        response = await llm_with_tools.ainvoke(messages)
        if not response.tool_calls:
            logger.info("agentic_rag_stream: no more tool calls after round=%d", round_num)
            break
        messages.append(response)
        for tc in response.tool_calls:
            result = search_tool.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    # Citations: emit reranked children (best signal for "what backs the answer")
    seen: set = set()
    uniq = []
    for c in collected_children:
        if c.chunk_index not in seen:
            seen.add(c.chunk_index)
            uniq.append({"chunk_index": c.chunk_index, "content": c.content[:400]})

    logger.info("agentic_rag_stream: done citations=%d", len(uniq))
    yield {"type": "citations", "chunks": uniq}
    yield {"type": "done"}
