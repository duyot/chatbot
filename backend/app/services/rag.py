import logging
from typing import AsyncGenerator
from sqlalchemy.orm import Session
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from ..config import settings
from ..models import DocumentChunk
from .ingestion import embed_text

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions strictly based on the provided document. "
    "Use the search_document tool to retrieve relevant context before answering. "
    "You may search up to 3 times with different query phrasings if the first results are insufficient. "
    "Once you have sufficient context, provide a clear and thorough answer."
)


def make_search_tool(document_id: str, db: Session, retrieved_chunks: list):
    @tool
    def search_document(query: str) -> str:
        """Search the document for chunks relevant to the query.
        Call with different phrasings if the first results are insufficient."""
        vector = embed_text(query)
        chunks = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(5)
            .all()
        )
        retrieved_chunks.extend(chunks)
        return "\n\n---\n\n".join(c.content for c in chunks) or "No relevant content found."

    return search_document


async def agentic_rag_stream(
    document_id: str,
    message: str,
    db: Session,
) -> AsyncGenerator[dict, None]:
    logger.info("agentic_rag_stream: start document_id=%s query=%.120s", document_id, message)
    retrieved_chunks: list = []
    search_tool = make_search_tool(document_id, db, retrieved_chunks)

    llm = ChatOllama(model=settings.ollama_chat_model, base_url=settings.ollama_base_url)
    llm_with_tools = llm.bind_tools([search_tool])

    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(message)]

    # Phase 1: Tool-calling rounds (not streamed — model is reasoning/acting)
    for round_num in range(3):
        response = await llm_with_tools.ainvoke(messages)
        if not response.tool_calls:
            logger.info("agentic_rag_stream: no more tool calls after round=%d", round_num)
            break
        messages.append(response)
        for tc in response.tool_calls:
            logger.info("agentic_rag_stream: tool_call=%s args=%.120s", tc["name"], str(tc["args"]))
            result = search_tool.invoke(tc["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    logger.info("agentic_rag_stream: retrieved_chunks=%d", len(retrieved_chunks))

    # Phase 2: Final answer generation (streamed)
    logger.info("agentic_rag_stream: streaming final answer")
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield {"type": "token", "content": chunk.content}

    # Emit deduplicated citations (truncated to 400 chars for UI)
    seen: set = set()
    unique_chunks = []
    for c in retrieved_chunks:
        if c.chunk_index not in seen:
            seen.add(c.chunk_index)
            unique_chunks.append({"chunk_index": c.chunk_index, "content": c.content[:400]})

    logger.info("agentic_rag_stream: done citations=%d document_id=%s", len(unique_chunks), document_id)
    yield {"type": "citations", "chunks": unique_chunks}
    yield {"type": "done"}
