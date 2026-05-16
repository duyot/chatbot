import re
import logging
from typing import AsyncGenerator
from uuid import UUID
from sqlalchemy import text as sa_text
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
    "You may search up to 3 times with different query phrasings if the first results are insufficient.\n\n"
    "Search strategy:\n"
    "- If the question asks for the value of a named field or property "
    "(e.g. 'what is Corporate Name?', 'what is the Registration Number?'), "
    "search using the exact field name as the query (e.g. 'Corporate Name', 'Registration Number'). "
    "This produces better results for structured documents than asking a full question.\n"
    "- If the first search returns no useful results, try a shorter or synonymous term.\n"
    "- Once you have sufficient context, provide a clear and thorough answer."
)

_QUESTION_PREFIXES = re.compile(
    r"^(?:what\s+is\s+(?:the\s+)?(?:value\s+of\s+)?|"
    r"what\s+are\s+(?:the\s+)?|"
    r"tell\s+me\s+(?:the\s+)?|"
    r"show\s+me\s+(?:the\s+)?|"
    r"find\s+(?:the\s+)?|"
    r"give\s+me\s+(?:the\s+)?)",
    re.IGNORECASE,
)
_QUESTION_SUFFIXES = re.compile(
    r"[?.]?\s*(?:in\s+(?:the\s+)?document|from\s+(?:the\s+)?document)?[?.]?\s*$",
    re.IGNORECASE,
)


def _preprocess_query(query: str) -> str:
    """Strip question framing to expose the core field/entity name for better embedding and FTS."""
    stripped = _QUESTION_PREFIXES.sub("", query.strip())
    stripped = _QUESTION_SUFFIXES.sub("", stripped).strip()
    return stripped if stripped else query.strip()


def make_search_tool(document_id: str, db: Session, retrieved_chunks: list):
    @tool
    def search_document(query: str) -> str:
        """Search the document for chunks relevant to the query.
        Call with different phrasings if the first results are insufficient."""
        logger.info("search_document raw_query=%.120s", query)

        core_query = _preprocess_query(query)
        logger.info("search_document core_query=%.120s", core_query)

        # Leg 1: vector search (semantic similarity)
        vector = embed_text(core_query)
        vector_chunks: list[DocumentChunk] = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.embedding.cosine_distance(vector))
            .limit(8)
            .all()
        )

        # Leg 2: PostgreSQL full-text search (keyword match)
        fts_sql = sa_text(
            """
            SELECT id
            FROM   document_chunks
            WHERE  document_id = :doc_id
              AND  to_tsvector('english', content)
                   @@ plainto_tsquery('english', :q)
            ORDER BY ts_rank(to_tsvector('english', content),
                             plainto_tsquery('english', :q)) DESC
            LIMIT  8
            """
        )
        fts_rows = db.execute(fts_sql, {"doc_id": document_id, "q": core_query}).fetchall()
        fts_ids: set[UUID] = {row.id for row in fts_rows}

        # Fetch FTS-only hits not already in vector results
        vector_ids: set[UUID] = {c.id for c in vector_chunks}
        extra_ids = fts_ids - vector_ids
        extra_chunks: list[DocumentChunk] = []
        if extra_ids:
            extra_chunks = (
                db.query(DocumentChunk)
                .filter(DocumentChunk.id.in_(extra_ids))
                .all()
            )

        # Merge: vector results first (semantically ranked), then FTS-only additions
        seen_ids: set[UUID] = set()
        unique: list[DocumentChunk] = []
        for c in list(vector_chunks) + extra_chunks:
            if c.id not in seen_ids:
                seen_ids.add(c.id)
                unique.append(c)

        final = unique[:8]
        logger.info(
            "search_document results: vector=%d fts_extra=%d total=%d",
            len(vector_chunks),
            len(extra_chunks),
            len(final),
        )

        if not final:
            logger.warning("search_document no chunks found for query=%.120s", query)

        retrieved_chunks.extend(final)
        return "\n\n---\n\n".join(c.content for c in final) or "No relevant content found."

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
