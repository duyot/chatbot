import asyncio
import uuid
import pytest
from app.models import Document, DocumentChunk


def test_search_tool_returns_formatted_chunks(mocker, db):
    mocker.patch("app.services.rag.embed_text", return_value=[0.1] * 1536)

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="test.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    chunk = DocumentChunk(
        document_id=doc_id,
        chunk_index=0,
        content="The revenue was $100M in Q3.",
        embedding=[0.1] * 1536,
    )
    db.add(chunk)
    db.flush()

    from app.services.rag import make_search_tool

    retrieved = []
    tool = make_search_tool(str(doc_id), db, retrieved)
    result = tool.invoke({"query": "Q3 revenue"})

    assert "The revenue was $100M in Q3." in result
    assert len(retrieved) == 1
    assert retrieved[0].chunk_index == 0


def test_search_tool_returns_no_results_message(mocker, db):
    mocker.patch("app.services.rag.embed_text", return_value=[0.9] * 1536)

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="empty.pdf", file_path="/tmp/e.pdf", status="done")
    db.add(doc)
    db.flush()

    from app.services.rag import make_search_tool

    retrieved = []
    tool = make_search_tool(str(doc_id), db, retrieved)
    result = tool.invoke({"query": "something"})

    assert result == "No relevant content found."
    assert retrieved == []


def test_agentic_rag_stream_yields_tokens_and_citations(mocker, db):
    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="test.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    chunk = DocumentChunk(
        document_id=doc_id,
        chunk_index=0,
        content="The revenue was $100M in Q3.",
        embedding=[0.1] * 1536,
    )
    db.add(chunk)
    db.flush()

    mocker.patch("app.services.rag.embed_text", return_value=[0.1] * 1536)

    mock_ai_with_tc = mocker.MagicMock()
    mock_ai_with_tc.tool_calls = [{"id": "tc1", "args": {"query": "Q3 revenue"}}]

    mock_ai_no_tc = mocker.MagicMock()
    mock_ai_no_tc.tool_calls = []

    mock_llm_with_tools = mocker.MagicMock()
    mock_llm_with_tools.ainvoke = mocker.AsyncMock(
        side_effect=[mock_ai_with_tc, mock_ai_no_tc]
    )

    mock_token1 = mocker.MagicMock()
    mock_token1.content = "Revenue"
    mock_token2 = mocker.MagicMock()
    mock_token2.content = " was $100M."

    async def mock_astream(_messages):
        for tok in [mock_token1, mock_token2]:
            yield tok

    mock_llm = mocker.MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm.astream = mock_astream

    mocker.patch("app.services.rag.ChatOllama", return_value=mock_llm)

    from app.services.rag import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "What was Q3 revenue?", db)]

    events = asyncio.run(run())

    token_events = [e for e in events if e["type"] == "token"]
    citation_event = next(e for e in events if e["type"] == "citations")
    done_event = next((e for e in events if e["type"] == "done"), None)

    assert len(token_events) == 2
    assert token_events[0]["content"] == "Revenue"
    assert token_events[1]["content"] == " was $100M."
    assert len(citation_event["chunks"]) == 1
    assert citation_event["chunks"][0]["chunk_index"] == 0
    assert done_event is not None


def test_agentic_rag_stream_skips_empty_token_content(mocker, db):
    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.flush()

    mocker.patch("app.services.rag.embed_text", return_value=[0.1] * 1536)

    mock_ai_no_tc = mocker.MagicMock()
    mock_ai_no_tc.tool_calls = []

    mock_llm_with_tools = mocker.MagicMock()
    mock_llm_with_tools.ainvoke = mocker.AsyncMock(return_value=mock_ai_no_tc)

    mock_empty = mocker.MagicMock()
    mock_empty.content = ""
    mock_real = mocker.MagicMock()
    mock_real.content = "Hello"

    async def mock_astream(_messages):
        for tok in [mock_empty, mock_real]:
            yield tok

    mock_llm = mocker.MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm.astream = mock_astream

    mocker.patch("app.services.rag.ChatOllama", return_value=mock_llm)

    from app.services.rag import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "hi", db)]

    events = asyncio.run(run())
    token_events = [e for e in events if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["content"] == "Hello"
