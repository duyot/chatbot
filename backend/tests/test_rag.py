import asyncio
import uuid
from unittest.mock import MagicMock


def test_make_search_tool_returns_parents(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk
    from app.services._rag_legacy import make_search_tool

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    parent = DocumentParentChunk(document_id=doc_id, parent_index=0,
                                 content="The revenue was $100M in Q3.")
    db.add(parent)
    db.flush()
    child = DocumentChunk(document_id=doc_id, parent_id=parent.id, chunk_index=0,
                          content="Revenue $100M Q3", embedding=[0.1] * 2560)
    db.add(child)
    db.flush()

    fake_children = [child]
    fake_parents = [parent]
    fake_scores = [0.9]
    mocker.patch(
        "app.services._rag_legacy.retrieve",
        return_value=(fake_children, fake_parents, fake_scores),
    )

    collected_c, collected_p, collected_s = [], [], []
    tool = make_search_tool(str(doc_id), db, collected_c, collected_p, collected_s)
    result = tool.invoke({"query": "Q3 revenue"})

    assert "The revenue was $100M in Q3." in result
    assert collected_p == [parent]


def test_make_search_tool_no_results_sentinel(mocker, db):
    from app.services._rag_legacy import make_search_tool

    mocker.patch(
        "app.services._rag_legacy.retrieve",
        return_value=([], [], []),
    )
    collected_c, collected_p, collected_s = [], [], []
    tool = make_search_tool("doc-id", db, collected_c, collected_p, collected_s)
    assert tool.invoke({"query": "x"}) == "NO_RELEVANT_CHUNKS"


def test_agentic_rag_stream_yields_tokens_and_citations(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    parent = DocumentParentChunk(document_id=doc_id, parent_index=0, content="Q3 revenue $100M")
    db.add(parent)
    db.flush()
    child = DocumentChunk(document_id=doc_id, parent_id=parent.id, chunk_index=0,
                          content="Q3 revenue $100M", embedding=[0.1] * 2560)
    db.add(child)
    db.flush()

    mocker.patch(
        "app.services._rag_legacy.retrieve",
        return_value=([child], [parent], [0.9]),
    )

    mock_ai_with_tc = MagicMock()
    mock_ai_with_tc.tool_calls = [{"id": "tc1", "args": {"query": "Q3 revenue"}}]
    mock_ai_no_tc = MagicMock()
    mock_ai_no_tc.tool_calls = []

    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.ainvoke = mocker.AsyncMock(side_effect=[mock_ai_with_tc, mock_ai_no_tc])

    mock_token = MagicMock()
    mock_token.content = "Revenue $100M."

    async def mock_astream(_messages):
        yield mock_token

    mock_llm = MagicMock()
    mock_llm.bind_tools.return_value = mock_llm_with_tools
    mock_llm.astream = mock_astream
    mocker.patch("app.services._rag_legacy.ChatOllama", return_value=mock_llm)

    from app.services._rag_legacy import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "What was Q3 revenue?", db)]

    events = asyncio.run(run())
    token_events = [e for e in events if e["type"] == "token"]
    citations = next(e for e in events if e["type"] == "citations")
    assert token_events[0]["content"] == "Revenue $100M."
    assert len(citations["chunks"]) == 1
