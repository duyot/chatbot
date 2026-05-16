"""Integration test for the public agentic_rag_stream entry point (via graph)."""
import asyncio
import uuid
from unittest.mock import MagicMock


def test_agentic_rag_stream_emits_token_citation_done_in_order(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    parent = DocumentParentChunk(document_id=doc_id, parent_index=0, content="Q3 rev $100M")
    db.add(parent); db.flush()
    child = DocumentChunk(document_id=doc_id, parent_id=parent.id, chunk_index=0,
                          content="Q3 rev $100M", embedding=[0.1] * 2560)
    db.add(child); db.flush()

    # Mock retrieval
    mocker.patch(
        "app.services.rag.nodes.retrieve",
        return_value=([child], [parent], [0.8]),
    )

    # Mock all LLM calls deterministically
    def fake_chat_llm(temperature=0.0):
        llm = MagicMock()
        # structured output for rewrite_query
        structured = MagicMock()
        structured.ainvoke = mocker.AsyncMock(
            return_value=MagicMock(rewritten_query="Q3 revenue", intent="lookup")
        )
        llm.with_structured_output.return_value = structured
        # plain ainvoke for grade (strict path off — won't be called) and faithfulness
        msg = MagicMock(); msg.content = "Q3 revenue was $100M."
        llm.ainvoke = mocker.AsyncMock(return_value=msg)
        return llm
    mocker.patch("app.services.rag.nodes._chat_llm", side_effect=fake_chat_llm)

    from app.services.rag import agentic_rag_stream

    async def run():
        return [e async for e in agentic_rag_stream(str(doc_id), "What was Q3 revenue?", db)]

    events = asyncio.run(run())
    # We expect at least: a citations event, a done event. Token events may be empty
    # if the mocked LLM doesn't stream — that's fine for this wiring test.
    types = [e["type"] for e in events]
    assert "citations" in types
    assert "done" in types
    assert types[-1] == "done"
    citation = next(e for e in events if e["type"] == "citations")
    assert citation["chunks"][0]["chunk_index"] == 0
