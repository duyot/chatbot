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
