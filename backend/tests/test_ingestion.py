import os
import pytest
from unittest.mock import MagicMock, patch

def test_chunk_text_splits_long_content():
    from app.services.ingestion import chunk_text
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 1200  # chunk_size=1000 + some overlap buffer

def test_chunk_text_short_content_stays_one_chunk():
    from app.services.ingestion import chunk_text
    text = "Short paragraph."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == "Short paragraph."

def test_parse_docx(tmp_path):
    from docx import Document as DocxDocument
    from app.services.ingestion import parse_file
    docx_path = tmp_path / "test.docx"
    doc = DocxDocument()
    doc.add_paragraph("Hello from DOCX")
    doc.save(str(docx_path))
    result = parse_file(str(docx_path), "test.docx")
    assert "Hello from DOCX" in result

def test_parse_image_returns_placeholder():
    from app.services.ingestion import parse_file
    result = parse_file("/any/path/photo.png", "photo.png")
    assert result == "[image: photo.png]"

def test_embed_chunks_calls_openai_and_returns_vectors():
    from app.services.ingestion import embed_chunks
    fake_embedding = [0.1] * 1536
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding)]
    with patch("app.services.ingestion.OpenAI") as MockOpenAI:
        mock_client = MockOpenAI.return_value
        mock_client.embeddings.create.return_value = mock_response
        result = embed_chunks(["some text"])
    assert len(result) == 1
    assert len(result[0]) == 1536

def test_store_chunks_inserts_rows():
    from app.services.ingestion import store_chunks
    from app.models import DocumentChunk
    import uuid

    mock_db = MagicMock()
    doc_id = str(uuid.uuid4())
    chunks = ["chunk one", "chunk two"]
    embeddings = [[0.1] * 1536, [0.2] * 1536]

    store_chunks(mock_db, doc_id, chunks, embeddings)

    mock_db.bulk_save_objects.assert_called_once()
    saved_objects = mock_db.bulk_save_objects.call_args[0][0]
    assert len(saved_objects) == 2
    assert isinstance(saved_objects[0], DocumentChunk)
    assert saved_objects[0].chunk_index == 0
    assert saved_objects[1].chunk_index == 1
    mock_db.commit.assert_called_once()
