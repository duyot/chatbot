import uuid
import pytest
from unittest.mock import patch, MagicMock

def make_doc(doc_id=None, status="pending", file_path="/tmp/test.pdf", file_name="test.pdf"):
    doc = MagicMock()
    doc.id = doc_id or str(uuid.uuid4())
    doc.status = status
    doc.file_path = file_path
    doc.file_name = file_name
    doc.error_msg = None
    return doc

def test_ingest_document_sets_done_on_success():
    from app.workers.tasks import ingest_document
    doc_id = str(uuid.uuid4())
    mock_doc = make_doc(doc_id=doc_id)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch("app.workers.tasks.SessionLocal", return_value=mock_db), \
         patch("app.workers.tasks.parse_file", return_value="some text"), \
         patch("app.workers.tasks.chunk_text", return_value=["chunk1"]), \
         patch("app.workers.tasks.embed_chunks", return_value=[[0.1] * 1536]), \
         patch("app.workers.tasks.store_chunks"):
        ingest_document.apply(args=[doc_id])

    assert mock_doc.status == "done"

def test_ingest_document_sets_failed_on_exception():
    from app.workers.tasks import ingest_document
    doc_id = str(uuid.uuid4())
    mock_doc = make_doc(doc_id=doc_id)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch("app.workers.tasks.SessionLocal", return_value=mock_db), \
         patch("app.workers.tasks.parse_file", side_effect=RuntimeError("parse error")):
        ingest_document.apply(args=[doc_id])

    assert mock_doc.status == "failed"
    assert "parse error" in (mock_doc.error_msg or "")
