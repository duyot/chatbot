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
    from app.services.ingestion import ParsedDocument, PageContent, ParentChunk, ChildChunk
    doc_id = str(uuid.uuid4())
    mock_doc = make_doc(doc_id=doc_id)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    parsed = ParsedDocument(
        pages=[PageContent(page=1, text="some text", source="native")],
        metadata={"mime_type": "application/pdf", "page_count": 1, "ocr_pages": 0},
    )
    parents = [ParentChunk(content="parent1", page_start=1, page_end=1, source="native")]
    children = [[ChildChunk(content="chunk1", page=1, source="native")]]

    with patch("app.workers.tasks.SessionLocal", return_value=mock_db), \
         patch("app.workers.tasks.parse_document", return_value=parsed), \
         patch("app.workers.tasks.chunk_document", return_value=(parents, children)), \
         patch("app.workers.tasks.embed_chunks", return_value=[[0.1] * 1536]), \
         patch("app.workers.tasks.store_chunks"):
        ingest_document.apply(args=[doc_id])

    assert mock_doc.status == "done"
    assert mock_doc.page_count == 1
    assert mock_doc.mime_type == "application/pdf"

def test_ingest_document_sets_failed_on_exception():
    from app.workers.tasks import ingest_document
    doc_id = str(uuid.uuid4())
    mock_doc = make_doc(doc_id=doc_id)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    with patch("app.workers.tasks.SessionLocal", return_value=mock_db), \
         patch("app.workers.tasks.parse_document", side_effect=RuntimeError("parse error")):
        ingest_document.apply(args=[doc_id])

    assert mock_doc.status == "failed"
    assert "parse error" in (mock_doc.error_msg or "")

def test_ingest_document_fails_on_empty_extraction():
    from app.workers.tasks import ingest_document
    from app.services.ingestion import ParsedDocument, PageContent
    doc_id = str(uuid.uuid4())
    mock_doc = make_doc(doc_id=doc_id)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.first.return_value = mock_doc

    parsed = ParsedDocument(pages=[PageContent(page=1, text="", source="ocr")], metadata={})

    with patch("app.workers.tasks.SessionLocal", return_value=mock_db), \
         patch("app.workers.tasks.parse_document", return_value=parsed), \
         patch("app.workers.tasks.chunk_document", return_value=([], [])):
        ingest_document.apply(args=[doc_id])

    assert mock_doc.status == "failed"
    assert "No extractable text" in (mock_doc.error_msg or "")
