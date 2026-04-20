import io
import pytest

def test_upload_pdf_returns_pending(client, mocker):
    mocker.patch("app.workers.tasks.ingest_document.delay")
    fake_pdf = io.BytesIO(b"%PDF-1.4 fake content")
    response = client.post(
        "/api/documents/upload",
        files={"file": ("report.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["file_name"] == "report.pdf"
    assert data["status"] == "pending"
    assert "id" in data

def test_upload_rejects_unsupported_type(client):
    response = client.post(
        "/api/documents/upload",
        files={"file": ("script.py", b"print('hi')", "text/x-python")},
    )
    assert response.status_code == 400
    assert "Unsupported" in response.json()["detail"]

def test_upload_rejects_oversized_file(client, mocker):
    mocker.patch("app.config.settings.max_upload_mb", 0)
    fake_pdf = io.BytesIO(b"%PDF-1.4 " + b"x" * 1025)
    response = client.post(
        "/api/documents/upload",
        files={"file": ("big.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 400

def test_status_stream_returns_done(client, db):
    from app.models import Document
    import uuid

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="test.pdf", file_path="/tmp/test.pdf", status="done")
    db.add(doc)
    db.flush()

    response = client.get(f"/api/documents/{doc_id}/status")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"status": "done"' in response.text

def test_status_stream_404_for_unknown_id(client):
    import uuid
    response = client.get(f"/api/documents/{uuid.uuid4()}/status")
    assert response.status_code == 404
