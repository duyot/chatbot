import io
import pytest

def test_upload_pdf_returns_pending(client, mocker):
    mocker.patch("app.routers.documents.ingest_document.delay")
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
