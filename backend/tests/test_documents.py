import uuid

import fitz
import pytest

from app.config import settings
from app.models import Document, User
from app.security import hash_password
from app.services import page_images


def _auth_headers(client, db, username="alice", password="secret123"):
    db.add(User(username=username, password_hash=hash_password(password), is_active=True))
    db.commit()
    login = client.post("/api/auth/login", json={"username": username, "password": password})
    token = login.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_file_endpoint_requires_token(client):
    response = client.get(f"/api/documents/{uuid.uuid4()}/file")
    assert response.status_code == 401


def test_file_endpoint_404_for_unknown_document(client, db):
    headers = _auth_headers(client, db)
    response = client.get(f"/api/documents/{uuid.uuid4()}/file", headers=headers)
    assert response.status_code == 404


def test_file_endpoint_404_when_file_missing_on_disk(client, db):
    headers = _auth_headers(client, db)
    doc = Document(
        id=uuid.uuid4(),
        file_name="ghost.pdf",
        file_path="/tmp/does-not-exist-anywhere.pdf",
        status="done",
        mime_type="application/pdf",
    )
    db.add(doc)
    db.flush()

    response = client.get(f"/api/documents/{doc.id}/file", headers=headers)
    assert response.status_code == 404


def test_file_endpoint_returns_content_with_mime_type(client, db, tmp_path):
    headers = _auth_headers(client, db)
    file_path = tmp_path / "report.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake content")

    doc = Document(
        id=uuid.uuid4(),
        file_name="report.pdf",
        file_path=str(file_path),
        status="done",
        mime_type="application/pdf",
    )
    db.add(doc)
    db.flush()

    response = client.get(f"/api/documents/{doc.id}/file", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/pdf")
    assert "report.pdf" in response.headers["content-disposition"]
    assert response.content == b"%PDF-1.4 fake content"


def test_list_documents_includes_mime_type(client, db):
    headers = _auth_headers(client, db)
    doc = Document(
        id=uuid.uuid4(),
        file_name="done.pdf",
        file_path="/tmp/d.pdf",
        status="done",
        mime_type="application/pdf",
    )
    db.add(doc)
    db.flush()

    response = client.get("/api/documents", headers=headers)
    assert response.status_code == 200
    items = response.json()
    assert items[0]["mime_type"] == "application/pdf"


# --- Preview page images ----------------------------------------------------

@pytest.fixture
def upload_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    return tmp_path


def _pdf_document(db, tmp_path, pages=2, name="report.pdf"):
    path = tmp_path / name
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 100), f"Page {i + 1}")
    doc.save(str(path))
    doc.close()

    row = Document(
        id=uuid.uuid4(),
        file_name=name,
        file_path=str(path),
        status="done",
        mime_type="application/pdf",
        page_count=pages,
    )
    db.add(row)
    db.flush()
    return row


def test_pages_endpoint_requires_token(client):
    assert client.get(f"/api/documents/{uuid.uuid4()}/pages").status_code == 401
    assert client.get(f"/api/documents/{uuid.uuid4()}/pages/1").status_code == 401


def test_pages_endpoint_404_for_unknown_document(client, db):
    headers = _auth_headers(client, db)
    assert client.get(f"/api/documents/{uuid.uuid4()}/pages", headers=headers).status_code == 404
    assert client.get(f"/api/documents/{uuid.uuid4()}/pages/1", headers=headers).status_code == 404


def test_pages_manifest_renders_on_first_request(client, db, tmp_path, upload_dir):
    """A document whose pages were never rendered (i.e. ingested before this
    feature existed) is backfilled by the manifest request itself."""
    headers = _auth_headers(client, db)
    doc = _pdf_document(db, tmp_path, pages=3)
    assert page_images.list_page_images(doc.id) == []

    response = client.get(f"/api/documents/{doc.id}/pages", headers=headers)

    assert response.status_code == 200
    items = response.json()
    assert [item["page"] for item in items] == [1, 2, 3]
    for item in items:
        assert item["width"] > 0 and item["height"] > 0
    assert len(page_images.list_page_images(doc.id)) == 3


def test_pages_manifest_empty_for_non_pdf(client, db, tmp_path, upload_dir):
    headers = _auth_headers(client, db)
    path = tmp_path / "scan.png"
    path.write_bytes(b"not really a png")
    doc = Document(
        id=uuid.uuid4(),
        file_name="scan.png",
        file_path=str(path),
        status="done",
        mime_type="image/png",
    )
    db.add(doc)
    db.flush()

    response = client.get(f"/api/documents/{doc.id}/pages", headers=headers)

    assert response.status_code == 200
    assert response.json() == []


def test_pages_manifest_empty_when_source_file_is_gone(client, db, upload_dir):
    headers = _auth_headers(client, db)
    doc = Document(
        id=uuid.uuid4(),
        file_name="ghost.pdf",
        file_path="/tmp/does-not-exist-anywhere.pdf",
        status="done",
        mime_type="application/pdf",
    )
    db.add(doc)
    db.flush()

    response = client.get(f"/api/documents/{doc.id}/pages", headers=headers)

    assert response.status_code == 200
    assert response.json() == []


def test_page_image_returns_bytes_with_media_type(client, db, tmp_path, upload_dir):
    headers = _auth_headers(client, db)
    doc = _pdf_document(db, tmp_path, pages=2)
    client.get(f"/api/documents/{doc.id}/pages", headers=headers)  # render

    response = client.get(f"/api/documents/{doc.id}/pages/2", headers=headers)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/webp"
    # No Content-Disposition: these bytes are consumed by fetch(), not downloaded.
    assert "content-disposition" not in response.headers
    assert "max-age" in response.headers["cache-control"]
    assert response.content[:4] == b"RIFF"  # webp container magic


@pytest.mark.parametrize("page", [0, 99])
def test_page_image_404_for_out_of_range_page(client, db, tmp_path, upload_dir, page):
    headers = _auth_headers(client, db)
    doc = _pdf_document(db, tmp_path, pages=2)
    client.get(f"/api/documents/{doc.id}/pages", headers=headers)  # render

    response = client.get(f"/api/documents/{doc.id}/pages/{page}", headers=headers)

    assert response.status_code == 404


def test_page_image_rejects_non_numeric_page(client, db, tmp_path, upload_dir):
    headers = _auth_headers(client, db)
    doc = _pdf_document(db, tmp_path, pages=1)

    response = client.get(f"/api/documents/{doc.id}/pages/..%2F..%2Fsecret", headers=headers)

    assert response.status_code in (404, 422)
