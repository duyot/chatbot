import uuid

from app.models import Document, User
from app.security import hash_password


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
