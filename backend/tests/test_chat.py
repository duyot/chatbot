import uuid


def test_chat_stream_404_for_unknown_document(client):
    response = client.post(
        "/api/chat/stream",
        json={"document_id": str(uuid.uuid4()), "message": "hello"},
    )
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_chat_stream_400_for_processing_document(client, db):
    from app.models import Document

    doc = Document(
        id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="processing"
    )
    db.add(doc)
    db.flush()

    response = client.post(
        "/api/chat/stream",
        json={"document_id": str(doc.id), "message": "hello"},
    )
    assert response.status_code == 400
    assert "not ready" in response.json()["detail"].lower()


def test_chat_stream_returns_event_stream_for_done_document(client, db, mocker):
    from app.models import Document

    doc = Document(
        id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done"
    )
    db.add(doc)
    db.flush()

    async def fake_rag(document_id, message, db):
        yield {"type": "token", "content": "Hello"}
        yield {"type": "citations", "chunks": []}
        yield {"type": "done"}

    mocker.patch("app.routers.chat.agentic_rag_stream", side_effect=fake_rag)

    response = client.post(
        "/api/chat/stream",
        json={"document_id": str(doc.id), "message": "hello"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert '"type": "token"' in response.text
    assert '"type": "done"' in response.text
