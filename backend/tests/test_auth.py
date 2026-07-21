from app.models import User
from app.security import hash_password


def _make_user(db, username="alice", password="secret123", is_active=True):
    user = User(
        username=username,
        password_hash=hash_password(password),
        is_active=is_active,
    )
    db.add(user)
    db.commit()
    return user


def test_login_success_returns_token(client, db):
    _make_user(db)
    res = client.post(
        "/api/auth/login", json={"username": "alice", "password": "secret123"}
    )
    assert res.status_code == 200
    body = res.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_wrong_password_401(client, db):
    _make_user(db)
    res = client.post(
        "/api/auth/login", json={"username": "alice", "password": "wrong"}
    )
    assert res.status_code == 401


def test_login_unknown_user_401(client):
    res = client.post(
        "/api/auth/login", json={"username": "ghost", "password": "x"}
    )
    assert res.status_code == 401


def test_me_requires_token(client):
    res = client.get("/api/auth/me")
    assert res.status_code == 401


def test_me_with_token_returns_user(client, db):
    _make_user(db)
    login = client.post(
        "/api/auth/login", json={"username": "alice", "password": "secret123"}
    )
    token = login.json()["access_token"]
    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    assert res.json()["username"] == "alice"


def test_protected_chat_requires_token(client):
    res = client.post(
        "/api/chat/stream", json={"document_id": "x", "message": "hi"}
    )
    assert res.status_code == 401


def test_protected_documents_list_requires_token(client):
    res = client.get("/api/documents")
    assert res.status_code == 401
