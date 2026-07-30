import threading
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import app as app_module
    return TestClient(app_module.app), app_module


def test_parse_returns_wire_body(client, mocker):
    c, app_module = client
    mocker.patch.object(app_module, "parse_bytes", return_value={
        "schema_version": 1,
        "metadata": {"page_count": 1},
        "pages": [],
        "elements": [{"id": "e0", "page": 1, "type": "paragraph",
                      "text": "hi", "bbox": None, "confidence": None}],
    })
    resp = c.post("/parse", files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")})

    assert resp.status_code == 200
    assert resp.json()["schema_version"] == 1
    assert resp.json()["elements"][0]["text"] == "hi"


def test_parse_returns_422_on_parse_error(client, mocker):
    c, app_module = client
    mocker.patch.object(app_module, "parse_bytes",
                        side_effect=app_module.ParseError("not a document"))
    resp = c.post("/parse", files={"file": ("broken.pdf", b"garbage", "application/pdf")})

    assert resp.status_code == 422
    assert "not a document" in resp.json()["detail"]


def test_parse_passes_filename_through(client, mocker):
    c, app_module = client
    patched = mocker.patch.object(app_module, "parse_bytes", return_value={
        "schema_version": 1, "metadata": {}, "pages": [], "elements": [],
    })
    c.post("/parse", files={"file": ("report.docx", b"PK\x03\x04", None)})

    assert patched.call_args.args[0] == b"PK\x03\x04"
    assert patched.call_args.kwargs["filename"] == "report.docx"


def test_health_stays_answerable_while_parse_is_in_flight(client, mocker):
    """FIX 2: /parse must not block the event loop.

    parse_bytes is patched to block on an Event so a call to /parse is
    guaranteed to still be running when /health is hit. If /parse ran
    directly on the event loop (the pre-fix `async def parse` calling a
    synchronous parse_bytes), this /health call would queue behind it and
    the assertion below would time out instead of returning quickly.
    """
    c, app_module = client
    started = threading.Event()
    release = threading.Event()

    def slow_parse_bytes(data, *, filename):
        started.set()
        release.wait(timeout=10)
        return {"schema_version": 1, "metadata": {}, "pages": [], "elements": []}

    mocker.patch.object(app_module, "parse_bytes", side_effect=slow_parse_bytes)

    parse_thread = threading.Thread(
        target=lambda: c.post("/parse", files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")})
    )
    parse_thread.start()
    try:
        assert started.wait(timeout=5), "parse_bytes never started"

        t0 = time.monotonic()
        resp = c.get("/health")
        elapsed = time.monotonic() - t0

        assert resp.status_code == 200
        assert elapsed < 2.0, (
            f"/health took {elapsed:.2f}s while a parse was in flight — "
            "the event loop is blocked"
        )
    finally:
        release.set()
        parse_thread.join(timeout=10)


def test_ocr_endpoint_still_exists(client):
    """The legacy path must survive — it is the documented rollback."""
    _, app_module = client
    assert any(getattr(r, "path", None) == "/ocr" for r in app_module.app.routes)
