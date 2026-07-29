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


def test_ocr_endpoint_still_exists(client):
    """The legacy path must survive — it is the documented rollback."""
    _, app_module = client
    assert any(getattr(r, "path", None) == "/ocr" for r in app_module.app.routes)
