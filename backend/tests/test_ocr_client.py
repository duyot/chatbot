import pytest
from unittest.mock import MagicMock


def _mock_httpx_client(mocker, body, raises=None):
    """Patch httpx.Client used in the ocr_client module (mirrors the reranker
    test helper). If `raises` is set, the .post call raises it."""
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = body
    fake_client = MagicMock()
    if raises is not None:
        fake_client.post.side_effect = raises
    else:
        fake_client.post.return_value = fake_response
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_client
    fake_cm.__exit__.return_value = None
    return mocker.patch("app.services.ocr_client.httpx.Client", return_value=fake_cm)


def test_ocr_image_joins_lines_and_averages_confidence(mocker):
    from app.services.ocr_client import ocr_image
    patched = _mock_httpx_client(mocker, {
        "lines": [
            {"text": "Hello world", "confidence": 0.9},
            {"text": "second line", "confidence": 0.8},
            {"text": "   ", "confidence": 0.5},  # blank -> excluded from text and avg
        ]
    })
    text, conf = ocr_image(b"\x89PNGfake", filename="scan.png")

    assert text == "Hello world\nsecond line"
    assert abs(conf - 0.85) < 1e-6  # (0.9 + 0.8) / 2; blank line excluded

    posted = patched.return_value.__enter__.return_value.post.call_args
    assert posted.args[0].endswith("/ocr")
    assert "files" in posted.kwargs


def test_ocr_image_empty_lines_returns_empty_and_none(mocker):
    from app.services.ocr_client import ocr_image
    _mock_httpx_client(mocker, {"lines": []})
    text, conf = ocr_image(b"x", filename="blank.png")
    assert text == ""
    assert conf is None


def test_ocr_image_raises_ocr_error_on_http_failure(mocker):
    from app.services.ocr_client import ocr_image, OCRError
    _mock_httpx_client(mocker, body=None, raises=RuntimeError("ocr service down"))
    with pytest.raises(OCRError):
        ocr_image(b"x", filename="scan.png")


def _wire_body(**overrides):
    body = {
        "schema_version": 1,
        "metadata": {"page_count": 1, "ocr_pages": 1, "native_pages": 0},
        "pages": [{"page": 1, "width": 600.0, "height": 800.0,
                   "source": "ocr", "ocr_confidence": 0.9}],
        "elements": [{"id": "e0", "page": 1, "type": "paragraph",
                      "text": "hello", "bbox": [0.1, 0.1, 0.5, 0.2],
                      "confidence": 0.9}],
    }
    body.update(overrides)
    return body


def test_parse_document_remote_returns_body(mocker):
    from app.services.ocr_client import parse_document_remote
    patched = _mock_httpx_client(mocker, _wire_body())

    body = parse_document_remote(b"%PDF-fake", filename="doc.pdf")

    assert body["elements"][0]["text"] == "hello"
    posted = patched.return_value.__enter__.return_value.post.call_args
    assert posted.args[0].endswith("/parse")
    assert "files" in posted.kwargs


def test_parse_document_remote_rejects_unknown_schema_version(mocker):
    from app.services.ocr_client import parse_document_remote, OCRError
    _mock_httpx_client(mocker, _wire_body(schema_version=2))

    with pytest.raises(OCRError, match="schema_version"):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_rejects_non_dict_body(mocker):
    from app.services.ocr_client import parse_document_remote, OCRError
    _mock_httpx_client(mocker, ["not", "a", "dict"])

    with pytest.raises(OCRError):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_raises_ocr_error_on_http_failure(mocker):
    from app.services.ocr_client import parse_document_remote, OCRError
    _mock_httpx_client(mocker, body=None, raises=RuntimeError("service down"))

    with pytest.raises(OCRError):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_raises_parse_timeout_on_timeout(mocker):
    """Timeout gets its own type so the worker can make it non-retryable."""
    import httpx
    from app.services.ocr_client import parse_document_remote, ParseTimeout
    _mock_httpx_client(mocker, body=None, raises=httpx.ReadTimeout("too slow"))

    with pytest.raises(ParseTimeout):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_uses_the_long_parse_timeout(mocker):
    from app.config import settings
    from app.services.ocr_client import parse_document_remote
    patched = _mock_httpx_client(mocker, _wire_body())

    parse_document_remote(b"x", filename="doc.pdf")

    assert patched.call_args.kwargs["timeout"] == settings.parse_timeout_s
