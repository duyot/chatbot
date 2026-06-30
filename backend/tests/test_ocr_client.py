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
