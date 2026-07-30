"""Exercises /ocr's handler body against rapidocr v3's real return shape.

This is the gap that let the v3 upgrade regress /ocr: v3's `RapidOCR.__call__`
returns a `RapidOCROutput` dataclass, not the old `(result, elapsed)` tuple of
`[box, text, score]` triples. `result, _elapse = _ocr(raw)` raised at runtime
because `RapidOCROutput` isn't iterable/unpackable at all, and nothing invoked
the handler body to catch it.

Run the slow (real inference) test explicitly inside the built image:
    docker run --rm --network none ocr-spike python -m pytest tests/test_ocr_endpoint.py -m slow -v
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw


@pytest.fixture
def client():
    import app as app_module
    return TestClient(app_module.app), app_module


@pytest.mark.slow
def test_ocr_endpoint_extracts_lines_from_a_rendered_image(client):
    """Real RapidOCR v3 inference through the actual /ocr route.

    A mock can only be as correct as the assumption it encodes — the original
    bug was exactly a wrong assumption about the return shape, so a mock
    would have reproduced it. This runs the real engine end-to-end and checks
    the wire contract that backend/app/services/ocr_client.py depends on.
    """
    c, _ = client
    img = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), "QUARTERLY REPORT", fill="black")
    draw.text((40, 160), "Revenue increased in the APAC region.", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    resp = c.post("/ocr", files={"file": ("scan.png", buf.getvalue(), "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    assert body["width"] == 900
    assert body["height"] == 300
    assert body["lines"], "expected at least one detected line of text"
    for line in body["lines"]:
        assert isinstance(line["text"], str) and line["text"]
        assert isinstance(line["confidence"], float)
        assert 0.0 <= line["confidence"] <= 1.0
        assert len(line["bbox"]) == 4
        for point in line["bbox"]:
            assert len(point) == 2
            assert isinstance(point[0], float) and isinstance(point[1], float)
    # bbox coordinates are image-pixel, not normalized (the backend
    # normalizes them) — text near the top of a 300px-tall image should
    # report y well below 300, not below 1.0.
    assert any(pt[1] > 1.0 for line in body["lines"] for pt in line["bbox"])


@pytest.mark.slow
def test_ocr_endpoint_handles_no_detections(client):
    """A blank image is a real, valid input that must not 500."""
    c, _ = client
    img = Image.new("RGB", (200, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    resp = c.post("/ocr", files={"file": ("blank.png", buf.getvalue(), "image/png")})

    assert resp.status_code == 200
    body = resp.json()
    assert body == {"lines": [], "width": 200, "height": 100}


def test_ocr_endpoint_shapes_response_from_rapidocr_output(client, mocker):
    """Fast, mocked check of the response-shaping logic only.

    Mocks at the `_ocr(raw)` call boundary but returns the *real*
    `RapidOCROutput` dataclass (imported from rapidocr itself, not
    hand-rolled) populated with fixture data. If a future rapidocr release
    renames `.boxes`/`.txts`/`.scores` or changes their types, constructing
    this real dataclass — or the handler's attribute access on it — would
    fail, so this still would have caught the field-name half of the
    original bug. It would NOT have caught the tuple-unpacking half (that
    needs the slow real-inference test above, which calls `_ocr` for real).
    """
    import numpy as np
    from rapidocr.utils.output import RapidOCROutput

    c, app_module = client
    fake_result = RapidOCROutput(
        boxes=np.array([[[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]]),
        txts=("hello",),
        scores=(0.987,),
    )
    mocker.patch.object(app_module, "_ocr", return_value=fake_result)

    img = Image.new("RGB", (64, 32), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    resp = c.post("/ocr", files={"file": ("x.png", buf.getvalue(), "image/png")})

    assert resp.status_code == 200
    assert resp.json() == {
        "lines": [{
            "text": "hello",
            "bbox": [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]],
            "confidence": 0.987,
        }],
        "width": 64,
        "height": 32,
    }


def test_ocr_endpoint_handles_none_result_fields(client, mocker):
    """Mocked no-detections case: boxes/txts/scores are all None."""
    from rapidocr.utils.output import RapidOCROutput

    c, app_module = client
    mocker.patch.object(app_module, "_ocr", return_value=RapidOCROutput())

    img = Image.new("RGB", (64, 32), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    resp = c.post("/ocr", files={"file": ("x.png", buf.getvalue(), "image/png")})

    assert resp.status_code == 200
    assert resp.json() == {"lines": [], "width": 64, "height": 32}
