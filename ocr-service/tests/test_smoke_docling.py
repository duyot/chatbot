"""One real Docling inference run. Excluded from default runs by pytest.ini
(`-m "not slow"`) because it loads model weights.

Run explicitly inside the built image:
    docker run --rm --network none ocr-spike python -m pytest tests/test_smoke_docling.py -m slow -v
"""
import io

import pytest
from PIL import Image, ImageDraw


@pytest.mark.slow
def test_parse_bytes_extracts_text_from_a_rendered_image():
    from parser import parse_bytes

    img = Image.new("RGB", (900, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 60), "QUARTERLY REPORT", fill="black")
    draw.text((40, 160), "Revenue increased in the APAC region.", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    body = parse_bytes(buf.getvalue(), filename="scan.png")

    assert body["schema_version"] == 1
    assert body["metadata"]["page_count"] >= 1
    assert body["elements"], "expected at least one element from the rendered text"
    for el in body["elements"]:
        assert el["type"] in {
            "heading", "paragraph", "list_item", "table",
            "caption", "figure", "page_header", "page_footer",
        }
        if el["bbox"] is not None:
            assert all(0.0 <= v <= 1.0 for v in el["bbox"])
