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


@pytest.mark.slow
def test_parse_bytes_bbox_is_not_vertically_mirrored():
    """Guards the bottom-left -> top-left bbox flip specifically.

    `0.0 <= v <= 1.0` (the assertion above) can't catch a mirrored bbox — a
    flip bug still produces in-range numbers, just for the wrong half of the
    page. These rects get overlaid on page images in the UI, where a mirror
    would be silently wrong on screen but invisible to that assertion.

    The image has text confined to a single line near the top, with the
    entire lower half blank, so there is only one place a genuine detection
    can land: a correct top-left-origin bbox must have y0 (and y1) in the
    upper half of the page. A bottom-left origin left unconverted (or a
    flip applied twice) would report this same text in the lower half
    instead.
    """
    from parser import parse_bytes

    img = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(img)
    draw.text((40, 40), "TOP OF PAGE ONLY", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    body = parse_bytes(buf.getvalue(), filename="top_text.png")

    elements_with_bbox = [el for el in body["elements"] if el["bbox"] is not None]
    assert elements_with_bbox, "expected at least one element with a bbox"
    for el in elements_with_bbox:
        y0, y1 = el["bbox"][1], el["bbox"][3]
        assert y0 < 0.5 and y1 < 0.5, (
            f"element {el['text']!r} bbox {el['bbox']} lands in the lower half of "
            "the page; the top-of-page text should never normalize below y=0.5 "
            "unless the bottom-left/top-left flip is wrong"
        )
