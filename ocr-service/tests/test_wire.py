import pytest

from wire import (
    SCHEMA_VERSION,
    RawElement,
    RawPage,
    element_type,
    normalize_bbox,
    to_wire,
)


def _page(n=1, w=600.0, h=800.0):
    return RawPage(page=n, width=w, height=h, source="ocr", ocr_confidence=0.9)


# --- element_type -----------------------------------------------------------

@pytest.mark.parametrize("label,expected", [
    ("title", "heading"),
    ("section_header", "heading"),
    ("text", "paragraph"),
    ("list_item", "list_item"),
    ("table", "table"),
    ("caption", "caption"),
    ("picture", "figure"),
    ("page_header", "page_header"),
    ("page_footer", "page_footer"),
])
def test_element_type_maps_known_labels(label, expected):
    assert element_type(label) == expected


def test_element_type_is_case_insensitive():
    assert element_type("SECTION_HEADER") == "heading"


def test_element_type_unknown_label_degrades_to_paragraph():
    # A Docling upgrade that introduces a new label must not crash the service.
    assert element_type("some_future_label") == "paragraph"


# --- normalize_bbox ---------------------------------------------------------

def test_normalize_bbox_divides_by_page_dimensions():
    assert normalize_bbox((60.0, 80.0, 300.0, 400.0), 600.0, 800.0) == [0.1, 0.1, 0.5, 0.5]


def test_normalize_bbox_clamps_out_of_range_values():
    # Docling can emit a box marginally outside the page rect.
    assert normalize_bbox((-10.0, -10.0, 700.0, 900.0), 600.0, 800.0) == [0.0, 0.0, 1.0, 1.0]


def test_normalize_bbox_returns_none_without_bbox():
    assert normalize_bbox(None, 600.0, 800.0) is None


def test_normalize_bbox_returns_none_on_zero_page_dimensions():
    assert normalize_bbox((0.0, 0.0, 10.0, 10.0), 0.0, 800.0) is None


# --- to_wire ----------------------------------------------------------------

def test_to_wire_preserves_reading_order_and_assigns_sequential_ids():
    elements = [
        RawElement(label="title", page=1, text="First", bbox_abs=(0, 0, 10, 10), level=1),
        RawElement(label="text", page=1, text="Second", bbox_abs=(0, 20, 10, 30)),
        RawElement(label="text", page=1, text="Third", bbox_abs=(0, 40, 10, 50)),
    ]
    body = to_wire(elements, [_page()], {"mime_type": "application/pdf"})

    assert [e["id"] for e in body["elements"]] == ["e0", "e1", "e2"]
    assert [e["text"] for e in body["elements"]] == ["First", "Second", "Third"]


def test_to_wire_sets_schema_version_and_page_count():
    body = to_wire([], [_page(1), _page(2)], {"mime_type": "application/pdf"})
    assert body["schema_version"] == SCHEMA_VERSION
    assert body["metadata"]["page_count"] == 2


def test_to_wire_includes_level_on_headings_only():
    elements = [
        RawElement(label="section_header", page=1, text="H", bbox_abs=(0, 0, 1, 1), level=2),
        RawElement(label="text", page=1, text="P", bbox_abs=(0, 2, 1, 3), level=2),
    ]
    body = to_wire(elements, [_page()], {})

    heading, paragraph = body["elements"]
    assert heading["level"] == 2
    assert "level" not in paragraph


def test_to_wire_normalizes_bbox_against_the_elements_own_page():
    elements = [
        RawElement(label="text", page=2, text="on page two", bbox_abs=(50.0, 100.0, 150.0, 200.0)),
    ]
    pages = [_page(1, 600.0, 800.0), _page(2, 500.0, 1000.0)]
    body = to_wire(elements, pages, {})

    assert body["elements"][0]["bbox"] == [0.1, 0.1, 0.3, 0.2]


def test_to_wire_emits_null_bbox_for_element_on_unknown_page():
    elements = [RawElement(label="text", page=99, text="orphan", bbox_abs=(0, 0, 1, 1))]
    body = to_wire(elements, [_page(1)], {})
    assert body["elements"][0]["bbox"] is None


def test_to_wire_carries_page_source_and_counts():
    pages = [RawPage(page=1, width=600.0, height=800.0, source="native", ocr_confidence=None)]
    body = to_wire([], pages, {})

    assert body["pages"][0] == {
        "page": 1, "width": 600.0, "height": 800.0,
        "source": "native", "ocr_confidence": None,
    }
    assert body["metadata"]["native_pages"] == 1
    assert body["metadata"]["ocr_pages"] == 0


def test_to_wire_empty_document_is_valid():
    body = to_wire([], [], {"mime_type": "application/pdf"})
    assert body["elements"] == []
    assert body["pages"] == []
    assert body["metadata"]["page_count"] == 0
