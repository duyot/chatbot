"""The `/parse` wire contract.

Deliberately free of any Docling import: this module holds every piece of logic
that can be wrong, so it is unit-testable without downloading model weights.
`parser.py` is the only place Docling is touched.

Consumer: backend/app/services/ocr_client.parse_document_remote.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

SCHEMA_VERSION = 1

# Docling item labels -> our element types. Anything absent degrades to
# DEFAULT_TYPE so a Docling upgrade that adds a label cannot break the service.
TYPE_BY_LABEL = {
    "title": "heading",
    "section_header": "heading",
    "text": "paragraph",
    "paragraph": "paragraph",
    "formula": "paragraph",
    "code": "paragraph",
    "list_item": "list_item",
    "table": "table",
    "caption": "caption",
    "picture": "figure",
    "page_header": "page_header",
    "page_footer": "page_footer",
}
DEFAULT_TYPE = "paragraph"


@dataclass(frozen=True)
class RawElement:
    """One document element as extracted from Docling, before normalization.

    `bbox_abs` is (x0, y0, x1, y1) in that page's units with a **top-left
    origin** — convert from Docling's bottom-left origin before constructing.
    """
    label: str
    page: int
    text: str
    bbox_abs: Optional[tuple]
    level: Optional[int] = None
    confidence: Optional[float] = None


@dataclass(frozen=True)
class RawPage:
    page: int
    width: float
    height: float
    source: str                        # "native" | "ocr"
    ocr_confidence: Optional[float]


def element_type(label: str) -> str:
    return TYPE_BY_LABEL.get((label or "").strip().lower(), DEFAULT_TYPE)


def _clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def normalize_bbox(
    bbox_abs: Optional[tuple], width: float, height: float
) -> Optional[list]:
    """Absolute page-unit rect -> normalized [x0, y0, x1, y1] in [0,1], clamped.

    Returns None when there is no bbox or the page dimensions are unusable, so
    a missing box is explicit rather than silently (0,0,0,0).
    """
    if not bbox_abs or not width or not height:
        return None
    x0, y0, x1, y1 = bbox_abs
    return [
        _clamp01(min(x0, x1) / width), _clamp01(min(y0, y1) / height),
        _clamp01(max(x0, x1) / width), _clamp01(max(y0, y1) / height),
    ]


def to_wire(
    elements: Sequence[RawElement],
    pages: Sequence[RawPage],
    metadata: dict,
) -> dict:
    """Assemble the response body. Element order is preserved verbatim — it is
    the reading order the backend chunker depends on."""
    dims = {p.page: (p.width, p.height) for p in pages}

    out_elements = []
    for i, el in enumerate(elements):
        width, height = dims.get(el.page, (0.0, 0.0))
        etype = element_type(el.label)
        item = {
            "id": f"e{i}",
            "page": el.page,
            "type": etype,
            "text": el.text,
            "bbox": normalize_bbox(el.bbox_abs, width, height),
            "confidence": el.confidence,
        }
        if etype == "heading":
            item["level"] = el.level if el.level is not None else 1
        out_elements.append(item)

    ocr_pages = sum(1 for p in pages if p.source == "ocr")
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": {
            **metadata,
            "page_count": len(pages),
            "ocr_pages": ocr_pages,
            "native_pages": len(pages) - ocr_pages,
        },
        "pages": [
            {
                "page": p.page, "width": p.width, "height": p.height,
                "source": p.source, "ocr_confidence": p.ocr_confidence,
            }
            for p in pages
        ],
        "elements": out_elements,
    }
