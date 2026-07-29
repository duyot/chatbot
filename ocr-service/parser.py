"""Docling adapter — the only module in this service that imports Docling.

Converts document bytes into the `wire.to_wire()` body. All normalization logic
lives in `wire.py`; this file is purely extraction and coordinate conversion.
"""
from __future__ import annotations

import glob
import logging
import os
import tempfile
from typing import List, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
import rapidocr

from wire import RawElement, RawPage, to_wire

logger = logging.getLogger("ocr-service.parser")


class ParseError(RuntimeError):
    """Raised when Docling cannot convert the input at all."""


_converter: Optional[DocumentConverter] = None


def _find_bundled_rapidocr_model(prefix: str) -> Optional[str]:
    """Path to a model file bundled inside the installed `rapidocr` package.

    Needed because Docling's `RapidOcrModel`, when given a global
    `artifacts_path` (set below for the layout/TableFormer models baked by
    `docling-tools models download`), resolves *every* model's path —
    including RapidOCR's — relative to that same `artifacts_path`. RapidOCR's
    own weights were never baked there; they ship inside the `rapidocr`
    package itself. Pointing `RapidOcrOptions` at the bundled files directly
    bypasses that resolution and keeps the OCR engine fully offline.
    """
    models_dir = os.path.join(os.path.dirname(rapidocr.__file__), "models")
    matches = glob.glob(os.path.join(models_dir, prefix + "*"))
    return matches[0] if matches else None


def _get_converter() -> DocumentConverter:
    """Built once per process. Construction loads model weights, so it must not
    happen per request.

    `artifacts_path` points at the DocLayNet/TableFormer weights baked into
    the image by `docling-tools models download` (Task 1) — without it,
    Docling's default `DocumentConverter()` always tries to fetch those
    weights from HuggingFace Hub on first use, even though they are already
    on disk, which breaks the offline container. `generate_parsed_pages`
    keeps each page's native text cells around after conversion; without it
    `page.cells` is always empty (cleared post-assembly) and every page
    would be misreported as OCR'd even when it had a native text layer.
    """
    global _converter
    if _converter is None:
        ocr_options = RapidOcrOptions(
            det_model_path=_find_bundled_rapidocr_model("PP-OCRv6_det"),
            cls_model_path=_find_bundled_rapidocr_model("ch_ppocr_mobile_v2.0_cls"),
            rec_model_path=_find_bundled_rapidocr_model("PP-OCRv6_rec"),
        )
        pipeline_options = PdfPipelineOptions()
        pipeline_options.artifacts_path = settings.cache_dir / "models"
        pipeline_options.ocr_options = ocr_options
        pipeline_options.generate_parsed_pages = True
        _converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
            }
        )
    return _converter


def _page_dims(dl_doc) -> dict:
    """{page_no: (width, height)} from the Docling document."""
    dims = {}
    for page_no, page in (getattr(dl_doc, "pages", {}) or {}).items():
        size = getattr(page, "size", None)
        if size is not None:
            dims[int(page_no)] = (float(size.width), float(size.height))
    return dims


def _native_cell_counts(result) -> dict:
    """{page_no: cell_count} from the conversion result's pipeline-internal
    pages (`result.pages`), NOT `result.document.pages`.

    These are two different objects: `result.document.pages` (dict[int,
    PageItem]) only carries page size/image and has no `cells` attribute at
    all. `result.pages` (list[Page], the pipeline's internal per-page model)
    is where `.cells` lives, as a view over `.parsed_page.textline_cells` —
    which is only populated when `generate_parsed_pages=True` (see
    `_get_converter`). This is what distinguishes a native text page from one
    that needed OCR.
    """
    counts = {}
    for page in getattr(result, "pages", None) or []:
        counts[int(page.page_no)] = len(getattr(page, "cells", None) or [])
    return counts


def _raw_pages(dl_doc, dims: dict, native_cell_counts: dict) -> List[RawPage]:
    """One RawPage per page. `source` is "ocr" when that page carried no native
    text cells, which is how we report ocr_pages/native_pages upstream."""
    pages = []
    for page_no in sorted(dims):
        width, height = dims[page_no]
        has_native_cells = native_cell_counts.get(page_no, 0) > 0
        pages.append(RawPage(
            page=page_no,
            width=width,
            height=height,
            source="native" if has_native_cells else "ocr",
            ocr_confidence=None,
        ))
    return pages


def _element_text(item, dl_doc) -> str:
    """Markdown for tables, plain text for everything else."""
    exporter = getattr(item, "export_to_markdown", None)
    if exporter is not None:
        try:
            return exporter(dl_doc)
        except TypeError:
            # Older Docling versions take no argument.
            return exporter()
    return getattr(item, "text", "") or ""


def _bbox_top_left(prov, page_height: float) -> Optional[tuple]:
    """Docling bboxes carry their own `coord_origin` (bottom-left for PDF/OCR
    provenance in practice, but the field is not hardcoded) — flip only if
    needed rather than assuming bottom-left unconditionally. `BoundingBox`
    ships exactly this conversion as `to_top_left_origin`, which is a no-op
    when the box is already top-left; the wire format and the page images
    the UI overlays both use top-left."""
    bbox = getattr(prov, "bbox", None)
    if bbox is None or not page_height:
        return None
    top_left = bbox.to_top_left_origin(page_height)
    return (float(top_left.l), float(top_left.t), float(top_left.r), float(top_left.b))


def _label_of(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


def _heading_level(item) -> Optional[int]:
    """Semantic heading level, e.g. h1 vs h2 — distinct from the tree-depth
    `level` `iterate_items()` yields alongside each item. Only
    `SectionHeaderItem` carries this; `TitleItem` has no `.level` attribute
    at all and is always the top-level heading, so `None` here (which
    `to_wire()` defaults to 1) is correct for it."""
    return getattr(item, "level", None)


def parse_bytes(data: bytes, *, filename: str) -> dict:
    """Convert document bytes into the /parse wire body.

    Raises ParseError when Docling cannot convert the input; the caller maps
    that to HTTP 422.
    """
    suffix = os.path.splitext(filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        try:
            result = _get_converter().convert(tmp_path)
        except Exception as exc:
            logger.warning("parse_bytes: convert failed file=%s err=%s", filename, exc)
            raise ParseError(str(exc)) from exc

        dl_doc = result.document
        dims = _page_dims(dl_doc)
        native_cell_counts = _native_cell_counts(result)
        pages = _raw_pages(dl_doc, dims, native_cell_counts)

        elements: List[RawElement] = []
        for item, _level in dl_doc.iterate_items():
            prov = (getattr(item, "prov", None) or [None])[0]
            if prov is None:
                continue
            page_no = int(prov.page_no)
            _, page_height = dims.get(page_no, (0.0, 0.0))
            text = _element_text(item, dl_doc)
            if not text.strip():
                continue
            elements.append(RawElement(
                label=_label_of(item),
                page=page_no,
                text=text,
                bbox_abs=_bbox_top_left(prov, page_height),
                level=_heading_level(item),
                confidence=None,
            ))

        body = to_wire(elements, pages, {"mime_type": None})
        logger.info(
            "parse_bytes: file=%s pages=%d elements=%d",
            filename, len(pages), len(elements),
        )
        return body
    finally:
        os.unlink(tmp_path)
