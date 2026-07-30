"""Rasterize PDF pages to images for the document preview.

The UI used to parse the PDF in the browser with pdf.js and paint every page to
a canvas. That cost ~1MB of JS plus main-thread rasterization on every preview.
Instead each page is rendered once — at ingestion, or lazily on first preview
for documents that predate this module — into

    {upload_dir}/pages/{document_id}/{page:04d}.{ext}

and served to the browser as a plain image.

Citation geometry keeps working for free: `DocumentChunk.bbox` rects are
normalized to [0,1] of the *source page rect* (see ingestion._native_pdf_lines),
and a rendered page is geometrically similar to that rect at any DPI, so the
same rects overlay the image unchanged.

Only PDFs get page images. Single-image uploads are already previewable as-is,
and DOCX has no page geometry to rasterize.
"""
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

from ..config import settings

logger = logging.getLogger(__name__)

_PDF_EXT = ".pdf"

# Formats we can encode. webp/jpg are lossy; png is lossless but 3-5x larger.
_MEDIA_TYPES = {
    "webp": "image/webp",
    "jpg": "image/jpeg",
    "png": "image/png",
}


@dataclass(frozen=True)
class PageImage:
    """One rendered page. `width`/`height` are the image's pixel dimensions,
    which the UI uses to reserve correctly-shaped space before the bytes
    arrive — so scroll offsets (and citation scroll-to-page) are right on the
    first paint."""
    page: int          # 1-based
    path: str
    width: int
    height: int


# --- Paths ------------------------------------------------------------------

def image_format() -> str:
    """Configured output format, normalized. Falls back to webp on a bad value
    rather than failing ingestion over a typo in .env."""
    fmt = (settings.page_image_format or "webp").lower().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in _MEDIA_TYPES:
        logger.warning("page_images: unsupported page_image_format=%r, using webp", fmt)
        return "webp"
    return fmt


def media_type() -> str:
    return _MEDIA_TYPES[image_format()]


def pages_dir(document_id) -> str:
    return os.path.join(settings.upload_dir, "pages", str(document_id))


def page_image_path(document_id, page: int) -> str:
    return os.path.join(pages_dir(document_id), f"{page:04d}.{image_format()}")


# --- Encoding ---------------------------------------------------------------

def _read_size(path: str) -> Optional[Tuple[int, int]]:
    """(width, height) of an already-rendered image, reading only its header.
    Returns None when the file is absent or corrupt; callers then re-render,
    which makes a half-written image self-healing."""
    if not os.path.exists(path):
        return None
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        logger.warning("page_images: unreadable image, will re-render path=%s", path)
        return None


def _write_pixmap(pix, path: str, fmt: str) -> None:
    """Encode a pixmap to `path` via a temp file + os.replace, so a crash
    mid-write cannot leave a truncated image that a later run would mistake for
    an already-rendered page."""
    tmp = f"{path}.tmp"
    try:
        if fmt == "webp":
            # PyMuPDF has no webp encoder; pil_save routes the pixmap through Pillow.
            pix.pil_save(tmp, format="WEBP", quality=settings.page_image_quality, method=4)
        elif fmt == "jpg":
            pix.save(tmp, output="jpg", jpg_quality=settings.page_image_quality)
        else:
            pix.save(tmp, output="png")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# --- Rendering --------------------------------------------------------------

def render_document_pages(file_path: str, document_id) -> List[PageImage]:
    """Render every page of a PDF that isn't on disk yet, and return all of its
    page images in order.

    Idempotent: pages already rendered are reused, so calling this again is
    cheap and the lazy-backfill path costs one `fitz.open`. A page that fails to
    render is skipped with a warning — one bad page must not cost the rest of
    the document. Non-PDFs and a disabled feature flag return [].
    """
    if not settings.page_images_enabled:
        return []
    if os.path.splitext(file_path)[1].lower() != _PDF_EXT:
        return []
    if not os.path.exists(file_path):
        logger.warning("render_document_pages: source missing document_id=%s path=%s",
                       document_id, file_path)
        return []

    fmt = image_format()
    os.makedirs(pages_dir(document_id), exist_ok=True)

    images: List[PageImage] = []
    rendered = reused = failed = 0
    with fitz.open(file_path) as doc:
        for i, page in enumerate(doc):
            n = i + 1
            path = page_image_path(document_id, n)

            size = _read_size(path)
            if size:
                images.append(PageImage(page=n, path=path, width=size[0], height=size[1]))
                reused += 1
                continue

            try:
                pix = page.get_pixmap(dpi=settings.page_image_dpi)
                _write_pixmap(pix, path, fmt)
            except Exception:
                failed += 1
                logger.warning("render_document_pages: page %d failed document_id=%s",
                               n, document_id, exc_info=True)
                continue

            images.append(PageImage(page=n, path=path, width=pix.width, height=pix.height))
            rendered += 1

    logger.info(
        "render_document_pages: document_id=%s pages=%d rendered=%d reused=%d failed=%d fmt=%s dpi=%d",
        document_id, len(images), rendered, reused, failed, fmt, settings.page_image_dpi,
    )
    return images


def list_page_images(document_id) -> List[PageImage]:
    """Page images already on disk, ordered by page. Never opens the PDF, so the
    preview manifest is a directory scan in the common case."""
    out_dir = pages_dir(document_id)
    if not os.path.isdir(out_dir):
        return []

    fmt = image_format()
    images: List[PageImage] = []
    for name in os.listdir(out_dir):
        stem, ext = os.path.splitext(name)
        if ext.lstrip(".").lower() != fmt or not stem.isdigit():
            continue
        path = os.path.join(out_dir, name)
        size = _read_size(path)
        if not size:
            continue
        images.append(PageImage(page=int(stem), path=path, width=size[0], height=size[1]))

    return sorted(images, key=lambda p: p.page)


def delete_page_images(document_id) -> int:
    """Remove every rendered page for a document. Returns the number of files
    deleted. Used by the re-render script's --force mode."""
    out_dir = pages_dir(document_id)
    if not os.path.isdir(out_dir):
        return 0
    removed = 0
    for name in os.listdir(out_dir):
        try:
            os.unlink(os.path.join(out_dir, name))
            removed += 1
        except OSError:
            logger.warning("delete_page_images: could not remove %s in %s", name, out_dir)
    try:
        os.rmdir(out_dir)
    except OSError:
        pass  # non-empty (something we didn't recognize) — leave it
    return removed
