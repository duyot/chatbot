"""Thin HTTP client for the PaddleOCR microservice.

The Celery worker stays lean (no PaddlePaddle dependency) and calls the OCR
service over HTTP. The service returns line-level results:

    {"lines": [{"text": str, "bbox": [...], "confidence": float}, ...],
     "width": int, "height": int}

`ocr_image` joins non-blank lines in returned (reading) order and averages the
confidence of the contributing lines. Transport/HTTP failures raise OCRError so
the caller (ingestion) can decide how to degrade.
"""
from __future__ import annotations
import logging

import httpx

from ..config import settings

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """Raised when the OCR microservice is unreachable or returns an error."""


class ParseTimeout(OCRError):
    """The parse exceeded settings.parse_timeout_s.

    Distinct from OCRError so the worker can skip retrying it — a parse that
    timed out once will time out again.
    """


# The only wire schema this client understands. A mismatch means ocr-service
# was deployed with an incompatible contract; fail loudly rather than guess.
SUPPORTED_SCHEMA_VERSION = 1


def ocr_image_lines(image_bytes: bytes, *, filename: str = "image") -> dict:
    """OCR a single rendered page / image, preserving line-level geometry.

    Returns the service's structured result:
        {"lines": [{"text": str, "bbox": [[x, y], ...], "confidence": float}, ...],
         "width": int, "height": int}
    where bbox is a polygon in image-pixel coordinates and width/height are the
    image dimensions (both needed to normalize bboxes downstream). Transport/HTTP
    failures raise OCRError so the caller can decide how to degrade.
    """
    url = settings.ocr_service_url.rstrip("/") + "/ocr"
    try:
        with httpx.Client(timeout=settings.ocr_timeout_s) as client:
            resp = client.post(url, files={"file": (filename, image_bytes)})
            resp.raise_for_status()
            body = resp.json()
    except Exception as exc:  # transport, HTTP status, or JSON decode
        logger.warning("ocr_image: request failed url=%s file=%s err=%s", url, filename, exc)
        raise OCRError(str(exc)) from exc

    if not isinstance(body, dict):
        return {"lines": [], "width": 0, "height": 0}
    return {
        "lines": body.get("lines") or [],
        "width": int(body.get("width") or 0),
        "height": int(body.get("height") or 0),
    }


def ocr_image(image_bytes: bytes, *, filename: str = "image") -> tuple[str, float | None]:
    """OCR a single rendered page / image. Returns (text, avg_confidence).

    Thin text-only wrapper over ocr_image_lines: joins non-blank lines in returned
    (reading) order and averages the confidence of the contributing lines.
    avg_confidence is None when the service returns no usable lines.
    """
    body = ocr_image_lines(image_bytes, filename=filename)
    texts: list[str] = []
    confs: list[float] = []
    for ln in body["lines"]:
        text = (ln.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        conf = ln.get("confidence")
        if isinstance(conf, (int, float)):
            confs.append(float(conf))

    joined = "\n".join(texts)
    avg_conf = (sum(confs) / len(confs)) if confs else None
    logger.info(
        "ocr_image: file=%s lines=%d text_len=%d avg_conf=%s",
        filename, len(texts), len(joined),
        f"{avg_conf:.3f}" if avg_conf is not None else "n/a",
    )
    return joined, avg_conf


def parse_document_remote(file_bytes: bytes, *, filename: str) -> dict:
    """Structure-aware parse of a whole document by ocr-service POST /parse.

    Returns the wire body: {"schema_version", "metadata", "pages", "elements"}
    where elements are typed, in reading order, with bboxes normalized to
    [0,1]. See ocr-service/wire.py for the full contract.

    Raises ParseTimeout on timeout and OCRError on any other transport, HTTP,
    JSON, or schema-version failure.
    """
    url = settings.ocr_service_url.rstrip("/") + "/parse"
    try:
        with httpx.Client(timeout=settings.parse_timeout_s) as client:
            resp = client.post(url, files={"file": (filename, file_bytes)})
            resp.raise_for_status()
            body = resp.json()
    except httpx.TimeoutException as exc:
        logger.warning(
            "parse_document_remote: timed out after %ss url=%s file=%s",
            settings.parse_timeout_s, url, filename,
        )
        raise ParseTimeout(
            f"parse exceeded {settings.parse_timeout_s}s for {filename}"
        ) from exc
    except Exception as exc:  # transport, HTTP status, or JSON decode
        logger.warning(
            "parse_document_remote: request failed url=%s file=%s err=%s",
            url, filename, exc,
        )
        raise OCRError(str(exc)) from exc

    if not isinstance(body, dict):
        raise OCRError(f"/parse returned {type(body).__name__}, expected object")
    version = body.get("schema_version")
    if version != SUPPORTED_SCHEMA_VERSION:
        raise OCRError(
            f"/parse returned schema_version={version!r}, "
            f"this client supports {SUPPORTED_SCHEMA_VERSION}"
        )

    logger.info(
        "parse_document_remote: file=%s pages=%s elements=%d",
        filename, body.get("metadata", {}).get("page_count"),
        len(body.get("elements") or []),
    )
    return body
