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
