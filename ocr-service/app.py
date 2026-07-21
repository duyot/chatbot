"""OCR microservice — PP-OCR models served via RapidOCR (ONNXRuntime).

Runs natively on arm64/amd64 with no PaddlePaddle dependency. Kept separate
from the Celery worker; the worker calls it over HTTP.

Endpoints:
  GET  /health  -> readiness probe
  POST /ocr     -> multipart image file; returns line-level text + bbox + score
"""
import io
import logging

from fastapi import FastAPI, File, UploadFile
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-service")

app = FastAPI(title="ocr-service")

# Load the OCR engine once at process start (PP-OCR det+rec+cls ONNX models,
# bundled with the package).
_ocr = RapidOCR()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    raw = await file.read()
    # PIL only for dimensions; RapidOCR decodes the bytes itself (correct color).
    width, height = Image.open(io.BytesIO(raw)).size

    result, _elapse = _ocr(raw)

    lines = []
    # RapidOCR returns a list of [box, text, score]; box is 4 [x, y] points.
    for item in (result or []):
        box, text, score = item[0], item[1], item[2]
        lines.append({
            "text": text,
            "bbox": [[float(p[0]), float(p[1])] for p in box],
            "confidence": float(score),
        })
        logger.info(f"box: {box}, text: {text}, score: {score}")

    logger.info("ocr: file=%s lines=%d", file.filename, len(lines))
    return {"lines": lines, "width": width, "height": height}
