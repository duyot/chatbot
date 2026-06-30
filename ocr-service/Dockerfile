FROM python:3.10-slim

# System libs required by OpenCV / ONNXRuntime (used internally by rapidocr).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Warm up: import + init so the bundled PP-OCR ONNX models are unpacked and the
# import path is verified at build time (no network download at runtime).
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
