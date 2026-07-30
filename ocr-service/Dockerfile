FROM python:3.10-slim

# System libs required by OpenCV / ONNXRuntime (used internally by rapidocr)
# and by Docling's PDF backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Plain `docling` (no [onnxruntime] extra: it adds onnxruntime-gpu, a GPU
# package, without avoiding torch). torch/torchvision are pulled in
# transitively via docling-ibm-models for TableFormer regardless of extras;
# requirements.txt's `--extra-index-url` (PyTorch's CPU wheel index) resolves
# CPU-only torch/torchvision instead of the default CUDA ones, which drops
# ~19 nvidia-* packages and keeps the image size sane.
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image. Docling otherwise downloads DocLayNet +
# TableFormer from HuggingFace on first use, which breaks an offline container
# and makes the first parse pay several hundred MB. Mirrors the existing
# RapidOCR warm-up below.
#
# The two caches are NOT the same place: `docling-tools` writes its weights to
# /root/.cache/docling/models, while HF_HOME only redirects huggingface_hub
# downloads. Both the prefetch above and the uvicorn runtime below run as root,
# so the baked weights are readable and `--network none` startup works. If this
# service is ever switched to a non-root user, /root/.cache stops being
# readable and the container will silently re-download models at runtime —
# relocate the docling cache as well if that changes.
ENV HF_HOME=/app/.cache/huggingface
RUN docling-tools models download layout tableformer \
    && python -c "from rapidocr import RapidOCR; RapidOCR()"

# app.py, wire.py, and parser.py (Docling adapter) — the whole application.
COPY app.py wire.py parser.py .

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
