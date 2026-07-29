# Structure-Aware OCR and Layout-Aware Chunking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace flat line-based OCR with Docling structure extraction in `ocr-service`, and replace character-count chunking with layout-aware chunking that keeps tables intact and attaches section headings to every chunk.

**Architecture:** `ocr-service` gains `POST /parse`, which takes a whole document and returns a versioned JSON list of typed elements (`heading`, `paragraph`, `table`, …) each with a normalized `[0,1]` bbox. The backend worker stops parsing files itself and becomes a client of that endpoint. A new `backend/app/services/chunking.py` walks those elements maintaining a heading stack, packs prose into parents up to a token budget, emits each table as its own atomic markdown chunk, and prefixes the heading path into every chunk's content.

**Tech Stack:** Docling (DocLayNet layout + TableFormer), RapidOCR as Docling's OCR backend, FastAPI, Celery, SQLAlchemy + Alembic, pytest, tiktoken, LangChain `RecursiveCharacterTextSplitter`.

**Spec:** `docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md`

## Global Constraints

- **CPU only, no GPU.** Never add a dependency or config that assumes CUDA.
- **Budget: 5-10s per page** is acceptable; ingestion is an async Celery job.
- **`schema_version` is `1`.** The backend client MUST reject any other value with `OCRError`.
- **Array order of `elements` is reading order.** The chunker depends on it; never sort it.
- **Unknown element types degrade to `paragraph`.** Never raise on an unrecognized Docling label.
- **`bbox` is `[x0, y0, x1, y1]` normalized to `[0,1]` and clamped**, top-left origin, converted in the service.
- **Fail loud.** A parse failure marks the document `status="failed"`. There is no automatic fallback to structure-less parsing.
- **`docling_enabled` is a manual rollback switch only.** The legacy path stays alive for exactly one release.
- **No backfill for any schema change.** `python -m scripts.reingest_all` is mandatory after this lands.
- **`build_embedding_input()` and `reranker._rerank_text()` must stay byte-identical.** The heading path goes *inside* `content`, which is what makes this possible. If you find yourself editing either function, stop — you have taken a wrong turn.
- **Never write `status="ready"`.** The production value is `status="done"`.
- Tests must not hit the network or run model inference in the default `pytest` run.

## File Structure

**`ocr-service/` (new files)**

| File | Responsibility |
|---|---|
| `wire.py` | The wire contract: `SCHEMA_VERSION`, `RawElement`/`RawPage` dataclasses, label→type mapping, bbox normalization, `to_wire()`. Pure functions, no Docling import. |
| `parser.py` | The only file that imports Docling. Converts bytes → `RawElement`/`RawPage` lists → `to_wire()`. |
| `tests/test_wire.py` | Unit tests for `wire.py`. No Docling, no inference. |
| `tests/test_parse_endpoint.py` | Endpoint tests with `parse_bytes` patched. |
| `tests/test_smoke_docling.py` | One real inference test, `@pytest.mark.slow`, excluded by default. |
| `pytest.ini` | `addopts = -m "not slow"`, mirroring the backend's `-m "not eval"` convention. |

The `wire.py` / `parser.py` split exists so the contract is testable without downloading models — `wire.py` is where all the logic that can be wrong lives, and it has zero heavy imports.

**`ocr-service/` (modified)**

| File | Change |
|---|---|
| `app.py` | Add `POST /parse`. Keep `POST /ocr` unchanged (legacy rollback path). |
| `requirements.txt` | Pins resolved by Task 1. |
| `Dockerfile` | Model prefetch at build time. |

**`backend/` (new files)**

| File | Responsibility |
|---|---|
| `app/services/chunking.py` | Layout-aware chunking. Pure functions over element lists — no DB, no HTTP, no LLM. |
| `tests/test_chunking.py` | Unit tests for the above. |
| `alembic/versions/0012_chunk_element_type.py` | Adds `document_chunks.element_type`. |

**`backend/` (modified)**

| File | Change |
|---|---|
| `app/services/ocr_client.py` | Add `parse_document_remote()`. Keep `ocr_image_lines`/`ocr_image`. |
| `app/services/ingestion.py` | Add `Element` dataclass + `ParsedDocument.elements`; `parse_document()` delegates to remote or legacy; `chunk_document()` delegates to `chunking.py`. |
| `app/models.py` | `DocumentChunk.element_type` column. |
| `app/config.py` | 8 new settings. |
| `app/workers/tasks.py` | Non-retryable parse timeout. |
| `evals/golden_set.yaml` | Table-lookup and cross-section questions. |
| `CLAUDE.md` | Document the new pipeline. |

---

### Task 1: Build spike — resolve dependencies and model prefetch

**This task gates every other task.** It commits no application logic; its deliverable is a buildable image and pinned dependency versions. If it fails, the whole approach changes.

**Files:**
- Modify: `ocr-service/requirements.txt`
- Modify: `ocr-service/Dockerfile`
- Modify: `docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md` (append a "Spike findings" section)

**Interfaces:**
- Consumes: nothing.
- Produces: a working `ocr-service` image with Docling installed and models baked in; the exact pinned versions later tasks build against.

**Context you need:** `ocr-service/requirements.txt` currently reads:

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.12
rapidocr
pillow==10.4.0
numpy==1.26.4
```

Note line 4 is bare `rapidocr` (the v3 unified package), not the older `rapidocr-onnxruntime`. v3 changed the import path and the result object shape, and Docling's `[rapidocr]` extra may expect one generation or the other. Resolving this is part of the spike.

The existing `Dockerfile` warms models at build time (`RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"`) so there is no runtime download. Docling fetches DocLayNet and TableFormer weights from HuggingFace on first use, so an equivalent prefetch is mandatory or the container is broken offline.

- [ ] **Step 1: Probe the dependency graph without building an image**

```bash
cd /tmp && rm -rf docling-spike && mkdir docling-spike && cd docling-spike
docker run --rm -v "$PWD:/out" python:3.10-slim bash -c '
  pip install --quiet --dry-run --report /out/report-full.json docling 2>&1 | tail -5
  pip install --quiet --dry-run --report /out/report-onnx.json "docling[onnxruntime]" 2>&1 | tail -5
'
python - <<"PY"
import json
for name in ("full", "onnx"):
    with open(f"report-{name}.json") as f:
        r = json.load(f)
    pkgs = sorted(i["metadata"]["name"].lower() for i in r["install"])
    print(name, "torch?", [p for p in pkgs if "torch" in p], "count:", len(pkgs))
PY
```

Record the answer to: **does `[onnxruntime]` avoid torch?**

- [ ] **Step 2: Determine which RapidOCR generation Docling drives**

```bash
docker run --rm python:3.10-slim bash -c '
  pip install --quiet "docling[rapidocr]" 2>&1 | tail -3
  pip list 2>/dev/null | grep -i rapid
  python -c "from docling.datamodel.pipeline_options import RapidOcrOptions; print(RapidOcrOptions())"
'
```

Record which package name and version resolve, and pin that exact spec in `requirements.txt`.

- [ ] **Step 3: Write the Dockerfile with model prefetch**

Replace `ocr-service/Dockerfile` with:

```dockerfile
FROM python:3.10-slim

# System libs required by OpenCV / ONNXRuntime (used internally by rapidocr)
# and by Docling's PDF backend.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image. Docling otherwise downloads DocLayNet +
# TableFormer from HuggingFace on first use, which breaks an offline container
# and makes the first parse pay several hundred MB. Mirrors the existing
# RapidOCR warm-up below.
ENV HF_HOME=/app/.cache/huggingface
RUN docling-tools models download layout tableformer \
    && python -c "from rapidocr import RapidOCR; RapidOCR()"

COPY wire.py parser.py app.py ./

EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

If `docling-tools models download` is not the correct command for the resolved version, find the right one with `docker run --rm <image> docling-tools --help` and use it. Do not skip the prefetch.

If Step 2 pinned `rapidocr-onnxruntime` rather than `rapidocr`, change the warm-up import to `from rapidocr_onnxruntime import RapidOCR`.

- [ ] **Step 4: Build and measure**

```bash
cd /d/development/chatbot/ocr-service
docker build -t ocr-spike .
docker images ocr-spike --format '{{.Size}}'
```

Record the size. Current image is ~500MB.

- [ ] **Step 5: Prove it works offline**

```bash
docker run --rm --network none ocr-spike python -c "
from docling.document_converter import DocumentConverter
DocumentConverter()
print('converter constructed offline OK')
"
```

Expected: prints the message. If it instead tries to reach `huggingface.co` and fails, the prefetch in Step 3 is wrong — fix it before proceeding.

- [ ] **Step 6: Record findings in the spec and decide**

Append to the spec, filling in real measured values:

```markdown
## Spike findings (Task 1, YYYY-MM-DD)

- `docling` on `python:3.10-slim`: <clean / needed X>
- `[onnxruntime]` extra avoids torch: <yes / no>
- Resolved RapidOCR package: `<name==version>`
- Final image size: <N> GB (was ~500 MB)
- Model prefetch command: `<command>`
- Offline construction verified: <yes / no>

**Decision:** <proceed with Docling / fall back to RapidLayout + RapidTable>
```

**STOP AND ASK THE USER** if the image exceeds ~4GB or torch proved unavoidable *and* the size is a problem. The spec's documented fallback is RapidLayout + RapidTable, which is a different plan.

- [ ] **Step 7: Commit**

```bash
git add ocr-service/requirements.txt ocr-service/Dockerfile \
        docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md
git commit -m "build: install docling in ocr-service with baked-in model weights

Prefetches DocLayNet and TableFormer at build time so the container works
offline, matching the existing RapidOCR warm-up. Records spike findings
(dependency graph, torch, image size) in the design spec."
```

---

### Task 2: Wire contract and mapper (pure, no Docling)

**Files:**
- Create: `ocr-service/wire.py`
- Create: `ocr-service/tests/__init__.py` (empty)
- Create: `ocr-service/tests/test_wire.py`
- Create: `ocr-service/pytest.ini`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `SCHEMA_VERSION: int = 1`
  - `RawElement(label: str, page: int, text: str, bbox_abs: tuple | None, level: int | None = None, confidence: float | None = None)`
  - `RawPage(page: int, width: float, height: float, source: str, ocr_confidence: float | None)`
  - `element_type(label: str) -> str`
  - `normalize_bbox(bbox_abs, width: float, height: float) -> list | None`
  - `to_wire(elements: list, pages: list, metadata: dict) -> dict`

Task 3 calls `to_wire`; Task 4 consumes the dict it returns.

- [ ] **Step 1: Write the failing tests**

Create `ocr-service/tests/test_wire.py`:

```python
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
```

- [ ] **Step 2: Add the pytest config so `slow` is excluded by default**

Create `ocr-service/pytest.ini`:

```ini
[pytest]
markers =
    slow: runs real Docling model inference (excluded from default runs)
addopts = -m "not slow"
```

- [ ] **Step 3: Run the tests to verify they fail**

```bash
cd /d/development/chatbot/ocr-service && python -m pytest tests/test_wire.py -v
```

Expected: collection error, `ModuleNotFoundError: No module named 'wire'`.

- [ ] **Step 4: Implement `wire.py`**

Create `ocr-service/wire.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /d/development/chatbot/ocr-service && python -m pytest tests/test_wire.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add ocr-service/wire.py ocr-service/pytest.ini ocr-service/tests/
git commit -m "feat(ocr): add /parse wire contract with element type mapping

Pure module with no Docling import so the contract is testable without
model weights. Unknown labels degrade to paragraph; bboxes normalize to
[0,1] against their own page and clamp."
```

---

### Task 3: Docling adapter and `POST /parse`

**Files:**
- Create: `ocr-service/parser.py`
- Create: `ocr-service/tests/test_parse_endpoint.py`
- Create: `ocr-service/tests/test_smoke_docling.py`
- Modify: `ocr-service/app.py`

**Interfaces:**
- Consumes: `wire.RawElement`, `wire.RawPage`, `wire.to_wire` (Task 2).
- Produces:
  - `parser.parse_bytes(data: bytes, *, filename: str) -> dict` — returns a `to_wire()` body.
  - `parser.ParseError(RuntimeError)` — raised on unconvertible input.
  - `POST /parse` accepting multipart `file`, returning the wire body, or 422 `{"detail": "..."}` on `ParseError`.

Task 4's client consumes this endpoint.

**Context:** the existing `ocr-service/app.py` has `GET /health` and `POST /ocr`. **Keep both unchanged** — `/ocr` is the legacy rollback path per the spec.

Docling's `DocumentConverter.convert()` returns a result whose `.document` is a `DoclingDocument`. Iterate with `doc.iterate_items()`, which yields `(item, level)`. Items carry `.label`, `.prov[0].page_no`, `.prov[0].bbox`, and tables expose `.export_to_markdown()`. Docling bboxes use a **bottom-left origin**, so y must be flipped against page height before constructing a `RawElement`.

- [ ] **Step 1: Write the failing endpoint tests**

Create `ocr-service/tests/test_parse_endpoint.py`:

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import app as app_module
    return TestClient(app_module.app), app_module


def test_parse_returns_wire_body(client, mocker):
    c, app_module = client
    mocker.patch.object(app_module, "parse_bytes", return_value={
        "schema_version": 1,
        "metadata": {"page_count": 1},
        "pages": [],
        "elements": [{"id": "e0", "page": 1, "type": "paragraph",
                      "text": "hi", "bbox": None, "confidence": None}],
    })
    resp = c.post("/parse", files={"file": ("doc.pdf", b"%PDF-fake", "application/pdf")})

    assert resp.status_code == 200
    assert resp.json()["schema_version"] == 1
    assert resp.json()["elements"][0]["text"] == "hi"


def test_parse_returns_422_on_parse_error(client, mocker):
    c, app_module = client
    mocker.patch.object(app_module, "parse_bytes",
                        side_effect=app_module.ParseError("not a document"))
    resp = c.post("/parse", files={"file": ("broken.pdf", b"garbage", "application/pdf")})

    assert resp.status_code == 422
    assert "not a document" in resp.json()["detail"]


def test_parse_passes_filename_through(client, mocker):
    c, app_module = client
    patched = mocker.patch.object(app_module, "parse_bytes", return_value={
        "schema_version": 1, "metadata": {}, "pages": [], "elements": [],
    })
    c.post("/parse", files={"file": ("report.docx", b"PK\x03\x04", None)})

    assert patched.call_args.args[0] == b"PK\x03\x04"
    assert patched.call_args.kwargs["filename"] == "report.docx"


def test_ocr_endpoint_still_exists(client):
    """The legacy path must survive — it is the documented rollback."""
    _, app_module = client
    assert any(getattr(r, "path", None) == "/ocr" for r in app_module.app.routes)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /d/development/chatbot/ocr-service && python -m pytest tests/test_parse_endpoint.py -v
```

Expected: FAIL — `app` has no attribute `parse_bytes`, and `/parse` returns 404.

- [ ] **Step 3: Implement `parser.py`**

Create `ocr-service/parser.py`:

```python
"""Docling adapter — the only module in this service that imports Docling.

Converts document bytes into the `wire.to_wire()` body. All normalization logic
lives in `wire.py`; this file is purely extraction and coordinate conversion.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import List, Optional

from docling.document_converter import DocumentConverter

from wire import RawElement, RawPage, to_wire

logger = logging.getLogger("ocr-service.parser")


class ParseError(RuntimeError):
    """Raised when Docling cannot convert the input at all."""


_converter: Optional[DocumentConverter] = None


def _get_converter() -> DocumentConverter:
    """Built once per process. Construction loads model weights, so it must not
    happen per request."""
    global _converter
    if _converter is None:
        _converter = DocumentConverter()
    return _converter


def _page_dims(dl_doc) -> dict:
    """{page_no: (width, height)} from the Docling document."""
    dims = {}
    for page_no, page in (getattr(dl_doc, "pages", {}) or {}).items():
        size = getattr(page, "size", None)
        if size is not None:
            dims[int(page_no)] = (float(size.width), float(size.height))
    return dims


def _raw_pages(dl_doc, dims: dict) -> List[RawPage]:
    """One RawPage per page. `source` is "ocr" when that page carried no native
    text cells, which is how we report ocr_pages/native_pages upstream."""
    pages = []
    for page_no in sorted(dims):
        width, height = dims[page_no]
        page = (getattr(dl_doc, "pages", {}) or {}).get(page_no)
        cells = getattr(page, "cells", None) or []
        pages.append(RawPage(
            page=page_no,
            width=width,
            height=height,
            source="native" if cells else "ocr",
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
    """Docling bboxes use a bottom-left origin; the wire format and the page
    images the UI overlays both use top-left. Flip y here."""
    bbox = getattr(prov, "bbox", None)
    if bbox is None or not page_height:
        return None
    return (
        float(bbox.l),
        page_height - float(bbox.t),
        float(bbox.r),
        page_height - float(bbox.b),
    )


def _label_of(item) -> str:
    label = getattr(item, "label", "")
    return str(getattr(label, "value", label))


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
        pages = _raw_pages(dl_doc, dims)

        elements: List[RawElement] = []
        for item, level in dl_doc.iterate_items():
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
                level=level,
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
```

- [ ] **Step 4: Add the `/parse` endpoint**

In `ocr-service/app.py`, replace the existing FastAPI import line with these two lines:

```python
from fastapi import FastAPI, File, HTTPException, UploadFile

from parser import ParseError, parse_bytes
```

Then append this endpoint after the existing `/ocr` handler:

```python
@app.post("/parse")
async def parse(file: UploadFile = File(...)):
    """Structure-aware parse of a whole document (PDF / DOCX / image).

    Returns typed elements in reading order with normalized bboxes — see
    wire.py for the contract. 422 means "this input is not convertible",
    which the caller surfaces to the user; anything else is a real 500.
    """
    raw = await file.read()
    try:
        body = parse_bytes(raw, filename=file.filename or "document")
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    logger.info(
        "parse: file=%s pages=%s elements=%d",
        file.filename, body["metadata"].get("page_count"), len(body["elements"]),
    )
    return body
```

- [ ] **Step 5: Run the endpoint tests to verify they pass**

```bash
cd /d/development/chatbot/ocr-service && python -m pytest tests/test_parse_endpoint.py -v
```

Expected: all PASS.

- [ ] **Step 6: Add the real-inference smoke test**

Create `ocr-service/tests/test_smoke_docling.py`:

```python
"""One real Docling inference run. Excluded from default runs by pytest.ini
(`-m "not slow"`) because it loads model weights.

Run explicitly inside the built image:
    docker run --rm ocr-spike python -m pytest tests/test_smoke_docling.py -m slow -v
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
```

- [ ] **Step 7: Run the smoke test inside the image**

```bash
cd /d/development/chatbot/ocr-service
docker build -t ocr-spike .
docker run --rm --network none -v "$PWD/tests:/app/tests" ocr-spike \
  python -m pytest tests/test_smoke_docling.py -m slow -v
```

Expected: PASS, with no network access. If it fails reaching HuggingFace, Task 1 Step 3's prefetch is incomplete.

- [ ] **Step 8: Commit**

```bash
git add ocr-service/parser.py ocr-service/app.py ocr-service/tests/
git commit -m "feat(ocr): add POST /parse backed by docling

Whole-document parse returning typed elements in reading order with
bboxes normalized to [0,1] and flipped to a top-left origin. Converter is
built once per process. Legacy POST /ocr is untouched as the rollback path."
```

---

### Task 4: Backend client for `/parse`

**Files:**
- Modify: `backend/app/config.py`
- Modify: `backend/app/services/ocr_client.py`
- Modify: `backend/tests/test_ocr_client.py`

**Interfaces:**
- Consumes: the `POST /parse` contract from Task 3.
- Produces:
  - `ocr_client.parse_document_remote(file_bytes: bytes, *, filename: str) -> dict` — the validated wire body; raises `OCRError` on any transport, HTTP, JSON, or schema-version failure.
  - `ocr_client.ParseTimeout(OCRError)` — raised specifically on timeout, so Task 10 can make it non-retryable.
  - `ocr_client.SUPPORTED_SCHEMA_VERSION: int = 1`
  - New settings: `docling_enabled`, `docling_ocr_backend`, `docling_table_mode`, `parse_timeout_s`, `drop_element_types`, `parent_max_tokens`, `table_max_tokens`, `table_row_group_rows`.

- [ ] **Step 1: Add the settings**

In `backend/app/config.py`, insert after the OCR block (which ends with `ocr_dpi: int = 200`):

```python
    # --- Structure-aware parsing (see
    # docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md).
    # The worker POSTs the whole document to ocr-service /parse and receives
    # typed elements. Set False to fall back to the legacy per-page-image
    # line-based path, which is kept for exactly one release.
    docling_enabled: bool = True
    docling_ocr_backend: str = "rapidocr"
    docling_table_mode: str = "accurate"
    # A 100-page scan at 5-10s/page is a 10-15 minute synchronous request.
    # ocr_timeout_s (60s) sizes the per-image legacy call and is far too small.
    parse_timeout_s: float = 1800.0

    # --- Layout-aware chunking (same spec).
    # Per-page running noise, dropped before chunking.
    drop_element_types: list[str] = ["page_header", "page_footer"]
    # Token budgets, measured with tiktoken cl100k_base to match the splitter.
    parent_max_tokens: int = 1500
    table_max_tokens: int = 1500
    # Data rows per group when a table exceeds table_max_tokens. The header row
    # and its separator are repeated in every group.
    table_row_group_rows: int = 10
```

- [ ] **Step 2: Write the failing client tests**

Append to `backend/tests/test_ocr_client.py`:

```python
def _wire_body(**overrides):
    body = {
        "schema_version": 1,
        "metadata": {"page_count": 1, "ocr_pages": 1, "native_pages": 0},
        "pages": [{"page": 1, "width": 600.0, "height": 800.0,
                   "source": "ocr", "ocr_confidence": 0.9}],
        "elements": [{"id": "e0", "page": 1, "type": "paragraph",
                      "text": "hello", "bbox": [0.1, 0.1, 0.5, 0.2],
                      "confidence": 0.9}],
    }
    body.update(overrides)
    return body


def test_parse_document_remote_returns_body(mocker):
    from app.services.ocr_client import parse_document_remote
    patched = _mock_httpx_client(mocker, _wire_body())

    body = parse_document_remote(b"%PDF-fake", filename="doc.pdf")

    assert body["elements"][0]["text"] == "hello"
    posted = patched.return_value.__enter__.return_value.post.call_args
    assert posted.args[0].endswith("/parse")
    assert "files" in posted.kwargs


def test_parse_document_remote_rejects_unknown_schema_version(mocker):
    from app.services.ocr_client import parse_document_remote, OCRError
    _mock_httpx_client(mocker, _wire_body(schema_version=2))

    with pytest.raises(OCRError, match="schema_version"):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_rejects_non_dict_body(mocker):
    from app.services.ocr_client import parse_document_remote, OCRError
    _mock_httpx_client(mocker, ["not", "a", "dict"])

    with pytest.raises(OCRError):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_raises_ocr_error_on_http_failure(mocker):
    from app.services.ocr_client import parse_document_remote, OCRError
    _mock_httpx_client(mocker, body=None, raises=RuntimeError("service down"))

    with pytest.raises(OCRError):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_raises_parse_timeout_on_timeout(mocker):
    """Timeout gets its own type so the worker can make it non-retryable."""
    import httpx
    from app.services.ocr_client import parse_document_remote, ParseTimeout
    _mock_httpx_client(mocker, body=None, raises=httpx.ReadTimeout("too slow"))

    with pytest.raises(ParseTimeout):
        parse_document_remote(b"x", filename="doc.pdf")


def test_parse_document_remote_uses_the_long_parse_timeout(mocker):
    from app.config import settings
    from app.services.ocr_client import parse_document_remote
    patched = _mock_httpx_client(mocker, _wire_body())

    parse_document_remote(b"x", filename="doc.pdf")

    assert patched.call_args.kwargs["timeout"] == settings.parse_timeout_s
```

- [ ] **Step 3: Run to verify they fail**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_ocr_client.py -v
```

Expected: FAIL — `cannot import name 'parse_document_remote'`.

- [ ] **Step 4: Implement the client**

In `backend/app/services/ocr_client.py` (`import httpx` is already present), add after the `OCRError` class:

```python
class ParseTimeout(OCRError):
    """The parse exceeded settings.parse_timeout_s.

    Distinct from OCRError so the worker can skip retrying it — a parse that
    timed out once will time out again.
    """


# The only wire schema this client understands. A mismatch means ocr-service
# was deployed with an incompatible contract; fail loudly rather than guess.
SUPPORTED_SCHEMA_VERSION = 1


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
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_ocr_client.py -v
```

Expected: all PASS, including the three pre-existing `ocr_image` tests.

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/app/services/ocr_client.py backend/tests/test_ocr_client.py
git commit -m "feat: add parse_document_remote client for ocr-service /parse

Validates schema_version at the boundary and raises a distinct
ParseTimeout so the worker can treat timeouts as non-retryable. Adds the
docling and layout-chunking settings."
```

---

### Task 5: `parse_document()` switches to the remote parser

**Files:**
- Modify: `backend/app/services/ingestion.py`
- Modify: `backend/tests/test_ingestion.py`

**Interfaces:**
- Consumes: `ocr_client.parse_document_remote` (Task 4).
- Produces:
  - `ingestion.Element(id: str, page: int, type: str, text: str, bbox: Optional[List[float]] = None, level: Optional[int] = None, confidence: Optional[float] = None)`
  - `ParsedDocument.elements: List[Element]` — empty on the legacy path.
  - `parse_document(file_path, file_name) -> ParsedDocument` — unchanged signature.

Tasks 6-8's chunker consumes `Element`; Task 9 wires it in.

**Context:** `ParsedDocument` already has `pages: List[PageContent]` and a `.text` property joining page texts. `contextualizer.contextualize_with_stats(parsed, children_per_parent)` depends on both `parsed.text` and per-page text, so **`pages` must keep being populated** — the new code builds `PageContent` from the elements grouped by page.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_ingestion.py`:

```python
def _wire(elements, pages=None):
    return {
        "schema_version": 1,
        "metadata": {"page_count": len(pages or [1]), "ocr_pages": 1, "native_pages": 0},
        "pages": pages or [{"page": 1, "width": 600.0, "height": 800.0,
                            "source": "ocr", "ocr_confidence": 0.91}],
        "elements": elements,
    }


def test_parse_document_uses_remote_parser_and_keeps_elements(tmp_path, mocker):
    from app.services import ingestion
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-fake")
    mocker.patch.object(ingestion.settings, "docling_enabled", True)
    mocker.patch.object(ingestion, "parse_document_remote", return_value=_wire([
        {"id": "e0", "page": 1, "type": "heading", "level": 2,
         "text": "3.2 Revenue", "bbox": [0.1, 0.1, 0.6, 0.14], "confidence": None},
        {"id": "e1", "page": 1, "type": "table",
         "text": "| R | Q1 |\n|---|---|\n| APAC | 12 |",
         "bbox": [0.1, 0.2, 0.9, 0.5], "confidence": 0.9},
    ]))

    parsed = ingestion.parse_document(str(pdf), "scan.pdf")

    assert [e.type for e in parsed.elements] == ["heading", "table"]
    assert parsed.elements[0].level == 2
    assert parsed.elements[1].bbox == [0.1, 0.2, 0.9, 0.5]


def test_parse_document_builds_pages_from_elements(tmp_path, mocker):
    """contextualizer needs parsed.text and per-page text, so pages must still
    be populated even though chunking now works off elements."""
    from app.services import ingestion
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-fake")
    mocker.patch.object(ingestion.settings, "docling_enabled", True)
    mocker.patch.object(ingestion, "parse_document_remote", return_value=_wire(
        [
            {"id": "e0", "page": 1, "type": "paragraph", "text": "page one prose",
             "bbox": None, "confidence": None},
            {"id": "e1", "page": 2, "type": "paragraph", "text": "page two prose",
             "bbox": None, "confidence": None},
        ],
        pages=[
            {"page": 1, "width": 600.0, "height": 800.0, "source": "ocr",
             "ocr_confidence": 0.9},
            {"page": 2, "width": 600.0, "height": 800.0, "source": "native",
             "ocr_confidence": None},
        ],
    ))

    parsed = ingestion.parse_document(str(pdf), "scan.pdf")

    assert [p.page for p in parsed.pages] == [1, 2]
    assert parsed.pages[0].text == "page one prose"
    assert parsed.pages[0].source == "ocr"
    assert parsed.pages[1].source == "native"
    assert "page one prose" in parsed.text and "page two prose" in parsed.text


def test_parse_document_sets_mime_type_and_engine(tmp_path, mocker):
    from app.services import ingestion
    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-fake")
    mocker.patch.object(ingestion.settings, "docling_enabled", True)
    mocker.patch.object(ingestion, "parse_document_remote", return_value=_wire([]))

    parsed = ingestion.parse_document(str(pdf), "scan.pdf")

    assert parsed.metadata["mime_type"] == "application/pdf"
    assert parsed.metadata["engine"] == "docling"


def test_parse_document_legacy_path_when_docling_disabled(tmp_path, mocker):
    """docling_enabled=False must reproduce the old behaviour exactly."""
    from docx import Document as DocxDocument
    from app.services import ingestion
    docx_path = tmp_path / "legacy.docx"
    doc = DocxDocument()
    doc.add_paragraph("Legacy path still works")
    doc.save(str(docx_path))
    mocker.patch.object(ingestion.settings, "docling_enabled", False)
    remote = mocker.patch.object(ingestion, "parse_document_remote")

    parsed = ingestion.parse_document(str(docx_path), "legacy.docx")

    remote.assert_not_called()
    assert parsed.elements == []
    assert "Legacy path still works" in parsed.pages[0].text
```

The pre-existing `parse_document` tests in this file (`test_parse_document_docx_returns_native_page`, `test_parse_document_image_uses_ocr`) exercise the legacy path. Add `mocker.patch.object(ingestion.settings, "docling_enabled", False)` as the first line of each, adding a `mocker` argument where one is missing — they are legacy-path tests now and should say so.

- [ ] **Step 2: Run to verify they fail**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_ingestion.py -v
```

Expected: FAIL — `ingestion` has no attribute `parse_document_remote`.

- [ ] **Step 3: Add the `Element` dataclass and `elements` field**

In `backend/app/services/ingestion.py`, change the ocr_client import line to:

```python
from .ocr_client import ocr_image_lines, parse_document_remote, OCRError
```

Add this dataclass immediately after `LayoutLine`:

```python
@dataclass
class Element:
    """One typed document element from ocr-service /parse.

    `bbox` is [x0, y0, x1, y1] normalized to [0,1] of its page, or None when the
    service could not determine one. `type` is one of the eight wire types; see
    ocr-service/wire.py. Reading order is list order — never sort these.
    """
    id: str
    page: int
    type: str
    text: str
    bbox: Optional[List[float]] = None
    level: Optional[int] = None
    confidence: Optional[float] = None
```

Add the field to `ParsedDocument`:

```python
@dataclass
class ParsedDocument:
    pages: List[PageContent]
    metadata: dict
    # Typed elements in reading order. Empty on the legacy (docling_enabled=False)
    # path, which is what makes the legacy chunker's fallback detectable.
    elements: List[Element] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(p.text for p in self.pages if p.text)
```

- [ ] **Step 4: Implement the remote parse path**

Add these functions to `ingestion.py` immediately before `def parse_document(`:

```python
def _elements_from_wire(body: dict) -> List[Element]:
    return [
        Element(
            id=el.get("id") or f"e{i}",
            page=int(el.get("page") or 1),
            type=el.get("type") or "paragraph",
            text=el.get("text") or "",
            bbox=el.get("bbox"),
            level=el.get("level"),
            confidence=el.get("confidence"),
        )
        for i, el in enumerate(body.get("elements") or [])
    ]


def _pages_from_wire(body: dict, elements: List[Element]) -> List[PageContent]:
    """Rebuild PageContent from the wire body.

    Chunking works off `elements`, but the contextualizer still needs
    `parsed.text` and per-page text, so every page gets its elements joined in
    reading order. `lines` stays empty — bbox attribution is per-element now.
    """
    text_by_page: dict = {}
    for el in elements:
        text_by_page.setdefault(el.page, []).append(el.text)

    pages: List[PageContent] = []
    for p in body.get("pages") or []:
        page_no = int(p.get("page") or 1)
        pages.append(PageContent(
            page=page_no,
            text="\n".join(text_by_page.get(page_no, [])),
            source=p.get("source") or "ocr",
            ocr_confidence=p.get("ocr_confidence"),
            width=p.get("width"),
            height=p.get("height"),
        ))
    return pages


def _parse_remote(file_path: str, file_name: str) -> ParsedDocument:
    """Structure-aware parse via ocr-service /parse. Handles PDF, DOCX and
    images uniformly — Docling detects the format itself.

    Errors propagate as OCRError/ParseTimeout: a failed parse must fail the
    document rather than silently degrade to structure-less chunks.
    """
    with open(file_path, "rb") as f:
        data = f.read()
    body = parse_document_remote(data, filename=file_name)
    elements = _elements_from_wire(body)
    pages = _pages_from_wire(body, elements)
    metadata = dict(body.get("metadata") or {})
    metadata["engine"] = "docling"
    return ParsedDocument(pages=pages, metadata=metadata, elements=elements)
```

- [ ] **Step 5: Route `parse_document` through it**

Replace the body of `parse_document` with:

```python
def parse_document(file_path: str, file_name: str) -> ParsedDocument:
    """Parse a source file into typed elements + per-page text + metadata.

    With docling_enabled (the default) this is a single call to ocr-service
    /parse, which handles PDF, DOCX and images. The legacy per-format,
    line-based path is kept behind the flag for one release as the documented
    rollback; see the design spec.
    """
    ext = os.path.splitext(file_name)[1].lower()
    if settings.docling_enabled:
        parsed = _parse_remote(file_path, file_name)
    elif ext == ".pdf":
        parsed = _parse_pdf(file_path, file_name)
    elif ext == ".docx":
        parsed = _parse_docx(file_path)
    elif ext in _IMAGE_EXTS:
        parsed = _parse_image(file_path, file_name)
    else:
        parsed = ParsedDocument(pages=[], metadata={"page_count": 0})
    parsed.metadata["mime_type"] = _MIME_BY_EXT.get(ext)
    total_len = sum(len(p.text) for p in parsed.pages)
    logger.info(
        "parse_document: file=%s type=%s pages=%d elements=%d ocr_pages=%s text_len=%d",
        file_name, ext or "?", len(parsed.pages), len(parsed.elements),
        parsed.metadata.get("ocr_pages"), total_len,
    )
    return parsed
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_ingestion.py -v
```

Expected: all PASS, including the legacy tests updated in Step 1.

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/ingestion.py backend/tests/test_ingestion.py
git commit -m "feat: parse documents via ocr-service /parse

parse_document now makes one remote call for PDF, DOCX and images and
returns typed elements alongside per-page text, which the contextualizer
still needs. Legacy per-format path kept behind docling_enabled."
```

---

### Task 6: Heading stack and drop rules

**Files:**
- Create: `backend/app/services/chunking.py`
- Create: `backend/tests/test_chunking.py`

**Interfaces:**
- Consumes: `ingestion.Element` (Task 5).
- Produces:
  - `chunking.PROSE_TYPES: set` = `{"paragraph", "list_item", "caption"}`
  - `chunking.Section(heading_path: str, elements: List[Element])`
  - `chunking.keep_elements(elements: List[Element]) -> List[Element]`
  - `chunking.split_sections(elements: List[Element]) -> List[Section]`

Tasks 7-9 consume `Section`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_chunking.py`:

```python
from app.services.ingestion import Element


def _el(type_, text, page=1, level=None, bbox=None, id_=None, confidence=None):
    return Element(
        id=id_ or f"e-{text[:8]}", page=page, type=type_, text=text,
        bbox=bbox, level=level, confidence=confidence,
    )


# --- keep_elements (drop rules) --------------------------------------------

def test_keep_elements_drops_page_headers_and_footers():
    from app.services.chunking import keep_elements
    kept = keep_elements([
        _el("page_header", "ACME Confidential"),
        _el("paragraph", "Real content"),
        _el("page_footer", "Page 3 of 12"),
    ])
    assert [e.text for e in kept] == ["Real content"]


def test_keep_elements_drops_uncaptioned_figure():
    from app.services.chunking import keep_elements
    kept = keep_elements([_el("figure", "chart.png"), _el("paragraph", "Body")])
    assert [e.type for e in kept] == ["paragraph"]


def test_keep_elements_keeps_caption_that_follows_a_figure():
    from app.services.chunking import keep_elements
    kept = keep_elements([
        _el("figure", "chart.png"),
        _el("caption", "Figure 1: revenue by region"),
    ])
    assert [e.type for e in kept] == ["caption"]
    assert kept[0].text == "Figure 1: revenue by region"


def test_keep_elements_drops_blank_text():
    from app.services.chunking import keep_elements
    kept = keep_elements([_el("paragraph", "   "), _el("paragraph", "Real")])
    assert [e.text for e in kept] == ["Real"]


def test_keep_elements_respects_configured_drop_list(mocker):
    from app.services import chunking
    mocker.patch.object(chunking.settings, "drop_element_types", ["caption"])
    kept = chunking.keep_elements([
        _el("caption", "dropped now"),
        _el("page_header", "kept now"),
    ])
    assert [e.text for e in kept] == ["kept now"]


# --- split_sections (heading stack) ----------------------------------------

def test_split_sections_builds_nested_heading_path():
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "3. Financials", level=1),
        _el("paragraph", "intro"),
        _el("heading", "3.2 Revenue", level=2),
        _el("paragraph", "revenue prose"),
    ])
    assert [s.heading_path for s in sections] == [
        "3. Financials",
        "3. Financials > 3.2 Revenue",
    ]
    assert [e.text for e in sections[1].elements] == ["revenue prose"]


def test_split_sections_pops_stack_on_shallower_heading():
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "A", level=1),
        _el("heading", "A.1", level=2),
        _el("paragraph", "deep"),
        _el("heading", "B", level=1),
        _el("paragraph", "shallow"),
    ])
    assert sections[-1].heading_path == "B"
    assert [e.text for e in sections[-1].elements] == ["shallow"]


def test_split_sections_tolerates_level_jumps():
    """h1 -> h3 with no h2 must not crash or produce an empty path segment."""
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "A", level=1),
        _el("heading", "A.0.1", level=3),
        _el("paragraph", "body"),
    ])
    assert sections[-1].heading_path == "A > A.0.1"


def test_split_sections_content_before_any_heading_has_empty_path():
    from app.services.chunking import split_sections
    sections = split_sections([_el("paragraph", "preamble")])
    assert sections[0].heading_path == ""
    assert [e.text for e in sections[0].elements] == ["preamble"]


def test_split_sections_omits_sections_with_no_body():
    """A heading immediately followed by another heading yields no section."""
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "A", level=1),
        _el("heading", "A.1", level=2),
        _el("paragraph", "body"),
    ])
    assert len(sections) == 1
    assert sections[0].heading_path == "A > A.1"


def test_split_sections_defaults_missing_level_to_one():
    from app.services.chunking import split_sections
    sections = split_sections([
        _el("heading", "No level given", level=None),
        _el("paragraph", "body"),
    ])
    assert sections[0].heading_path == "No level given"


def test_split_sections_empty_input_returns_empty():
    from app.services.chunking import split_sections
    assert split_sections([]) == []
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_chunking.py -v
```

Expected: collection error, `No module named 'app.services.chunking'`.

- [ ] **Step 3: Implement the module**

Create `backend/app/services/chunking.py`:

```python
"""Layout-aware chunking over typed elements from ocr-service /parse.

Split out of ingestion.py, which already covers parsing, embedding and
persistence. Everything here is a pure function over element lists — no DB, no
HTTP, no LLM — so it is cheap to test exhaustively.

Design: docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from ..config import settings
from .ingestion import Element

logger = logging.getLogger(__name__)

# Element types that pack together into prose parents.
PROSE_TYPES = {"paragraph", "list_item", "caption"}


@dataclass
class Section:
    """A heading path plus the body elements beneath it, in reading order.

    Parents never span two sections, which is what keeps unrelated topics out
    of the same chunk.
    """
    heading_path: str
    elements: List[Element] = field(default_factory=list)


def keep_elements(elements: List[Element]) -> List[Element]:
    """Drop per-page noise before chunking.

    Removes configured types (running headers/footers by default), blank text,
    and figures — a figure's only useful text is its caption, which arrives as
    its own `caption` element and is kept.
    """
    dropped = set(settings.drop_element_types or [])
    kept = [
        el for el in elements
        if el.type not in dropped and el.type != "figure" and el.text.strip()
    ]
    if len(kept) != len(elements):
        logger.debug(
            "keep_elements: dropped %d of %d", len(elements) - len(kept), len(elements)
        )
    return kept


def split_sections(elements: List[Element]) -> List[Section]:
    """Group elements into sections using a heading stack.

    A `heading` of level L truncates the stack to L-1 entries and pushes its
    text, so the path is the chain of enclosing titles. Level jumps (h1 -> h3)
    simply produce a shorter chain rather than empty segments. Sections with no
    body elements are omitted — a heading followed immediately by another
    heading carries no content of its own.
    """
    kept = keep_elements(elements)
    sections: List[Section] = []
    stack: List[str] = []
    current = Section(heading_path="")

    for el in kept:
        if el.type == "heading":
            if current.elements:
                sections.append(current)
            level = el.level if el.level and el.level > 0 else 1
            del stack[level - 1:]
            stack.append(el.text.strip())
            current = Section(heading_path=" > ".join(stack))
        else:
            current.elements.append(el)

    if current.elements:
        sections.append(current)
    return sections
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_chunking.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/chunking.py backend/tests/test_chunking.py
git commit -m "feat: add heading stack and drop rules for layout-aware chunking

Sections come from a heading stack so no parent spans two sections. Running
headers/footers and uncaptioned figures are dropped; captions survive."
```

---

### Task 7: Prose parents, heading-path prefix, children, and bboxes

**Files:**
- Modify: `backend/app/services/chunking.py`
- Modify: `backend/app/services/ingestion.py`
- Modify: `backend/tests/test_chunking.py`

**Interfaces:**
- Consumes: `Section`, `PROSE_TYPES`, `split_sections` (Task 6); `ingestion.ParentChunk`, `ingestion.ChildChunk`, `ingestion._child_splitter`, `ingestion._find_from`.
- Produces:
  - `chunking.ELEMENT_JOINER: str = "\n\n"`
  - `chunking.count_tokens(text: str) -> int`
  - `chunking.with_heading(heading_path: str, body: str) -> str`
  - `chunking.ElementSpan(start: int, end: int, element: Element)`
  - `chunking.pack_prose(elements: List[Element], max_tokens: int) -> List[List[Element]]`
  - `chunking.build_prose_parent(heading_path: str, group: List[Element]) -> tuple`
  - `chunking._group_source(group: List[Element]) -> str`
  - `ingestion.ChildChunk.element_type: Optional[str]`

Task 8 reuses `with_heading`, `count_tokens`, and `_group_source`; Task 9 calls `build_prose_parent`.

**Context:** `ingestion.ParentChunk(content, page_start, page_end, source)` already exists unchanged. `ingestion._find_from(haystack, needle, cursor)` is the existing cursor-tracking `str.find`, returning `-1` when absent. `ingestion._child_splitter()` returns the 300/50 splitter.

`ELEMENT_JOINER` must be `"\n\n"` and used consistently, because the char-span arithmetic that recovers bboxes depends on knowing exactly how many characters sit between elements.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_chunking.py`:

```python
# --- with_heading ----------------------------------------------------------

def test_with_heading_prefixes_path():
    from app.services.chunking import with_heading
    assert with_heading("A > B", "body text") == "A > B\n\nbody text"


def test_with_heading_returns_body_unchanged_when_path_empty():
    from app.services.chunking import with_heading
    assert with_heading("", "body text") == "body text"


# --- pack_prose ------------------------------------------------------------

def test_pack_prose_keeps_small_elements_in_one_group():
    from app.services.chunking import pack_prose
    els = [_el("paragraph", "short one"), _el("paragraph", "short two")]
    groups = pack_prose(els, max_tokens=1500)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_pack_prose_splits_when_budget_exceeded():
    from app.services.chunking import pack_prose
    els = [_el("paragraph", "word " * 200, id_=f"e{i}") for i in range(6)]
    groups = pack_prose(els, max_tokens=300)
    assert len(groups) > 1
    # every element lands in exactly one group, order preserved
    flat = [e.id for g in groups for e in g]
    assert flat == [e.id for e in els]


def test_pack_prose_never_drops_an_oversized_single_element():
    from app.services.chunking import pack_prose
    els = [_el("paragraph", "word " * 2000)]
    groups = pack_prose(els, max_tokens=100)
    assert len(groups) == 1 and len(groups[0]) == 1


def test_pack_prose_empty_input_returns_empty():
    from app.services.chunking import pack_prose
    assert pack_prose([], max_tokens=1500) == []


# --- build_prose_parent ----------------------------------------------------

def test_build_prose_parent_prefixes_heading_on_parent_and_every_child():
    from app.services.chunking import build_prose_parent
    els = [_el("paragraph", "word " * 200), _el("paragraph", "other " * 200)]
    parent, children = build_prose_parent("A > B", els)

    assert parent.content.startswith("A > B\n\n")
    assert len(children) > 1, "expected the 300-token splitter to produce several children"
    for child in children:
        assert child.content.startswith("A > B\n\n"), \
            "every child needs the header, not just the first"


def test_build_prose_parent_spans_pages():
    from app.services.chunking import build_prose_parent
    els = [_el("paragraph", "on one", page=4), _el("paragraph", "on two", page=5)]
    parent, _ = build_prose_parent("", els)
    assert (parent.page_start, parent.page_end) == (4, 5)


def test_build_prose_parent_child_collects_bboxes_of_overlapping_elements():
    from app.services.chunking import build_prose_parent
    a = _el("paragraph", "alpha text here", bbox=[0.0, 0.0, 1.0, 0.1])
    b = _el("paragraph", "beta text here", bbox=[0.0, 0.2, 1.0, 0.3])
    parent, children = build_prose_parent("", [a, b])

    # Short content -> one child spanning both elements, so it carries both rects.
    assert len(children) == 1
    assert children[0].bbox == [[0.0, 0.0, 1.0, 0.1], [0.0, 0.2, 1.0, 0.3]]


def test_build_prose_parent_skips_none_bboxes():
    from app.services.chunking import build_prose_parent
    els = [_el("paragraph", "no box here", bbox=None)]
    _, children = build_prose_parent("", els)
    assert children[0].bbox == []


def test_build_prose_parent_marks_children_as_text_element_type():
    from app.services.chunking import build_prose_parent
    _, children = build_prose_parent("", [_el("paragraph", "prose")])
    assert children[0].element_type == "text"


def test_build_prose_parent_child_inherits_page_and_confidence():
    from app.services.chunking import build_prose_parent
    el = _el("paragraph", "content", page=7, confidence=0.83)
    _, children = build_prose_parent("", [el])
    assert children[0].page == 7
    assert children[0].ocr_confidence == 0.83


def test_build_prose_parent_source_is_ocr_when_any_element_has_confidence():
    from app.services.chunking import build_prose_parent
    parent, _ = build_prose_parent("", [
        _el("paragraph", "native bit"),
        _el("paragraph", "scanned bit", confidence=0.7),
    ])
    assert parent.source == "ocr"


def test_build_prose_parent_source_is_native_without_confidence():
    from app.services.chunking import build_prose_parent
    parent, _ = build_prose_parent("", [_el("paragraph", "native only")])
    assert parent.source == "native"
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_chunking.py -v
```

Expected: FAIL — `cannot import name 'with_heading'`.

- [ ] **Step 3: Add `element_type` to `ChildChunk`**

In `backend/app/services/ingestion.py`, add this field to `ChildChunk`:

```python
    # "table" for chunks derived from a table element, "text" otherwise. None on
    # the legacy path. Lets the eval harness score table questions separately.
    element_type: Optional[str] = None
```

- [ ] **Step 4: Implement prose packing and parent building**

Append to `backend/app/services/chunking.py`:

```python
import tiktoken

from .ingestion import ChildChunk, ParentChunk, _child_splitter, _find_from

# Elements are joined with a blank line. The char-span arithmetic in
# _element_spans depends on this exact separator length — do not change one
# without the other.
ELEMENT_JOINER = "\n\n"

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """cl100k_base token count, matching the splitter's encoder so the parent
    budget here and the child budget there are measured the same way."""
    return len(_encoding.encode(text or ""))


def with_heading(heading_path: str, body: str) -> str:
    """Prepend the section path to a chunk body.

    The header goes *inside* content deliberately: it is then embedded and
    BM25-indexed automatically via the search_text generated column, so
    build_embedding_input() and reranker._rerank_text() need no change.
    """
    if not heading_path:
        return body
    return f"{heading_path}\n\n{body}"


@dataclass(frozen=True)
class ElementSpan:
    """Char [start, end) of one element within the joined body text."""
    start: int
    end: int
    element: Element


def _element_spans(elements: List[Element]) -> List[ElementSpan]:
    spans: List[ElementSpan] = []
    pos = 0
    for el in elements:
        spans.append(ElementSpan(pos, pos + len(el.text), el))
        pos += len(el.text) + len(ELEMENT_JOINER)
    return spans


def _rects_for_span(spans: List[ElementSpan], start: int, end: int) -> List[List[float]]:
    """Bboxes of every element whose char span overlaps [start, end).

    Per-element coarse attribution: a chunk highlights the regions of the
    elements it came from, not the exact glyph run.
    """
    return [
        s.element.bbox for s in spans
        if s.element.bbox and s.start < end and s.end > start
    ]


def _group_source(group: List[Element]) -> str:
    """"ocr" if any element in the group carries an OCR confidence, else
    "native". Keeps the existing source semantics on chunks."""
    return "ocr" if any(el.confidence is not None for el in group) else "native"


def _owning_element(spans: List[ElementSpan], offset: int):
    """The element a child starts inside, used for its page and confidence."""
    if offset >= 0:
        for s in spans:
            if s.start <= offset < s.end:
                return s.element
    return spans[0].element if spans else None


def pack_prose(elements: List[Element], max_tokens: int) -> List[List[Element]]:
    """Greedily group consecutive prose elements up to max_tokens.

    A single element larger than the budget becomes its own group rather than
    being dropped — the child splitter breaks it up afterwards.
    """
    groups: List[List[Element]] = []
    current: List[Element] = []
    current_tokens = 0
    for el in elements:
        tokens = count_tokens(el.text)
        if current and current_tokens + tokens > max_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(el)
        current_tokens += tokens
    if current:
        groups.append(current)
    return groups


def build_prose_parent(heading_path: str, group: List[Element]) -> tuple:
    """One prose parent plus its children.

    Children are split from the *body* and get the heading prefixed afterwards,
    so every child carries the header rather than only the first. Each child's
    bbox list comes from the elements its char span overlaps.
    """
    body = ELEMENT_JOINER.join(el.text for el in group)
    spans = _element_spans(group)
    pages = [el.page for el in group]
    source = _group_source(group)

    parent = ParentChunk(
        content=with_heading(heading_path, body),
        page_start=min(pages),
        page_end=max(pages),
        source=source,
    )

    children: List[ChildChunk] = []
    cursor = 0
    for piece in _child_splitter().split_text(body):
        offset = _find_from(body, piece, cursor)
        if offset >= 0:
            cursor = offset + 1  # children overlap, so advance minimally
        rects = _rects_for_span(spans, offset, offset + len(piece)) if offset >= 0 else []
        owner = _owning_element(spans, offset) or group[0]
        children.append(ChildChunk(
            content=with_heading(heading_path, piece),
            page=owner.page,
            source=source,
            ocr_confidence=owner.confidence,
            bbox=rects,
            element_type="text",
        ))
    return parent, children
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_chunking.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/chunking.py backend/app/services/ingestion.py backend/tests/test_chunking.py
git commit -m "feat: pack prose into parents with heading headers and element bboxes

Heading path is prefixed after splitting so every child carries it, which
also means it is embedded and BM25-indexed with no change to
build_embedding_input or reranker._rerank_text. Child bboxes come from the
elements their char span overlaps."
```

---

### Task 8: Atomic table chunks with row-group splitting

**Files:**
- Modify: `backend/app/services/chunking.py`
- Modify: `backend/tests/test_chunking.py`

**Interfaces:**
- Consumes: `with_heading`, `count_tokens`, `_group_source` (Task 7).
- Produces:
  - `chunking.split_markdown_table(markdown: str, rows_per_group: int) -> List[str]`
  - `chunking.build_table_parent(heading_path: str, element: Element) -> tuple`

Task 9 calls `build_table_parent`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_chunking.py`:

```python
# --- split_markdown_table --------------------------------------------------

_TABLE_HEAD = "| Region | Q1 | Q2 |\n|---|---|---|"


def _table_md(n_rows):
    rows = "\n".join(f"| R{i} | {i} | {i * 2} |" for i in range(n_rows))
    return f"{_TABLE_HEAD}\n{rows}"


def test_split_markdown_table_repeats_header_and_separator_in_every_group():
    from app.services.chunking import split_markdown_table
    groups = split_markdown_table(_table_md(25), rows_per_group=10)

    assert len(groups) == 3
    for g in groups:
        lines = g.splitlines()
        assert lines[0] == "| Region | Q1 | Q2 |"
        assert "---" in lines[1]


def test_split_markdown_table_distributes_all_data_rows_exactly_once():
    from app.services.chunking import split_markdown_table
    groups = split_markdown_table(_table_md(25), rows_per_group=10)
    data_lines = [
        line for g in groups for line in g.splitlines() if line.startswith("| R")
    ]
    assert len(data_lines) == 25
    assert len(set(data_lines)) == 25


def test_split_markdown_table_small_table_is_one_group():
    from app.services.chunking import split_markdown_table
    groups = split_markdown_table(_table_md(3), rows_per_group=10)
    assert groups == [_table_md(3)]


def test_split_markdown_table_degenerate_input_returns_one_opaque_group():
    """No separator line, or too few lines to have a header at all."""
    from app.services.chunking import split_markdown_table
    assert split_markdown_table("just one line of junk", rows_per_group=10) == \
        ["just one line of junk"]
    assert split_markdown_table("| a | b |", rows_per_group=10) == ["| a | b |"]


def test_split_markdown_table_empty_string_returns_empty_list():
    from app.services.chunking import split_markdown_table
    assert split_markdown_table("", rows_per_group=10) == []


# --- build_table_parent ----------------------------------------------------

def test_build_table_parent_small_table_yields_exactly_one_child(mocker):
    from app.services import chunking
    mocker.patch.object(chunking.settings, "table_max_tokens", 1500)
    el = _el("table", _table_md(3), page=2, bbox=[0.1, 0.2, 0.9, 0.5])

    parent, children = chunking.build_table_parent("A > B", el)

    assert len(children) == 1
    assert children[0].content.startswith("A > B\n\n")
    assert "| Region | Q1 | Q2 |" in children[0].content
    assert parent.content.startswith("A > B\n\n")


def test_build_table_parent_splits_oversized_table_into_row_groups(mocker):
    from app.services import chunking
    mocker.patch.object(chunking.settings, "table_max_tokens", 40)
    mocker.patch.object(chunking.settings, "table_row_group_rows", 5)
    el = _el("table", _table_md(30), page=1, bbox=[0.0, 0.0, 1.0, 1.0])

    _, children = chunking.build_table_parent("", el)

    assert len(children) == 6
    for child in children:
        assert "| Region | Q1 | Q2 |" in child.content


def test_build_table_parent_every_child_carries_the_whole_table_bbox():
    from app.services import chunking
    el = _el("table", _table_md(3), page=2, bbox=[0.1, 0.2, 0.9, 0.5])
    _, children = chunking.build_table_parent("", el)
    assert children[0].bbox == [[0.1, 0.2, 0.9, 0.5]]


def test_build_table_parent_marks_children_as_table_element_type():
    from app.services import chunking
    _, children = chunking.build_table_parent("", _el("table", _table_md(2)))
    assert all(c.element_type == "table" for c in children)


def test_build_table_parent_page_span_is_the_elements_page():
    from app.services import chunking
    parent, _ = chunking.build_table_parent("", _el("table", _table_md(2), page=9))
    assert (parent.page_start, parent.page_end) == (9, 9)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_chunking.py -v
```

Expected: FAIL — `cannot import name 'split_markdown_table'`.

- [ ] **Step 3: Implement table handling**

Append to `backend/app/services/chunking.py`:

```python
def split_markdown_table(markdown: str, rows_per_group: int) -> List[str]:
    """Split a markdown table into row groups, repeating the header row and its
    separator in each one.

    A table whose markdown is degenerate — fewer than three lines, or a missing
    `|---|` separator — is returned as a single opaque group rather than
    guessed at. Losing the row grouping is much better than mangling content.
    """
    if not markdown.strip():
        return []
    lines = [ln for ln in markdown.splitlines() if ln.strip()]
    if len(lines) < 3 or "---" not in lines[1]:
        return [markdown]

    header, separator, data = lines[0], lines[1], lines[2:]
    if len(data) <= rows_per_group:
        return [markdown]

    return [
        "\n".join([header, separator, *data[i:i + rows_per_group]])
        for i in range(0, len(data), rows_per_group)
    ]


def build_table_parent(heading_path: str, element: Element) -> tuple:
    """One atomic parent for a table, plus its children.

    The whole table is one parent so whole-table questions ("what is the
    total?") can be answered. Children are row groups only when the table
    exceeds table_max_tokens; a small table yields exactly one child equal to
    the whole table. A table is never split mid-row.

    Every child's bbox is the whole table region — per-element coarse
    attribution, so the preview highlights the table rather than a row.
    """
    source = _group_source([element])
    parent = ParentChunk(
        content=with_heading(heading_path, element.text),
        page_start=element.page,
        page_end=element.page,
        source=source,
    )

    if count_tokens(element.text) > settings.table_max_tokens:
        bodies = split_markdown_table(element.text, settings.table_row_group_rows)
    else:
        bodies = [element.text]

    rects = [element.bbox] if element.bbox else []
    children = [
        ChildChunk(
            content=with_heading(heading_path, body),
            page=element.page,
            source=source,
            ocr_confidence=element.confidence,
            bbox=rects,
            element_type="table",
        )
        for body in bodies
    ]
    return parent, children
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_chunking.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/chunking.py backend/tests/test_chunking.py
git commit -m "feat: emit tables as atomic markdown chunks

A table is one parent so whole-table reasoning works. Oversized tables
split by row groups with the header and separator repeated, never
mid-row. Degenerate markdown falls back to one opaque chunk."
```

---

### Task 9: Wire it together — `chunk_document`, model column, migration

**Files:**
- Modify: `backend/app/services/chunking.py`
- Modify: `backend/app/services/ingestion.py`
- Modify: `backend/app/models.py`
- Create: `backend/alembic/versions/0012_chunk_element_type.py`
- Modify: `backend/tests/test_chunking.py`

**Interfaces:**
- Consumes: everything from Tasks 6-8.
- Produces:
  - `chunking.chunk_elements(elements: List[Element]) -> tuple`
  - `ingestion.chunk_document(parsed)` — unchanged signature, routes to `chunk_elements` when `parsed.elements` is non-empty.
  - `ingestion._chunk_document_legacy(parsed)` — the previous implementation, renamed.
  - `DocumentChunk.element_type` column.

- [ ] **Step 1: Write the failing integration tests**

Append to `backend/tests/test_chunking.py`:

```python
# --- chunk_elements (end to end over an element list) ----------------------

def test_chunk_elements_never_merges_a_table_with_prose():
    from app.services.chunking import chunk_elements
    parents, _ = chunk_elements([
        _el("heading", "Financials", level=1),
        _el("paragraph", "Prose before the table."),
        _el("table", _table_md(3)),
        _el("paragraph", "Prose after the table."),
    ])

    assert len(parents) == 3
    contents = [p.content for p in parents]
    assert "Prose before" in contents[0] and "| Region" not in contents[0]
    assert "| Region" in contents[1]
    assert "Prose after" in contents[2] and "| Region" not in contents[2]


def test_chunk_elements_flushes_parent_at_a_heading_boundary():
    from app.services.chunking import chunk_elements
    parents, _ = chunk_elements([
        _el("heading", "A", level=1),
        _el("paragraph", "short a"),
        _el("heading", "B", level=1),
        _el("paragraph", "short b"),
    ])

    assert len(parents) == 2
    assert parents[0].content.startswith("A\n\n")
    assert parents[1].content.startswith("B\n\n")


def test_chunk_elements_children_align_with_parents():
    from app.services.chunking import chunk_elements
    parents, children = chunk_elements([
        _el("paragraph", "one"),
        _el("table", _table_md(2)),
    ])
    assert len(parents) == len(children)
    assert all(len(group) >= 1 for group in children)


def test_chunk_elements_empty_input_yields_nothing():
    from app.services.chunking import chunk_elements
    assert chunk_elements([]) == ([], [])


def test_chunk_elements_drops_noise_before_chunking():
    from app.services.chunking import chunk_elements
    parents, _ = chunk_elements([
        _el("page_header", "CONFIDENTIAL"),
        _el("paragraph", "real body"),
        _el("page_footer", "1 of 9"),
    ])
    assert len(parents) == 1
    assert "CONFIDENTIAL" not in parents[0].content
    assert "1 of 9" not in parents[0].content


def test_chunk_document_routes_to_layout_chunker_when_elements_present():
    from app.services.ingestion import ParsedDocument, PageContent, chunk_document
    parsed = ParsedDocument(
        pages=[PageContent(page=1, text="Body prose", source="ocr")],
        metadata={},
        elements=[
            _el("heading", "Section One", level=1),
            _el("paragraph", "Body prose"),
        ],
    )
    parents, _ = chunk_document(parsed)

    assert parents[0].content.startswith("Section One\n\n")


def test_chunk_document_falls_back_to_legacy_when_no_elements():
    from app.services.ingestion import ParsedDocument, PageContent, chunk_document
    parsed = ParsedDocument(
        pages=[PageContent(page=1, text="Legacy body text", source="native")],
        metadata={},
        elements=[],
    )
    parents, _ = chunk_document(parsed)

    assert len(parents) == 1
    assert parents[0].content == "Legacy body text"
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_chunking.py -v
```

Expected: FAIL — `cannot import name 'chunk_elements'`.

- [ ] **Step 3: Implement `chunk_elements`**

Append to `backend/app/services/chunking.py`:

```python
def chunk_elements(elements: List[Element]) -> tuple:
    """Chunk a document's typed elements into parents + children.

    Per section: prose accumulates into token-budgeted groups, and a table
    interrupts that accumulation to become its own parent. No parent crosses a
    heading boundary or splits a table.
    """
    parents: List[ParentChunk] = []
    children_per_parent: List[List[ChildChunk]] = []

    for section in split_sections(elements):
        pending: List[Element] = []

        def flush(pending=pending, heading_path=section.heading_path):
            for group in pack_prose(pending, settings.parent_max_tokens):
                parent, children = build_prose_parent(heading_path, group)
                parents.append(parent)
                children_per_parent.append(children)
            pending.clear()

        for el in section.elements:
            if el.type == "table":
                flush()
                parent, children = build_table_parent(section.heading_path, el)
                parents.append(parent)
                children_per_parent.append(children)
            else:
                # PROSE_TYPES plus any unknown-but-kept type: treat as prose
                # rather than discard content we do not recognize.
                pending.append(el)
        flush()

    n_children = sum(len(c) for c in children_per_parent)
    logger.info(
        "chunk_elements: elements=%d parents=%d children=%d",
        len(elements), len(parents), n_children,
    )
    return parents, children_per_parent
```

The default-argument binding on `flush` is deliberate: it captures this iteration's `pending` list and `heading_path` rather than closing over the loop variable, which would silently use the last section's heading for every flush.

- [ ] **Step 4: Route `chunk_document` to it**

In `backend/app/services/ingestion.py`, rename the existing `chunk_document` to `_chunk_document_legacy` (leave its body untouched), then add this new function directly after it:

```python
def chunk_document(parsed: ParsedDocument) -> Tuple[List[ParentChunk], List[List[ChildChunk]]]:
    """Chunk a parsed document.

    Uses layout-aware chunking when the parser returned typed elements; falls
    back to the legacy per-page line-based chunker when it did not (the
    docling_enabled=False rollback path).
    """
    if parsed.elements:
        from .chunking import chunk_elements  # local: chunking imports this module
        return chunk_elements(parsed.elements)
    return _chunk_document_legacy(parsed)
```

The local import is required — `chunking.py` imports `Element`/`ParentChunk`/`ChildChunk` from `ingestion.py`, so a module-level import here would be circular.

- [ ] **Step 5: Add the model column**

In `backend/app/models.py`, add to `DocumentChunk` after the `bbox` column:

```python
    # "table" for chunks derived from a table element, "text" otherwise. NULL on
    # pre-0012 rows. Lets the eval harness score table questions separately and
    # lets the UI render table chunks as markdown.
    element_type = Column(String, nullable=True)
```

- [ ] **Step 6: Persist it**

In `store_chunks` in `backend/app/services/ingestion.py`, add to the `DocumentChunk(...)` construction:

```python
                element_type=child.element_type,
```

- [ ] **Step 7: Write the migration**

Create `backend/alembic/versions/0012_chunk_element_type.py`:

```python
"""add element_type to document_chunks

Distinguishes table-derived chunks from prose so the eval harness can score
table questions separately and the UI can render markdown tables.

No backfill: layout-aware chunking changes every chunk boundary, so
`python -m scripts.reingest_all` is required after this migration regardless.
Existing rows keep element_type NULL, which reads as "unknown, pre-0012".

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-29
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        "document_chunks",
        sa.Column("element_type", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("document_chunks", "element_type")
```

- [ ] **Step 8: Run the full backend suite**

```bash
cd /d/development/chatbot/backend && python -m pytest -v
```

Expected: all PASS. `test_tasks.py`, `test_documents.py`, and `test_ingestion.py` all exercise ingestion — if any fail, that is a real integration break, so fix the code rather than the test, *unless* the test encodes the old chunking contract (one parent per page), in which case update the test and say so in the commit.

- [ ] **Step 9: Apply the migration**

```bash
cd /d/development/chatbot/backend && alembic upgrade head
```

- [ ] **Step 10: Commit**

```bash
git add backend/app/services/chunking.py backend/app/services/ingestion.py \
        backend/app/models.py backend/alembic/versions/0012_chunk_element_type.py \
        backend/tests/test_chunking.py
git commit -m "feat: route chunk_document through the layout-aware chunker

Tables never merge with prose, parents never cross a heading boundary, and
chunks record element_type. Legacy line-based chunker still serves the
docling_enabled=False path. Migration 0012 adds the column; no backfill,
reingest is required."
```

---

### Task 10: Worker — non-retryable timeout

**Files:**
- Modify: `backend/app/workers/tasks.py`
- Modify: `backend/tests/test_tasks.py`

**Interfaces:**
- Consumes: `ocr_client.ParseTimeout` (Task 4).
- Produces: no new API. Behaviour change only.

**Context:** `ingest_document` is decorated `@celery_app.task(bind=True, max_retries=1, default_retry_delay=10)` and its `except Exception` block calls `self.retry(exc=exc)`. `celery_app.py` sets **no** `task_time_limit`, so the default is unlimited and a 15-30 minute task will not be killed. Verify that in Step 1 rather than assuming it; do not add a limit.

- [ ] **Step 1: Verify there is no task time limit to fight**

```bash
cd /d/development/chatbot/backend && python -c "
from app.workers.celery_app import celery_app
print('task_time_limit      =', celery_app.conf.task_time_limit)
print('task_soft_time_limit =', celery_app.conf.task_soft_time_limit)
"
```

Expected: both `None`. If either is set, raise it above `parse_timeout_s` and say so in the commit message.

- [ ] **Step 2: Write the failing test**

Append to `backend/tests/test_tasks.py`:

```python
def test_ingest_document_does_not_retry_a_parse_timeout(db, mocker):
    """A parse that timed out once will time out again; retrying just burns
    another full timeout of worker time."""
    import uuid
    from app.models import Document
    from app.services.ocr_client import ParseTimeout
    from app.workers import tasks

    doc_id = str(uuid.uuid4())
    db.add(Document(
        id=doc_id, file_name="huge.pdf", file_path="/tmp/huge.pdf", status="pending",
    ))
    db.commit()

    mocker.patch.object(tasks, "SessionLocal", return_value=db)
    mocker.patch.object(tasks, "parse_document",
                        side_effect=ParseTimeout("parse exceeded 1800.0s"))
    retry = mocker.patch.object(tasks.ingest_document, "retry")

    tasks.ingest_document(doc_id)

    retry.assert_not_called()
    refreshed = db.query(Document).filter(Document.id == doc_id).first()
    assert refreshed.status == "failed"
    assert "1800" in refreshed.error_msg


def test_ingest_document_still_retries_other_failures(db, mocker):
    import uuid
    from celery.exceptions import MaxRetriesExceededError
    from app.models import Document
    from app.workers import tasks

    doc_id = str(uuid.uuid4())
    db.add(Document(
        id=doc_id, file_name="x.pdf", file_path="/tmp/x.pdf", status="pending",
    ))
    db.commit()

    mocker.patch.object(tasks, "SessionLocal", return_value=db)
    mocker.patch.object(tasks, "parse_document", side_effect=RuntimeError("transient"))
    retry = mocker.patch.object(
        tasks.ingest_document, "retry", side_effect=MaxRetriesExceededError(),
    )

    tasks.ingest_document(doc_id)

    retry.assert_called_once()
```

If `test_tasks.py` already has a helper for seeding a `Document` row, use it instead of the inline `db.add(...)` — match the file's existing style.

- [ ] **Step 3: Run to verify the first test fails**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_tasks.py -v -k timeout
```

Expected: FAIL — `retry` was called.

- [ ] **Step 4: Implement**

In `backend/app/workers/tasks.py`, add the import:

```python
from ..services.ocr_client import ParseTimeout
```

Then add a dedicated handler **after** `except Retry:` and **before** the existing `except Exception` block:

```python
    except ParseTimeout as exc:
        # Deliberately not retried: a parse that exceeded parse_timeout_s once
        # will exceed it again, and each attempt costs another full timeout of
        # worker time. Surface it and let the user retry explicitly.
        logger.error(
            "[task:%s] parse timed out, not retrying document_id=%s",
            self.request.id, document_id,
        )
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = "failed"
            doc.error_msg = str(exc)[:500]
            db.commit()
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
cd /d/development/chatbot/backend && python -m pytest tests/test_tasks.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/workers/tasks.py backend/tests/test_tasks.py
git commit -m "fix: do not retry a parse timeout

A parse that exceeded parse_timeout_s will exceed it again; retrying costs
another full timeout of worker time. Other failures still retry. Verified
celery sets no task_time_limit, so long parses are not killed."
```

---

### Task 11: Eval gate — baseline, golden set, comparison

**This task is the acceptance criterion.** The plan is not done when tests pass; it is done when the eval shows this was worth doing.

**Files:**
- Modify: `backend/evals/golden_set.yaml`
- Modify: `docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md` (append results)

**Interfaces:**
- Consumes: the fully wired pipeline from Tasks 1-10.
- Produces: a pass/fail decision on shipping.

**Ordering warning:** the baseline must reflect the **pre-change** pipeline. If Tasks 5-9 are already merged, get it by setting `DOCLING_ENABLED=false` — that is exactly what the rollback flag is for.

- [ ] **Step 1: Capture the baseline**

```bash
cd /d/development/chatbot/backend
# If Tasks 5-9 are already merged, force the legacy pipeline first:
#   echo "DOCLING_ENABLED=false" >> .env && python -m scripts.reingest_all
python -m evals.run_eval --name pre-docling
```

Record the score. If this command errors, fix the harness before continuing — a comparison without a baseline is worthless.

- [ ] **Step 2: Read the golden set to match its schema**

```bash
cd /d/development/chatbot/backend && head -40 evals/golden_set.yaml
```

Note the exact per-question field names (e.g. `question`, `expected`, `doc`) and whether a `tags` field exists. The next step must match what you see, not the illustrative shape below.

- [ ] **Step 3: Add table and cross-section questions**

Append to `backend/evals/golden_set.yaml`, adapting field names to Step 2 and referencing documents that actually exist in your eval corpus:

```yaml
# --- Table-structure questions (added 2026-07-29 with layout-aware chunking).
# The pre-existing questions cannot measure table understanding: before this
# change a table was linearized into unlabelled cell values, so any correct
# answer was luck. Tag these so their score can be read separately.
- question: "In the quarterly revenue table, what was the APAC figure for Q2?"
  expected: "The Q2 APAC revenue figure from the table"
  tags: [table]

- question: "Which region had the highest total across all quarters?"
  expected: "The region with the highest row total"
  tags: [table, aggregation]

- question: "How many rows does the revenue breakdown table contain?"
  expected: "The row count of the table"
  tags: [table]

# --- Cross-section questions. These test that the heading path reached the
# chunk: without it a chunk about revenue carries no evidence of which section
# it came from, so a question naming the section cannot retrieve it.
- question: "What does section 3.2 say about revenue?"
  expected: "The content of the 3.2 Revenue section"
  tags: [heading]

- question: "Summarize the Financials section."
  expected: "A summary drawn from the Financials section only"
  tags: [heading]
```

- [ ] **Step 4: Rebuild the index with the new pipeline**

```bash
cd /d/development/chatbot
# Remove the DOCLING_ENABLED=false line from backend/.env if Step 1 added it.
docker compose up -d --build ocr
cd backend && python -m scripts.reingest_all
```

Watch the worker log for `parse_document: ... elements=N` with a non-zero count. If `elements=0` for every document, `/parse` is returning nothing and the rest of this task is meaningless — debug that first.

- [ ] **Step 5: Run the eval and compare**

```bash
cd /d/development/chatbot/backend
python -m evals.run_eval --name docling-layout
python -m evals.run_eval --compare pre-docling docling-layout
```

- [ ] **Step 6: Record the result and decide**

Append to the spec:

```markdown
## Eval results (Task 11, YYYY-MM-DD)

| Question group | pre-docling | docling-layout |
|---|---|---|
| Existing questions | <score> | <score> |
| Table questions | n/a (new) | <score> |
| Heading questions | n/a (new) | <score> |

**Verdict:** <ship / investigate>
```

**Ship criterion:** no regression on the existing questions, plus a measurable gain on table questions.

**If tables improved but prose regressed:** that is the documented signal to revisit semantic chunking within long sections. Do not tune `rrf_weight_*` or `rerank_*` to paper over it — report the numbers and stop.

- [ ] **Step 7: Commit**

```bash
git add backend/evals/golden_set.yaml \
        docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md
git commit -m "test: add table and heading eval questions with measured results

The pre-existing golden set cannot measure table understanding, so it
could not score this change. Records the pre-docling baseline against
docling-layout."
```

---

### Task 12: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md`

**Interfaces:**
- Consumes: the shipped behaviour from Tasks 1-11.
- Produces: nothing consumed by code.

- [ ] **Step 1: Document the pipeline in `CLAUDE.md`**

Insert a new subsection immediately before the existing `### Reingestion after schema changes`:

```markdown
### Structure-aware parsing and layout-aware chunking

`ocr-service` exposes **`POST /parse`**, which takes a whole document (PDF,
DOCX, or image) and returns typed elements in reading order — `heading`,
`paragraph`, `list_item`, `table`, `caption`, `figure`, `page_header`,
`page_footer` — each with a bbox normalized to `[0,1]` of its page. Tables come
back as **markdown**. The contract lives in `ocr-service/wire.py`
(`schema_version: 1`); `ocr-service/parser.py` is the only module that imports
Docling. `POST /ocr` is the legacy per-page-image endpoint, kept as the
rollback path.

The worker no longer parses files: `ingestion.parse_document()` is one call to
`/parse`. It still builds `PageContent` per page, because the contextualizer
needs `parsed.text` and per-page text.

`services/chunking.py` chunks the elements. A heading stack produces
`heading_path` (`3. Financials > 3.2 Revenue`); prose packs into 1500-token
parents; **a table is one atomic parent** (row groups with the header repeated
only when it exceeds `table_max_tokens`). Two invariants: **no parent crosses a
heading boundary**, and **no chunk splits a table mid-row**.

`heading_path` is prefixed into `content` itself — not stored in its own
column. That is deliberate: it makes the header embed and BM25-index
automatically via the `search_text` generated column, so
`build_embedding_input()` and `reranker._rerank_text()` need no change. **Do not
"improve" this by moving it into its own field** without re-reading the lockstep
warning in the contextual-retrieval section above.

Chunk bboxes are **per-element and coarse**: a chunk carries the rects of the
elements it overlaps, so a table chunk highlights the whole table region rather
than a matched row.

Parse failures **fail the document** (`status="failed"`) rather than degrading
to structure-less chunks — mixing chunk qualities in one index is
undiagnosable. `ParseTimeout` is deliberately **not retried**.

Set `docling_enabled=False` to fall back to the legacy line-based path. That
fallback, `_chunk_document_legacy`, `_parse_pdf`/`_parse_docx`/`_parse_image`,
`_quad_to_norm_rect`, and `POST /ocr` are kept for **one release** and should
then be deleted.

Design: `docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md`
```

- [ ] **Step 2: Note the follow-ups in the spec**

Append to the spec's "Deferred" section:

```markdown
- **Delete the legacy path.** After one release: `_parse_pdf`, `_parse_docx`,
  `_parse_image`, `_chunk_document_legacy`, `_quad_to_norm_rect`, `LayoutLine`,
  `ocr-service POST /ocr`, `ocr_image`, `ocr_image_lines`, and the
  `docling_enabled` flag itself.
- **Render table chunks as markdown in the UI.** `element_type` is populated
  and consumed by the eval harness; the frontend still renders every citation
  as plain text.
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md
git commit -m "docs: document structure-aware parsing and layout-aware chunking

Records the /parse contract, the two chunking invariants, why heading_path
lives inside content, and the one-release expiry on the legacy path."
```

---

## Plan self-review

**Spec coverage.** Every spec section maps to a task: decisions 1-2 → Tasks 1, 3; decision 3 (bboxes) → Tasks 2, 7, 8; decision 4 (atomic tables) → Task 8; decision 5 (layout-aware) → Tasks 6-9; decision 6 (heading in content) → Task 7; decision 7 (fail loud) → Tasks 4, 5, 10. Wire format → Task 2. Synchronous-parse limit → Task 4 (`parse_timeout_s`) and Task 10. Data model → Task 9. Configuration → Task 4. Error-handling table → Tasks 3, 4, 8, 10. Rollback → Tasks 5, 9, 12. Testing → Tasks 2, 3, 6-9. Eval gate → Task 11. Build spike → Task 1.

**Two corrections to the spec, made deliberately here:**

1. The spec says `_quad_to_norm_rect` "is deleted". It cannot be — the legacy path that uses it survives for one release. Task 12 lists it for deletion at legacy-removal time instead.
2. The spec's operational fix "Celery's task time limit must clear `parse_timeout_s`" needs no change: `celery_app.py` sets no limit, so the default is unlimited. Task 10 Step 1 verifies this rather than editing config.

**Known gap, stated rather than hidden.** Task 11's golden-set questions reference a placeholder document ("the quarterly revenue table"). They must be rewritten against real documents in your eval corpus — Task 11 Step 2 forces reading the existing file first for exactly this reason. This is the one place the plan cannot be concrete without your data.

**Type consistency.** `Element` is defined in Task 5 and imported by `chunking.py` in Task 6. `ChildChunk.element_type` is added in Task 7 Step 3, before Task 8 sets it to `"table"` and Task 9 persists it. `ParseTimeout` is defined in Task 4 and consumed in Task 10. `to_wire` is defined in Task 2 and called in Task 3. `with_heading` / `count_tokens` / `_group_source` are defined in Task 7 and reused in Task 8. `split_sections` / `pack_prose` / `build_prose_parent` / `build_table_parent` all exist before `chunk_elements` calls them in Task 9. `_element_spans` and `_owning_element` are defined alongside their only caller in Task 7.
