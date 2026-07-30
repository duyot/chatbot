# Structure-Aware OCR and Layout-Aware Chunking

**Date:** 2026-07-29
**Status:** Implemented (Tasks 1-10, plus an added Task 3b). **Task 11 (the eval
gate below) was deliberately skipped by the human** — no golden-set questions
were added, no baseline/post-change comparison was run, and the "Ship
criterion" was never measured. Treat this as shipped on tests-passing plus
human review, not on a passed eval.
**Supersedes parsing behaviour in:** `2026-04-20-file-upload-rag-ingestion-design.md`
**Builds on:** `2026-07-28-contextual-retrieval-design.md`

## Problem

`ocr-service` runs RapidOCR and returns a flat list of `{text, bbox, confidence}`
lines in detection reading order. Ingestion joins those lines with `\n` and splits
the result on character counts (1500-token parents, 300-token children).

Nothing in that pipeline knows what a heading or a table is. Consequences:

- A table is linearized into a wall of cell values with no column alignment, so
  the LLM has to guess which number belongs to which column.
- Chunk boundaries fall mid-table and mid-section, because the splitter only sees
  characters.
- Running headers and footers repeat into every chunk as noise.
- Section titles are not attached to the chunks beneath them, so a chunk about
  "Revenue" carries no evidence of which section it came from.

The corpus this runs against is **mostly scanned and table-heavy**, which makes
table structure the highest-value missing signal.

## Constraints

- Local deployment, **CPU only, no GPU**. Rules out VLM-based OCR (dots.ocr,
  olmOCR, MinerU).
- Ingestion is an async Celery job, so **5-10s per page is acceptable**.
- The document preview overlays `DocumentChunk.bbox` rects on server-rendered
  page images, so chunks must keep usable bboxes.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | Parse with **Docling** (DocLayNet layout + TableFormer) inside `ocr-service` | Best CPU-viable table accuracy; returns one unified document model with reading order, heading hierarchy, and per-item provenance bboxes already assembled |
| 2 | Service takes a **whole document**, not a page image | Docling then owns scan detection, reading order, and cross-page table continuation instead of us duplicating that logic |
| 3 | Bbox attribution is **per-element, coarse** | A table chunk highlights the whole table region. Simpler than cell-level mapping, and enough for the preview overlay |
| 4 | A table is an **atomic markdown chunk** prefixed with its heading path | Preserves whole-table reasoning ("what is the total?"); markdown gives the LLM explicit column alignment |
| 5 | Chunking is **layout-aware only**, no embedding-based semantic chunking | Typed elements provide better boundaries than cosine distance would guess, at zero API cost. Deferred, not rejected — see "Deferred" |
| 6 | Heading path is prefixed **into chunk content** | Gets embedded and BM25-indexed automatically, and requires no change to `build_embedding_input()` / `reranker._rerank_text()` |
| 7 | On parse failure, **fail the document loudly** | Automatic degradation would write two qualities of chunk into one index with no way to distinguish them |

### Rejected alternatives

- **RapidLayout + RapidTable inside the existing per-page service.** Stays in the
  ONNXRuntime family already deployed (no torch, slim image), but SLANet is weaker
  than TableFormer on borderless and merged-cell tables, and the element-assembly
  layer would be hand-written. Also blocked by a dependency conflict:
  `ocr-service` pins `numpy==1.26.4`, `rapid-layout` 1.2.1 requires `numpy>=2.0.0`.
  **Settled, not needed:** the Task 1 spike measured the Docling image at
  2.54 GB, under the 4 GB stop-and-ask threshold (see "Spike findings" below),
  so this fallback was never triggered.
- **Table detection with no layout model.** Cheapest, but without a `title` class
  there is no heading hierarchy, which is half the chunking win.
- **VLM OCR.** Requires a GPU.

## Architecture

### Flow

The worker stops parsing documents itself:

```
worker (tasks.py)
  └─ parse_document(path, name)          # now a thin client call
       └─ POST /parse  (multipart, whole file)  ──► ocr-service
                                                     ├─ Docling convert
                                                     ├─ DoclingDocument → wire format
                                                     └─ bbox normalize to [0,1]
       ◄── {schema_version, metadata, pages, elements}
  └─ chunk_document(parsed)              # layout-aware (services/chunking.py)
  └─ contextualize_with_stats(...)       # unchanged
  └─ embed_chunks / store_chunks         # unchanged
```

`_parse_pdf`, `_parse_docx`, and `_parse_image` collapse into the single `/parse`
call — Docling handles PDF, DOCX, and images natively.

Two things deliberately do **not** move:

- `services/page_images.render_document_pages()` keeps rasterizing pages with
  PyMuPDF in the backend. Preview rendering is a separate concern from parsing and
  must stay independently non-fatal.
- `ocr-service` stays **stateless**: bytes in, JSON out, nothing stored.

### Wire format

The service returns its own versioned schema, **not** a serialized
`DoclingDocument`. Docling releases fast (2.116.0 as pinned in
`ocr-service/requirements.txt`); its internal model shape must not become the
backend's contract.

```json
{
  "schema_version": 1,
  "metadata": {
    "page_count": 12, "ocr_pages": 12, "native_pages": 0,
    "engine": "docling", "mime_type": "application/pdf"
  },
  "pages": [
    {"page": 1, "width": 612.0, "height": 792.0,
     "source": "ocr", "ocr_confidence": 0.94}
  ],
  "elements": [
    {"id": "e0", "page": 1, "type": "heading", "level": 2,
     "text": "3.2 Revenue", "bbox": [0.08, 0.11, 0.62, 0.14],
     "confidence": 0.97},
    {"id": "e1", "page": 1, "type": "table",
     "text": "| Region | Q1 | Q2 |\n|---|---|---|\n| APAC | 12 | 15 |",
     "bbox": [0.08, 0.18, 0.94, 0.51], "confidence": 0.91}
  ]
}
```

Contract rules:

- `type` ∈ `heading | paragraph | list_item | table | caption | figure |
  page_header | page_footer`. Mapped from Docling item types by an explicit dict
  in the service; **anything unrecognized defaults to `paragraph`**, so a Docling
  upgrade that adds a type degrades instead of crashing.
- **Array order is reading order.** The chunker depends on this.
- `bbox` is normalized `[x0, y0, x1, y1]` in `[0,1]`, converted **in the service**
  from Docling's `prov` bbox using that page's dimensions, and clamped. The
  backend needs no width/height math; `_quad_to_norm_rect` is deleted.
- Table `text` is markdown, serialized in the service.
- `level` is present on `heading` only.
- `confidence` is `null` for native-text elements.

### Synchronous-parse limitation

A 100-page scan at 5-10s/page is a 10-15 minute synchronous HTTP request. This is
accepted for a single-user local deployment via a generous `parse_timeout_s`
(default 1800). **An async submit/poll endpoint is the correct fix if this ever
runs multi-user.** Recorded as a known limit, not built now.

## Layout-aware chunking

New module `backend/app/services/chunking.py`. `ingestion.py` is already 465 lines
covering parsing, chunking, embedding, and persistence; chunking is the part about
to get materially more complex. `chunk_document()` keeps its signature and
delegates.

### Heading stack

Walk elements in order maintaining a stack. On a `heading` of level *L*, truncate
the stack to *L−1* and push. `heading_path` is `" > ".join(stack)`, e.g.
`3. Financials > 3.2 Revenue`. Level jumps (h1 → h3) must not crash.

Drop rules, applied before grouping:

- `page_header` / `page_footer` → dropped (`drop_element_types`, configurable).
- `figure` with no caption → dropped.
- `figure` with a caption → the caption survives as prose.

### Parent construction

Accumulate consecutive prose elements (`paragraph`, `list_item`, `caption`),
joined with `\n\n`, until adding the next would exceed `parent_max_tokens` (1500,
`tiktoken` `cl100k_base`, matching the current splitter) → emit parent.

Two hard rules break the accumulator early:

1. **A heading boundary always flushes.** No parent spans two sections.
2. **A `table` is atomic.** Flush the prose accumulator, emit the table as its own
   parent, resume. A table over `table_max_tokens` splits by row groups
   (`table_row_group_rows`, default 10) — markdown-aware, so the header row **and
   its `|---|` separator** are repeated in every piece.

**The "one parent = exactly one page" invariant is dropped.** Parents may now span
pages; `page_start`/`page_end` take min/max of the contributing elements' pages.
Both columns already exist, so no migration is needed for this.

### Children

- **Prose parent** → the existing 300/50 `RecursiveCharacterTextSplitter`,
  unchanged.
- **Table parent** → children *are* the row groups (header repeated). A table is
  never split mid-row; a small table yields exactly one child.

### Contextual chunk headers

Parent and child content each get `{heading_path}\n\n{body}` prepended (omitted
when the stack is empty). Applied **after** splitting, so every child carries the
header rather than only the first.

This placement is load-bearing. It means:

- The header is embedded automatically.
- The header is BM25-indexed automatically via the `search_text` generated column
  (`coalesce(context,'') || ' ' || content`).
- `build_embedding_input()` and `reranker._rerank_text()` need **no change**,
  sidestepping the lockstep hazard documented in `CLAUDE.md`.

Cost is the header duplicated across sibling chunks — a few tokens each, which is
the right trade.

### Bboxes

Each parent tracks the char span of every element inside its body. A child's char
offset then maps to the elements it overlaps, and those elements' bboxes become
the child's `bbox` list. This is the existing `_line_spans` / `_rects_for_span`
mechanism generalized one granularity coarser, from lines to elements — the
helpers are reused, and the `bbox` JSON column is unchanged.

A table child's bbox is the whole table region (per decision 3).

## Data model

One nullable column:

```
document_chunks.element_type  TEXT NULL   -- "table" | "text"
```

This is **chunk-level**, not element-level: the eight wire-format element types
collapse to `"table"` for chunks derived from a `table` element and `"text"` for
everything else. `NULL` means a pre-migration chunk.

It lets the frontend render table chunks as markdown rather than preformatted
text, and lets the eval harness score table questions separately.

**No backfill.** Every chunk boundary changes, so
`python -m scripts.reingest_all` is mandatory after this lands regardless.

`heading_path` needs no column — it lives inside `content` by decision 6.

## Configuration

Added to `config.py`:

| Setting | Default | Purpose |
|---|---|---|
| `docling_enabled` | `True` | Manual rollback switch to the legacy path |
| `parse_timeout_s` | `1800` | Whole-document parse is minutes, not seconds |
| `drop_element_types` | `["page_header", "page_footer"]` | Running-noise removal |
| `parent_max_tokens` | `1500` | Matches current parent size |
| `table_max_tokens` | `1500` | Threshold for row-group splitting |
| `table_row_group_rows` | `10` | Rows per group when splitting |

**Correction (post-review):** `docling_ocr_backend` and `docling_table_mode`
were originally planned as backend-owned knobs but were never wired up —
parsing happens in `ocr-service`, which never reads backend settings.
`ocr-service/parser.py` hardcodes `RapidOcrOptions` and never sets
`table_structure_options.mode`; Docling 2.116.0's default already happens to
be `ACCURATE`. Both settings were removed from `config.py` rather than
plumbed through, since making them real knobs is new feature work, not this
fix.

## Error handling

**Policy: fail loud, never degrade silently.** On `OCRError` the document goes to
`status="failed"` with the message in `error_msg`. There is deliberately **no**
automatic fallback to structure-less parsing — mixing two chunk qualities in one
index with no way to distinguish them is worse than a visible failure the user
can retry. `docling_enabled` is a manual switch, not an automatic fallback.

| Case | Handling |
|---|---|
| Service down / 5xx / timeout | `OCRError` → doc `failed` |
| Docling raises on a corrupt file | Service returns 422 + message → surfaced in `error_msg` |
| Zero elements (blank scan) | Service returns 200 + empty list; worker's existing `if not parents: raise ValueError(...)` guard fires |
| `schema_version != 1` | Client rejects at the boundary with a clear `OCRError` |
| Unknown element `type` | Defaults to `paragraph` |
| Ragged/degenerate table markdown | Row-group splitter falls back to one opaque chunk instead of assuming a separator line exists |

Two operational fixes required as part of this work:

1. Celery's task time limit must clear `parse_timeout_s`. A 15-minute parse plus
   one retry is ~30 minutes of task life.
2. **Timeout must be non-retryable.** A parse that timed out once will time out
   again; other error classes still retry.

## Rollback

`docling_enabled=False` restores the legacy path. That requires keeping
`_parse_pdf` / `_parse_docx` / `_parse_image`, the line-based chunker, and the
service's old `POST /ocr` endpoint alive **for exactly one release** — deliberate
dead-ish code with an expiry date.

**Deletion trigger, revised.** The original criterion here was "delete once
the eval gate passes on real documents" — but Task 11 (that eval gate) was
deliberately skipped by the human and is not scheduled, so that condition
would never fire and the legacy path would sit dead-ish indefinitely. The
reachable trigger instead: **delete after one release during which
`docling_enabled` was never set to `False` in anger** (i.e. no production
rollback was needed). If anyone wants the quality evidence the original
criterion was meant to provide before deleting the fallback — a measured
comparison of table/prose retrieval before and after this change — running
the eval gate described under "Eval gate — the acceptance criterion" below is
the thing to do; it was never run, so that evidence does not currently exist.

## Testing

### `ocr-service` (new `tests/`, no tests exist today)

- Mapper unit tests over a fixture `DoclingDocument`: type mapping, reading order
  preserved, bbox normalized *and clamped* to `[0,1]`, unknown type → `paragraph`.
- Golden-file test for table → markdown serialization.
- **No model inference in the default run.** One real end-to-end smoke test on a
  tiny fixture PDF, marked `@pytest.mark.slow` and excluded by default, mirroring
  the existing `-m "not eval"` convention.

### `backend/tests/test_chunking.py`

Pure functions over synthetic element lists — no network, fast:

- Heading nesting, including level jumps (h1 → h3); `heading_path` string format.
- A heading boundary flushes the parent.
- Prose packing respects `parent_max_tokens`.
- A table never merges with prose and never splits mid-row.
- An oversized table repeats header **and** separator in every group.
- Degenerate table markdown does not crash.
- Drop rules: headers/footers removed; uncaptioned figure removed; captioned
  figure keeps its caption.
- `heading_path` present on **every** child, not only the first.
- A child spanning two elements collects both rects.
- A parent crossing a page boundary gets correct `page_start`/`page_end`.
- An empty document produces no parents, so the worker guard fires.

### `backend/tests/test_ocr_client.py`

Mocked-HTTP cases for each row of the error-handling table.

### Eval gate — the acceptance criterion

**Status: skipped, not passed.** The human deliberately chose not to run this
gate before shipping Tasks 1-10. Nothing below was executed — no golden-set
questions were added, no `pre-docling` baseline was captured, and no
`--compare` was run. Do not read the presence of this section as evidence the
ship criterion was met; it was not measured at all.

Tests prove the code does what it says; the eval proves it was worth doing.

The current `golden_set.yaml` predates tables having structure, so it cannot
measure what this builds. Therefore:

1. Add table-lookup and cross-section questions to
   `backend/evals/golden_set.yaml`.
2. Capture a baseline on the **current** pipeline before any change:
   `python -m evals.run_eval --name pre-docling`.
3. After implementation, reingest, then
   `python -m evals.run_eval --name docling-layout` and
   `python -m evals.run_eval --compare pre-docling docling-layout`.

**Ship criterion:** no regression on existing questions, plus a measurable gain on
table questions. If tables improve while prose regresses, that is the signal to
revisit semantic chunking within sections.

## Task 1: build spike (blocks everything else)

The current Dockerfile deliberately warms models at build so there is no runtime
download. **Docling fetches layout and TableFormer weights from HuggingFace on
first use**, so the image needs an explicit model-prefetch step at build or the
container is broken offline and the first parse pays a multi-hundred-MB download.

A throwaway container must answer four questions before any code lands:

1. Does `docling` install cleanly on `python:3.10-slim`?
2. Does the `[onnxruntime]` extra avoid pulling torch/torchvision? (torch arrives
   transitively via `docling-ibm-models` for TableFormer.)
3. Final image size. Current image is ~500MB; expect multi-GB with torch.
4. The exact model-prefetch command for the Dockerfile.

**If torch is unavoidable and the resulting image size is unacceptable, stop and
reconsider the RapidLayout + RapidTable alternative** documented under "Rejected
alternatives".

## Spike findings (Task 1, 2026-07-29)

All values measured, not estimated.

| Question | Answer |
|---|---|
| `docling` installs on `python:3.10-slim`? | Yes, clean. `docling==2.116.0` |
| Does `[onnxruntime]` avoid torch? | **No — and it is strictly worse.** Both `docling` (121 pkgs) and `docling[onnxruntime]` (126 pkgs) pull `torch` + `torchvision`; the extra additionally pulls `onnxruntime-gpu`, a GPU package, on a CPU-only host. Plain `docling` is used. |
| Resolved RapidOCR package | `rapidocr==3.9.2` (the v3 unified package), matching the bare `rapidocr` already pinned |
| Final image size | **2.54 GB** (was ~500 MB) — under the 4 GB stop-and-ask threshold |
| Model prefetch command | `docling-tools models download layout tableformer` → `Models downloaded into: /root/.cache/docling/models.` |
| Offline construction verified | **Yes.** `docker run --rm --network none ocr-spike python -c "…DocumentConverter()…"` prints `converter constructed offline OK`, no network reachout |

**Decision: proceed with Docling.** The RapidLayout + RapidTable fallback is
not needed.

**This plan was written against `docling==2.115.0`; the pin that actually
landed is `docling==2.116.0`, which required extra wiring the earlier
sections of this spec do not mention.** Anyone editing `ocr-service/parser.py`
needs these, and they are load-bearing, not incidental:

- An explicit `pipeline_options.artifacts_path` pointed at the
  `docling-tools`-baked weights — `DocumentConverter()` with no
  `artifacts_path` always tries HuggingFace Hub first, breaking the offline
  container.
- Explicit `RapidOcrOptions(det_model_path=..., cls_model_path=...,
  rec_model_path=...)` naming the bundled RapidOCR model files directly,
  because Docling's `RapidOcrModel` otherwise resolves *every* model
  (including RapidOCR's, which ships inside the `rapidocr` package, not under
  `artifacts_path`) relative to the same `artifacts_path`.
- `pipeline_options.generate_parsed_pages = True` — without it,
  `result.pages[*].cells` is always empty post-assembly and every page is
  misreported as OCR'd even when it has a native text layer.
- `BoundingBox.to_top_left_origin(page_height)` in place of a manual y-flip —
  Docling's provenance bbox carries its own `coord_origin`, so flip only via
  the method Docling ships rather than assuming bottom-left unconditionally.

### How the size was kept to 2.54 GB

torch is unavoidable — it arrives transitively via `docling-ibm-models` for
TableFormer regardless of extras. The default PyPI `torch` for Linux is the
**CUDA** build, which would have breached the threshold. Installing from the
CPU-only wheel index instead resolves `torch==2.13.0+cpu` /
`torchvision==0.28.0+cpu`, drops the package count from 121 to 102, and
eliminates all ~19 `nvidia-*` CUDA runtime libraries. The flag lives on line 1
of `ocr-service/requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

### Dependency corrections the spike forced

- **`onnxruntime` must be pinned explicitly.** RapidOCR v3 unbundled its
  inference engine: `rapidocr==3.9.2` resolves 17 packages and includes **no**
  ONNX runtime, and plain `docling` does not supply one either. Without it the
  Dockerfile's `RapidOCR()` warm-up has no engine to run on. Pinned
  `onnxruntime==1.23.2` (CPU build — this host has no GPU).
- **Test dependencies were missing entirely.** Tasks 2-3 run pytest with the
  `mocker` fixture, including inside the image, so `pytest==8.3.3`,
  `pytest-mock==3.14.0` and `httpx==0.28.1` are now pinned.
- **`numpy` 1.26.4 → 2.2.6 and `pillow` 10.4.0 → 12.3.0.** These pre-existing
  pins were raised to satisfy docling 2.116.0's constraints.
- **The `COPY` line names files that do not exist yet.** `wire.py` and
  `parser.py` arrive in Tasks 2 and 3, so the Dockerfile currently copies only
  `app.py` and carries a comment telling those tasks to extend the line. A
  gating task must not leave a Dockerfile that cannot build.

### Known wart

`ENV HF_HOME=/app/.cache/huggingface` is set, but `docling-tools` writes its
weights to `/root/.cache/docling/models` — a different location entirely. The
image works because both the prefetch and the runtime run as root. If this
service is ever changed to run as a non-root user, the baked weights become
unreadable and the container will silently start downloading at runtime.

## Deferred

- **Embedding-based semantic chunking within long prose sections.** Adds a paid
  embedding pass and a tuning knob (percentile threshold). Revisit only if the
  eval gate shows prose chunks still retrieve poorly.
- **Async submit/poll parse endpoint.** Needed only for multi-user deployment.
- **Cell-level bbox highlighting.** Per-element coarse bboxes are sufficient for
  the current preview overlay.
- **Cross-page table stitching verification.** Docling claims to handle it;
  measure before adding logic.
- **Delete the legacy path.** After one release: `_parse_pdf`, `_parse_docx`,
  `_parse_image`, `_chunk_document_legacy`, `_quad_to_norm_rect`, `LayoutLine`,
  `ocr-service POST /ocr`, `ocr_image`, `ocr_image_lines`, and the
  `docling_enabled` flag itself.
- **Render table chunks as markdown in the UI.** `element_type` is populated
  and consumed by the eval harness; the frontend still renders every citation
  as plain text.

## References

- [Docling](https://github.com/docling-project/docling) — DocLayNet + TableFormer
- [Docling supported formats](https://docling-project.github.io/docling/usage/supported_formats/)
- [RapidTable](https://github.com/RapidAI/RapidTable) / [rapid-layout](https://pypi.org/project/rapid-layout/) — fallback approach
- `features_planning/9.ocr_enhancement/Contextual_Chunk_Headers.md` — origin of
  the semantic-chunking request
