# Plan: Citation Bounding Boxes for Document Retrieval

**Source requirement**: `features_planning/doc_citiation/requirement.md`
**Complexity**: Large

## Summary
Today citations are text-only. This feature adds a "citation" icon button to each
assistant answer; clicking it draws the bounding box(es) of the answer's source
chunks onto the document shown alongside the chat. The OCR service already returns
per-line geometry, but `ocr_client` discards it and chunks store no coordinates —
so both the ingest-side geometry capture and the frontend overlay must be built.

## Decisions (confirmed)
- **Bbox mapping**: ingest-time. Capture per-line geometry, track char offsets
  during chunking, store normalized rects on `DocumentChunk.bbox`. Rects ride the
  existing `citations` SSE event; no new endpoint. Requires migration + reingest.
- **Rendering**: images-only overlay for the MVP (overlay on `<img>`). PDF stays an
  `<iframe>`; pdf.js overlay deferred to Phase 2.
- **Native-PDF geometry captured in Phase 1** (small `_parse_pdf` addition) so
  Phase 2 is frontend-only with no second reingest.

## Coordinate model
All boxes stored as normalized fractions `[x0,y0,x1,y1]` in `[0,1]` relative to page
width/height — render-size independent, unifies OCR-pixel / image-pixel / PDF-point
coordinates for the frontend overlay.

## Data shapes
- `DocumentChunk.bbox`: JSONB, nullable. `[[x0,y0,x1,y1], ...]` on the chunk's `page`.
  `[]`/`null` when unmappable (DOCX, no-match).
- Citation object: `{chunk_index, page, source, content, bbox}`.
- `highlightTarget` (frontend): `{documentId, page, rects: [[x0,y0,x1,y1], ...]}`.

## Phase 1 — Backend: capture & store geometry
1. `ocr_client.py`: add `ocr_image_lines()` returning structured lines + page dims.
2. `ingestion.py`: `LayoutLine`; `PageContent.{width,height,lines}`; native-PDF line
   geometry via PyMuPDF `get_text("dict")`; page text reconstructed from lines for
   exact offsets; `chunk_document` maps each child's char span -> covered line rects;
   `store_chunks` persists `bbox`.
3. `models.py`: `bbox = Column(JSONB, nullable=True)`; migration `0009_chunk_bbox.py`.
4. Unit tests for the offset->rect mapper (pure, deterministic).

## Phase 1 — Backend: surface + reingest
5. `graph.py:_build_citations()` adds `bbox`.
6. `python -m scripts.reingest_all` to backfill geometry.

## Phase 1 — Frontend: button, state, image overlay
7. `ChatMessage.jsx`: citation icon button in `chat-msg-actions`, shown when a
   citation has non-empty bbox; calls `onShowCitation(message)`.
8. Thread callback `ChatThread.jsx` -> `ChatPage.jsx`; ChatPage sets
   `selectedDocumentId` + `highlightTarget`.
9. `CitationOverlay.jsx` (+css): absolutely-positioned boxes = rect x displayed size.
10. `FilePreview.jsx`: wrap `<img>`, render overlay; phase-1 gate active only for
    image docs; PDFs keep the plain iframe.

## Phase 2 — deferred (frontend-only)
- `pdfjs-dist` canvas renderer replacing the PDF iframe; reuse `CitationOverlay`;
  scroll-to-page. No backend change / reingest.

## Validation
```bash
cd backend && pytest
cd backend && alembic upgrade head
cd backend && python -m scripts.reingest_all
npm run lint && npm run build
python -m evals.run_eval --name bbox_baseline
```

## Risks
| Risk | Likelihood | Mitigation |
|---|---|---|
| Native text reconstruction shifts chunk boundaries vs baseline | Med | eval validates retrieval unchanged; OCR path is exact |
| Requires full reingest | Med | documented workflow; historical messages simply lack boxes |
| PDF coord origin/rotation (Phase 2) | Med | normalize with PyMuPDF top-left coords; verify on rotated sample |
| DOCX has no geometry | Low | degrade: no button when no bbox |

## Acceptance
- [ ] Image doc: ask -> citation button -> boxes drawn on correct regions, scale on resize.
- [ ] `bbox` present in citations event + persisted; reloaded conversations highlight.
- [ ] DOCX / unmappable -> no button, no error.
- [ ] Retrieval eval unchanged vs baseline.
