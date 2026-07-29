"""Layout-aware chunking over typed elements from ocr-service /parse.

Split out of ingestion.py, which already covers parsing, embedding and
persistence. Everything here is a pure function over element lists — no DB, no
HTTP, no LLM — so it is cheap to test exhaustively.

Design: docs/superpowers/specs/2026-07-29-ocr-structure-extraction-design.md
"""
from __future__ import annotations

import logging
import tiktoken
from dataclasses import dataclass, field
from typing import List, Optional

from ..config import settings
from .ingestion import Element, ChildChunk, ParentChunk, _child_splitter, _find_from

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


# --- Prose packing and parent building -----------------------------------------------

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
    """'ocr' if any element in the group carries an OCR confidence, else
    'native'. Keeps the existing source semantics on chunks."""
    return "ocr" if any(el.confidence is not None for el in group) else "native"


def _owning_element(spans: List[ElementSpan], offset: int) -> tuple:
    """The element a child starts inside, used for its page and confidence.

    If the offset lands in a gap between elements (ELEMENT_JOINER space) or if
    the offset is not found, fall back to the nearest preceding element — the
    one immediately before the gap is almost certainly the correct owner.

    Returns (element, was_fallback: bool) so the caller can log with the actual
    child text when a fallback is taken."""
    if offset >= 0:
        for s in spans:
            if s.start <= offset < s.end:
                return s.element, False
        # Offset is in a gap or past all spans. Find the nearest preceding element.
        for i in range(len(spans) - 1, -1, -1):
            if spans[i].start <= offset:
                return spans[i].element, True
    # Offset is -1 (not found) or no spans. Fall back to first element.
    if spans:
        return spans[0].element, True
    return None, True


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
        owner, was_fallback = _owning_element(spans, offset)
        if owner is None:
            owner = group[0]
            was_fallback = True
        if was_fallback:
            logger.warning(
                "_owning_element fallback at offset %d, child text: %r, element id: %s",
                offset, piece[:40], owner.id
            )
        children.append(ChildChunk(
            content=with_heading(heading_path, piece),
            page=owner.page,
            source=source,
            ocr_confidence=owner.confidence,
            bbox=rects,
            element_type="text",
        ))
    return parent, children


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
