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


def _owning_element(spans: List[ElementSpan], offset: int) -> Optional[Element]:
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
