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
