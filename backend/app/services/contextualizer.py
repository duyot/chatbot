"""Generate a short context string situating each child chunk within its source
document, so the chunk embeds and indexes with the information it needs.

A chunk reading "the limit rises to 40% in the second year" is nearly useless in
isolation; prefixed with "Section 3 of the 2024 lease, on rent escalation" it is
findable. See the Anthropic contextual-embeddings cookbook and
docs/superpowers/specs/2026-07-28-contextual-retrieval-design.md.

Cost control rests on prompt caching: the document is sent once as a cached
block and re-read per chunk at ~10% of input price. Two properties of that are
load-bearing and easy to break silently:

1. The cached document block must come FIRST and the volatile chunk block
   SECOND. Caching is a prefix match.
2. The first call must complete BEFORE the rest fan out. A cache entry is only
   readable once the first response begins streaming, so a fully concurrent
   fan-out makes every call pay full input price — roughly 10x, with no error.

Failure is per-chunk and never fatal: a chunk that cannot be contextualized
gets None and is embedded on its content alone, matching how reranker.py and
ocr_client.py already degrade.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import tiktoken

from ..config import settings
from .ingestion import ChildChunk, ParsedDocument, _openai_client

logger = logging.getLogger(__name__)

TIER_FULL_DOC = "full_doc"
TIER_SUMMARY = "summary_plus_page"

DOCUMENT_CONTEXT_PROMPT = """<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""

SUMMARY_PROMPT = """<document>
{doc_content}
</document>

Summarize this document in at most 200 words. Focus on what it is, who it
concerns, and how it is organized, so the summary can help situate excerpts
taken from it. Answer only with the summary and nothing else."""

# Truncation guard for the summary tier: a document over the full-doc limit can
# still be enormous, and the summary call itself must not blow the window.
_SUMMARY_INPUT_CHAR_CAP = 400_000


def count_tokens(text: str) -> int:
    """Approximate token count via cl100k_base.

    This is OpenAI's tokenizer, not Claude's, and undercounts Claude tokens by
    roughly 15-20%. We use it because chat routes through OpenRouter, where the
    Anthropic count_tokens endpoint is unavailable. The full-doc threshold is
    set well below the context window to absorb the error.
    """
    if not text:
        return 0
    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _call_model(blocks: List[dict], max_tokens: int) -> str:
    """Single chat completion through OpenRouter. Raises on failure — callers
    decide whether that is fatal."""
    client = _openai_client()
    response = client.chat.completions.create(
        model=settings.contextualizer_model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[{"role": "user", "content": blocks}],
    )
    # A cache_control passthrough failure produces correct output at the
    # wrong cost (roughly 10x) rather than an error, so this is the only
    # signal that catches it happening in production. Never crash on a
    # missing/odd usage field — this is observability, not correctness.
    usage = getattr(response, "usage", None)
    if usage is not None:
        logger.info("_call_model: usage=%s", usage)
    return (response.choices[0].message.content or "").strip()


def _situate(doc_context: str, chunk_content: str) -> Optional[str]:
    """Generate one chunk's context. Returns None on any failure.

    Block order matters: the stable document first (cached), the volatile chunk
    second (uncached). Reversing them defeats the prefix match entirely.
    """
    blocks = [
        {
            "type": "text",
            "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc_context),
            "cache_control": {
                "type": "ephemeral",
                "ttl": settings.contextualizer_cache_ttl,
            },
        },
        {
            "type": "text",
            "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content),
        },
    ]
    try:
        text = _call_model(blocks, max_tokens=256)
    except Exception as exc:  # noqa: BLE001 - degrade, never fail the document
        logger.warning(
            "_situate: context generation failed, chunk will embed on content "
            "alone (%s)", exc,
        )
        return None
    return text or None


def _summarize_document(text: str) -> str:
    """Doc-level summary for the fallback tier. Returns "" on failure, which
    still leaves the child's own page as situating context."""
    blocks = [{
        "type": "text",
        "text": SUMMARY_PROMPT.format(doc_content=text[:_SUMMARY_INPUT_CHAR_CAP]),
    }]
    try:
        return _call_model(blocks, max_tokens=512)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_summarize_document: failed, falling back to page-only context (%s)", exc
        )
        return ""


def contextualize_with_stats(
    parsed: ParsedDocument,
    children_per_parent: List[List[ChildChunk]],
) -> Tuple[List[List[Optional[str]]], dict]:
    """Generate a context per child. Returns (contexts, stats).

    `contexts` mirrors the nesting of `children_per_parent` exactly, with None
    wherever generation failed. `stats` carries the tier used and how many
    children were successfully contextualized, for documents.doc_metadata.
    """
    total = sum(len(cs) for cs in children_per_parent)
    if total == 0:
        return [], {
            "tier": TIER_FULL_DOC,
            "contextualized_children": 0,
            "total_children": 0,
        }

    doc_text = parsed.text
    doc_tokens = count_tokens(doc_text)
    use_full_doc = doc_tokens <= settings.contextualizer_full_doc_token_limit
    tier = TIER_FULL_DOC if use_full_doc else TIER_SUMMARY

    if use_full_doc:
        page_text: dict = {}
        summary = ""
    else:
        page_text = {p.page: p.text for p in parsed.pages}
        summary = _summarize_document(doc_text)

    def doc_context_for(child: ChildChunk) -> str:
        if use_full_doc:
            return doc_text
        parts = []
        if summary:
            parts.append(f"Document summary:\n{summary}")
        page = page_text.get(child.page)
        if page:
            parts.append(f"Page {child.page} of the document:\n{page}")
        return "\n\n".join(parts)

    # Flatten to (parent_idx, child_idx, child) so results can be scattered back.
    flat = [
        (pi, ci, child)
        for pi, children in enumerate(children_per_parent)
        for ci, child in enumerate(children)
    ]

    logger.info(
        "contextualize: doc_tokens=%d tier=%s children=%d workers=%d",
        doc_tokens, tier, len(flat), settings.contextualizer_max_workers,
    )

    results: List[Optional[str]] = [None] * len(flat)

    # Warm the prompt cache with a single call before fanning out. Do NOT
    # collapse this into the pool: concurrent calls cannot read a cache entry
    # that is still being written, so all of them would pay full price.
    _, _, first_child = flat[0]
    results[0] = _situate(doc_context_for(first_child), first_child.content)

    if len(flat) > 1:
        with ThreadPoolExecutor(max_workers=settings.contextualizer_max_workers) as pool:
            futures = {
                pool.submit(_situate, doc_context_for(child), child.content): idx
                for idx, (_, _, child) in enumerate(flat)
                if idx > 0
            }
            for future, idx in futures.items():
                results[idx] = future.result()

    contexts: List[List[Optional[str]]] = [
        [None] * len(children) for children in children_per_parent
    ]
    for (pi, ci, _), ctx in zip(flat, results):
        contexts[pi][ci] = ctx

    succeeded = sum(1 for c in results if c)
    if succeeded < len(flat):
        logger.warning(
            "contextualize: %d/%d children failed contextualization",
            len(flat) - succeeded, len(flat),
        )
    logger.info("contextualize: done tier=%s ok=%d/%d", tier, succeeded, len(flat))

    return contexts, {
        "tier": tier,
        "contextualized_children": succeeded,
        "total_children": len(flat),
    }


def contextualize(
    parsed: ParsedDocument,
    children_per_parent: List[List[ChildChunk]],
) -> List[List[Optional[str]]]:
    """Convenience wrapper when the caller does not need the stats dict."""
    contexts, _ = contextualize_with_stats(parsed, children_per_parent)
    return contexts
