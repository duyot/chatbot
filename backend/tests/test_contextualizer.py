"""Unit tests for the contextualizer. No network calls: _situate,
_call_model, and _summarize_document are always patched."""
import pytest

from app.services.ingestion import ParsedDocument, PageContent, ChildChunk


def _parsed(*page_texts: str) -> ParsedDocument:
    return ParsedDocument(
        pages=[
            PageContent(page=i + 1, text=t, source="native")
            for i, t in enumerate(page_texts)
        ],
        metadata={"page_count": len(page_texts)},
    )


def _children(*specs: tuple[int, str]) -> list[list[ChildChunk]]:
    """specs is (page, content) per child; one parent per child for simplicity."""
    return [
        [ChildChunk(content=content, page=page, source="native")]
        for page, content in specs
    ]


def test_returns_same_nesting_shape_as_input(mocker):
    from app.services import contextualizer

    mocker.patch.object(contextualizer, "_situate", side_effect=lambda d, c: f"ctx:{c}")

    parsed = _parsed("page one text", "page two text")
    children = [
        [ChildChunk(content="a", page=1, source="native"),
         ChildChunk(content="b", page=1, source="native")],
        [ChildChunk(content="c", page=2, source="native")],
    ]

    result = contextualizer.contextualize(parsed, children)

    assert result == [["ctx:a", "ctx:b"], ["ctx:c"]]


def test_failed_chunk_yields_none_and_does_not_raise(mocker):
    from app.services import contextualizer

    def flaky(blocks, max_tokens):
        # The chunk text lives in the second content block.
        if "b" == _chunk_of(blocks):
            raise RuntimeError("openrouter 500")
        return f"ctx:{_chunk_of(blocks)}"

    mocker.patch.object(contextualizer, "_call_model", side_effect=flaky)

    parsed = _parsed("doc text")
    children = _children((1, "a"), (1, "b"), (1, "c"))

    result = contextualizer.contextualize(parsed, children)

    assert result == [["ctx:a"], [None], ["ctx:c"]]


def _chunk_of(blocks) -> str:
    """Pull the chunk content back out of the prompt blocks _situate built."""
    import re
    m = re.search(r"<chunk>\n(.*?)\n</chunk>", blocks[1]["text"], re.DOTALL)
    return m.group(1) if m else ""


def test_all_calls_failing_still_returns_full_shape(mocker):
    from app.services import contextualizer

    mocker.patch.object(contextualizer, "_call_model", side_effect=RuntimeError("down"))

    parsed = _parsed("doc text")
    children = _children((1, "a"), (1, "b"))

    result = contextualizer.contextualize(parsed, children)

    assert result == [[None], [None]]


def test_cache_is_warmed_before_fanout(mocker):
    """The first call must complete alone before the pool is created: a cache
    entry is only readable once the first response starts streaming, so a
    concurrent fan-out would make every call pay full input price."""
    from app.services import contextualizer

    events = []

    def spy_situate(doc_ctx, chunk):
        events.append(("situate", chunk))
        return f"ctx:{chunk}"

    real_pool = contextualizer.ThreadPoolExecutor

    def spy_pool(*args, **kwargs):
        events.append(("pool",))
        return real_pool(*args, **kwargs)

    mocker.patch.object(contextualizer, "_situate", side_effect=spy_situate)
    mocker.patch.object(contextualizer, "ThreadPoolExecutor", spy_pool)

    parsed = _parsed("doc text")
    children = _children((1, "a"), (1, "b"), (1, "c"))

    contextualizer.contextualize(parsed, children)

    assert events[0] == ("situate", "a"), "first chunk must be situated alone"
    assert events[1] == ("pool",), "pool must not be created until the warm call returns"


def test_single_child_document_creates_no_pool(mocker):
    from app.services import contextualizer

    events = []
    mocker.patch.object(contextualizer, "_situate", side_effect=lambda d, c: "ctx")
    mocker.patch.object(
        contextualizer, "ThreadPoolExecutor",
        side_effect=lambda *a, **k: events.append("pool"),
    )

    result = contextualizer.contextualize(_parsed("t"), _children((1, "only")))

    assert result == [["ctx"]]
    assert events == []


def test_full_doc_tier_passes_whole_document(mocker):
    from app.services import contextualizer

    seen = []
    mocker.patch.object(
        contextualizer, "_situate",
        side_effect=lambda d, c: seen.append(d) or "ctx",
    )

    parsed = _parsed("alpha page", "beta page")
    contextualizer.contextualize(parsed, _children((1, "a")))

    assert "alpha page" in seen[0]
    assert "beta page" in seen[0]


def test_oversized_doc_falls_back_to_summary_plus_page(mocker):
    from app.services import contextualizer

    mocker.patch.object(contextualizer.settings, "contextualizer_full_doc_token_limit", 5)
    mocker.patch.object(contextualizer, "_summarize_document", return_value="SUMMARY")

    seen = []
    mocker.patch.object(
        contextualizer, "_situate",
        side_effect=lambda d, c: seen.append(d) or "ctx",
    )

    parsed = _parsed("alpha " * 50, "beta page two")
    contextualizer.contextualize(parsed, _children((2, "chunk on page two")))

    assert "SUMMARY" in seen[0]
    assert "beta page two" in seen[0]
    assert "alpha" not in seen[0], "fallback must not include unrelated pages"


def test_stats_report_tier_and_success_count(mocker):
    from app.services import contextualizer

    def flaky(blocks, max_tokens):
        if _chunk_of(blocks) == "b":
            raise RuntimeError("boom")
        return "ctx"

    mocker.patch.object(contextualizer, "_call_model", side_effect=flaky)

    contexts, stats = contextualizer.contextualize_with_stats(
        _parsed("doc"), _children((1, "a"), (1, "b"), (1, "c"))
    )

    assert stats["tier"] == contextualizer.TIER_FULL_DOC
    assert stats["contextualized_children"] == 2
    assert stats["total_children"] == 3


def test_empty_children_returns_empty(mocker):
    from app.services import contextualizer
    mocker.patch.object(contextualizer, "_situate", side_effect=AssertionError("no calls"))
    assert contextualizer.contextualize(_parsed("t"), []) == []


def test_situate_sends_cache_control_on_document_block_only(mocker):
    """The document block carries cache_control and must come first; the
    volatile chunk block must follow it uncached, or the prefix never matches."""
    from app.services import contextualizer

    fake_client = mocker.MagicMock()
    fake_client.chat.completions.create.return_value = mocker.MagicMock(
        choices=[mocker.MagicMock(message=mocker.MagicMock(content="  the context  "))]
    )
    mocker.patch.object(contextualizer, "_openai_client", return_value=fake_client)

    out = contextualizer._situate("WHOLE DOC", "THE CHUNK")

    assert out == "the context"
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    blocks = kwargs["messages"][0]["content"]
    assert "WHOLE DOC" in blocks[0]["text"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "THE CHUNK" in blocks[1]["text"]
    assert "cache_control" not in blocks[1]


def test_count_tokens_is_monotonic():
    from app.services.contextualizer import count_tokens
    assert count_tokens("") == 0
    assert count_tokens("hello world") > 0
    assert count_tokens("hello world " * 100) > count_tokens("hello world")
