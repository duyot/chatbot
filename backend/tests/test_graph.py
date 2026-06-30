from unittest.mock import MagicMock


def test_build_citations_includes_page_and_source_and_dedups():
    from app.services.rag.graph import _build_citations
    c0 = MagicMock(chunk_index=0, page=1, source="native", content="alpha")
    c1 = MagicMock(chunk_index=1, page=3, source="ocr", content="beta")
    c_dup = MagicMock(chunk_index=0, page=1, source="native", content="alpha-dup")
    state = {"retrieved_children": [c0, c1, c_dup]}

    cites = _build_citations(state)

    assert len(cites) == 2  # deduped by chunk_index
    assert cites[0]["page"] == 1 and cites[0]["source"] == "native"
    assert cites[1]["page"] == 3 and cites[1]["source"] == "ocr"


def test_format_context_prefixes_page_number():
    from app.services.rag.nodes import _format_context
    p1 = MagicMock(page_start=2, content="hello")
    p2 = MagicMock(page_start=None, content="world")

    out = _format_context([p1, p2])

    assert "[page 2] hello" in out
    assert "world" in out
