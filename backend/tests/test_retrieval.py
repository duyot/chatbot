import pytest
from unittest.mock import MagicMock


def test_get_reranker_returns_configured_model_name():
    from app.services.rag.reranker import get_reranker
    from app.config import settings
    assert get_reranker() == settings.reranker_model


def _mock_rerank_response(mocker, body, raises=None):
    """Patch httpx.Client used in reranker module.

    If `raises` is provided, the .post call raises that exception (used to
    test the graceful fallback). Otherwise returns a response whose .json()
    yields `body`.
    """
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = body
    fake_client = MagicMock()
    if raises is not None:
        fake_client.post.side_effect = raises
    else:
        fake_client.post.return_value = fake_response
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_client
    fake_cm.__exit__.return_value = None
    return mocker.patch(
        "app.services.rag.reranker.httpx.Client", return_value=fake_cm
    )


def test_rerank_sorts_by_score(mocker):
    from app.services.rag.reranker import rerank

    # API returns sorted desc; the defensive sort in rerank() should be a no-op
    # but still validate it cope when the order is shuffled.
    patched = _mock_rerank_response(
        mocker,
        {"results": [
            {"index": 1, "relevance_score": 0.91},  # B (dog)
            {"index": 0, "relevance_score": 0.55},  # A (cat)
            {"index": 2, "relevance_score": 0.10},  # C (fish)
        ]},
    )

    chunk_a = MagicMock(id="A", content="cat")
    chunk_b = MagicMock(id="B", content="dog")
    chunk_c = MagicMock(id="C", content="fish")

    result = rerank("pets", [chunk_a, chunk_b, chunk_c], top_n=2)

    assert [c.id for c, _ in result] == ["B", "A"]
    assert result[0][1] == 0.91

    posted = patched.return_value.__enter__.return_value.post.call_args
    body = posted.kwargs["json"]
    assert body["query"] == "pets"
    assert body["documents"] == [{"text": "cat"}, {"text": "dog"}, {"text": "fish"}]
    assert body["top_n"] == 2


def test_rerank_handles_bare_list_response(mocker):
    """If a future variant returns a bare list instead of {results: [...]},
    rerank should still handle it."""
    from app.services.rag.reranker import rerank

    _mock_rerank_response(
        mocker,
        [
            {"index": 0, "relevance_score": 0.3},
            {"index": 1, "relevance_score": 0.8},
        ],
    )

    chunk_a = MagicMock(id="A", content="x")
    chunk_b = MagicMock(id="B", content="y")

    result = rerank("q", [chunk_a, chunk_b], top_n=2)
    assert [c.id for c, _ in result] == ["B", "A"]


def test_rerank_skips_out_of_range_index(mocker):
    from app.services.rag.reranker import rerank

    _mock_rerank_response(
        mocker,
        {"results": [
            {"index": 0, "relevance_score": 0.4},
            {"index": 99, "relevance_score": 0.9},  # bad index, must be skipped
        ]},
    )

    chunk_a = MagicMock(id="A", content="x")
    result = rerank("q", [chunk_a], top_n=5)

    assert [c.id for c, _ in result] == ["A"]
    assert result[0][1] == 0.4


def test_rerank_truncates_to_top_n(mocker):
    from app.services.rag.reranker import rerank

    _mock_rerank_response(
        mocker,
        {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.5},
            {"index": 0, "relevance_score": 0.1},
        ]},
    )

    chunks = [MagicMock(id=i, content=str(i)) for i in range(3)]
    result = rerank("q", chunks, top_n=2)
    assert len(result) == 2
    assert [c.id for c, _ in result] == [1, 2]


def test_rerank_falls_back_to_input_order_on_api_error(mocker):
    """If the HTTP call raises, rerank returns chunks[:top_n] with score 0."""
    from app.services.rag.reranker import rerank

    _mock_rerank_response(mocker, body=None, raises=RuntimeError("openrouter down"))

    chunks = [MagicMock(id=i, content=str(i)) for i in range(3)]
    result = rerank("q", chunks, top_n=2)
    assert [c.id for c, _ in result] == [0, 1]
    assert all(s == 0.0 for _, s in result)


def test_rerank_empty_chunks_returns_empty():
    from app.services.rag.reranker import rerank
    assert rerank("q", [], top_n=5) == []


import uuid


def test_rrf_fuse_combines_two_ranked_lists():
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    vec = [C("A"), C("B"), C("C")]  # ranks 0,1,2
    fts = [C("B"), C("A"), C("D")]  # ranks 0,1,2

    result = rrf_fuse(vec, fts, k=60)
    ids = [cid for cid, _ in result]

    # A and B both have RRF score 1/60 + 1/61; they should be above C and D.
    assert ids.index("A") < ids.index("C")
    assert ids.index("B") < ids.index("D")


def test_rrf_fuse_empty_legs():
    from app.services.rag.retrieval import rrf_fuse
    assert rrf_fuse([], [], k=60) == []


def test_fetch_parents_dedups_and_preserves_first_appearance(db):
    from app.services.rag.retrieval import fetch_parents
    from app.models import Document, DocumentParentChunk, DocumentChunk

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    p1 = DocumentParentChunk(document_id=doc.id, parent_index=0, content="P1")
    p2 = DocumentParentChunk(document_id=doc.id, parent_index=1, content="P2")
    db.add_all([p1, p2])
    db.flush()

    c1 = DocumentChunk(document_id=doc.id, parent_id=p1.id, chunk_index=0, content="C1", embedding=[0.0]*1536)
    c2 = DocumentChunk(document_id=doc.id, parent_id=p2.id, chunk_index=1, content="C2", embedding=[0.0]*1536)
    c3 = DocumentChunk(document_id=doc.id, parent_id=p1.id, chunk_index=2, content="C3", embedding=[0.0]*1536)
    db.add_all([c1, c2, c3])
    db.flush()

    # Children ordered [c1 (p1), c2 (p2), c3 (p1)] => expect parents [p1, p2]
    parents = fetch_parents(db, [c1, c2, c3])
    assert [p.id for p in parents] == [p1.id, p2.id]
