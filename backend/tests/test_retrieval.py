import pytest
from unittest.mock import MagicMock


def test_get_reranker_returns_configured_model_name():
    from app.services.rag.reranker import get_reranker
    from app.config import settings
    assert get_reranker() == settings.reranker_model


def _mock_llm(mocker, ranked):
    """Patch reranker._build_llm so .with_structured_output().invoke() returns
    a _RankResult built from `ranked` (list of {"index", "score"} dicts)."""
    from app.services.rag.reranker import _RankResult, _RankedItem

    fake_structured = MagicMock()
    fake_structured.invoke.return_value = _RankResult(
        scores=[_RankedItem(**r) for r in ranked]
    )
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    return mocker.patch(
        "app.services.rag.reranker._build_llm", return_value=fake_llm
    )


def test_rerank_sorts_by_score(mocker):
    from app.services.rag.reranker import rerank

    _mock_llm(
        mocker,
        [
            {"index": 1, "score": 9.0},  # chunk_b (dog)
            {"index": 0, "score": 5.0},  # chunk_a (cat)
            {"index": 2, "score": 1.0},  # chunk_c (fish)
        ],
    )

    chunk_a = MagicMock(id="A", content="cat")
    chunk_b = MagicMock(id="B", content="dog")
    chunk_c = MagicMock(id="C", content="fish")

    result = rerank("pets", [chunk_a, chunk_b, chunk_c], top_n=2)

    assert [c.id for c, _ in result] == ["B", "A"]
    assert result[0][1] == 9.0


def test_rerank_appends_missing_indices_with_zero(mocker):
    """If the LLM omits some indices, rerank fills them in with score 0.0."""
    from app.services.rag.reranker import rerank

    _mock_llm(mocker, [{"index": 0, "score": 8.0}])  # omits index 1

    chunk_a = MagicMock(id="A", content="x")
    chunk_b = MagicMock(id="B", content="y")

    result = rerank("q", [chunk_a, chunk_b], top_n=2)
    assert [c.id for c, _ in result] == ["A", "B"]
    assert result[0][1] == 8.0
    assert result[1][1] == 0.0


def test_rerank_skips_out_of_range_index(mocker):
    from app.services.rag.reranker import rerank

    _mock_llm(
        mocker,
        [
            {"index": 0, "score": 4.0},
            {"index": 99, "score": 9.0},  # bad index, must be skipped
        ],
    )

    chunk_a = MagicMock(id="A", content="x")
    result = rerank("q", [chunk_a], top_n=5)

    assert [c.id for c, _ in result] == ["A"]
    assert result[0][1] == 4.0


def test_rerank_truncates_to_top_n(mocker):
    from app.services.rag.reranker import rerank

    _mock_llm(
        mocker,
        [
            {"index": 0, "score": 1.0},
            {"index": 1, "score": 9.0},
            {"index": 2, "score": 5.0},
        ],
    )

    chunks = [MagicMock(id=i, content=str(i)) for i in range(3)]
    result = rerank("q", chunks, top_n=2)
    assert len(result) == 2
    assert [c.id for c, _ in result] == [1, 2]


def test_rerank_falls_back_to_input_order_on_llm_error(mocker):
    """If the LLM call raises, rerank returns chunks[:top_n] with score 0."""
    from app.services.rag.reranker import rerank

    fake_structured = MagicMock()
    fake_structured.invoke.side_effect = RuntimeError("openrouter down")
    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value = fake_structured
    mocker.patch("app.services.rag.reranker._build_llm", return_value=fake_llm)

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
