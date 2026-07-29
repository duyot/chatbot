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

    chunk_a = MagicMock(id="A", content="cat", context=None)
    chunk_b = MagicMock(id="B", content="dog", context=None)
    chunk_c = MagicMock(id="C", content="fish", context=None)

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

    chunk_a = MagicMock(id="A", content="x", context=None)
    chunk_b = MagicMock(id="B", content="y", context=None)

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

    chunk_a = MagicMock(id="A", content="x", context=None)
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

    chunks = [MagicMock(id=i, content=str(i), context=None) for i in range(3)]
    result = rerank("q", chunks, top_n=2)
    assert len(result) == 2
    assert [c.id for c, _ in result] == [1, 2]


def test_rerank_falls_back_to_input_order_on_api_error(mocker):
    """If the HTTP call raises, rerank returns chunks[:top_n] with score 0."""
    from app.services.rag.reranker import rerank

    _mock_rerank_response(mocker, body=None, raises=RuntimeError("openrouter down"))

    from app.services.rag.reranker import RERANK_FAILED_SCORE

    chunks = [MagicMock(id=i, content=str(i), context=None) for i in range(3)]
    result = rerank("q", chunks, top_n=2)
    assert [c.id for c, _ in result] == [0, 1]
    # Sentinel, not a real 0.0 relevance score, so a degraded rerank is
    # detectable downstream instead of looking like a genuine low score.
    assert all(s == RERANK_FAILED_SCORE for _, s in result)


def test_rerank_empty_chunks_returns_empty():
    from app.services.rag.reranker import rerank
    assert rerank("q", [], top_n=5) == []


def test_rerank_sends_context_with_content(mocker):
    """The cross-encoder must see the same contextual signal the retrieval arms
    did, or it re-ranks on strictly less information than recall used."""
    from app.services.rag.reranker import rerank

    patched = _mock_rerank_response(
        mocker, {"results": [{"index": 0, "relevance_score": 0.9}]}
    )

    with_ctx = MagicMock(id="A", content="the rate is 40%", context="Section 3.")
    without_ctx = MagicMock(id="B", content="bare", context=None)

    rerank("q", [with_ctx, without_ctx], top_n=2)

    body = patched.return_value.__enter__.return_value.post.call_args.kwargs["json"]
    assert body["documents"] == [
        {"text": "Section 3.\n\nthe rate is 40%"},
        {"text": "bare"},
    ]


def test_rerank_handles_chunks_without_context_attribute(mocker):
    """Defensive: rerank is also called with plain objects that only carry
    .id and .content."""
    from app.services.rag.reranker import rerank

    patched = _mock_rerank_response(
        mocker, {"results": [{"index": 0, "relevance_score": 0.5}]}
    )

    class Bare:
        id = "A"
        content = "text only"

    rerank("q", [Bare()], top_n=1)

    body = patched.return_value.__enter__.return_value.post.call_args.kwargs["json"]
    assert body["documents"] == [{"text": "text only"}]


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


def test_apply_metadata_boost_noop_by_default():
    from app.services.rag.retrieval import apply_metadata_boost
    a = MagicMock(source="ocr", ocr_confidence=0.1)
    b = MagicMock(source="native")
    reranked = [(a, 0.9), (b, 0.8)]
    out = apply_metadata_boost(reranked)
    assert [c for c, _ in out] == [a, b]  # unchanged with default 0.0 weights


def test_apply_metadata_boost_promotes_native(mocker):
    from app.services.rag import retrieval
    from app.services.rag.retrieval import apply_metadata_boost
    mocker.patch.object(retrieval.settings, "rerank_native_boost", 0.5)
    a = MagicMock(source="ocr", ocr_confidence=0.9)
    b = MagicMock(source="native")
    reranked = [(a, 0.9), (b, 0.8)]  # native b starts lower
    out = apply_metadata_boost(reranked)
    assert out[0][0] is b  # 0.8 + 0.5 = 1.3 beats 0.9


def test_apply_metadata_boost_penalizes_lowconf_ocr(mocker):
    from app.services.rag import retrieval
    from app.services.rag.retrieval import apply_metadata_boost
    mocker.patch.object(retrieval.settings, "rerank_lowconf_penalty", 0.5)
    mocker.patch.object(retrieval.settings, "rerank_lowconf_threshold", 0.5)
    a = MagicMock(source="ocr", ocr_confidence=0.1)  # below threshold -> penalized
    b = MagicMock(source="native")
    reranked = [(a, 0.9), (b, 0.6)]  # a starts higher, drops to 0.4
    out = apply_metadata_boost(reranked)
    assert out[0][0] is b


def test_hybrid_search_page_filter_restricts_results(db, mocker):
    from app.services.rag import retrieval
    from app.services.rag.retrieval import hybrid_search
    from app.models import Document, DocumentChunk
    mocker.patch.object(retrieval, "embed_text", return_value=[0.0] * 1536)

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.add_all([
        DocumentChunk(document_id=doc.id, chunk_index=0, content="alpha on page one",
                      embedding=[0.0] * 1536, page=1, source="native"),
        DocumentChunk(document_id=doc.id, chunk_index=1, content="alpha on page five",
                      embedding=[0.0] * 1536, page=5, source="native"),
    ])
    db.flush()

    vec_hits, fts_rows = hybrid_search(db, str(doc.id), "alpha", page_range=(1, 1))
    assert {c.page for c in vec_hits} == {1}


def test_bm25_available_false_when_setting_disabled(db, mocker):
    from app.services.rag import retrieval
    mocker.patch.object(retrieval.settings, "bm25_enabled", False)
    retrieval.reset_bm25_cache()
    assert retrieval.bm25_available(db) is False


def test_bm25_available_caches_after_first_probe(db, mocker):
    from app.services.rag import retrieval
    mocker.patch.object(retrieval.settings, "bm25_enabled", True)
    retrieval.reset_bm25_cache()

    spy = mocker.spy(db, "execute")
    first = retrieval.bm25_available(db)
    calls_after_first = spy.call_count
    second = retrieval.bm25_available(db)

    assert first == second
    assert spy.call_count == calls_after_first, "probe must run once per process"


def test_hybrid_search_uses_tsrank_when_bm25_unavailable(db, mocker):
    """The fallback must return real results, not an empty list — a dev without
    pg_search should still get working keyword recall."""
    from app.services.rag import retrieval
    from app.services.rag.retrieval import hybrid_search
    from app.models import Document, DocumentChunk

    mocker.patch.object(retrieval, "embed_text", return_value=[0.0] * 1536)
    mocker.patch.object(retrieval, "bm25_available", return_value=False)

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.add_all([
        DocumentChunk(document_id=doc.id, chunk_index=0, content="quarterly revenue grew",
                      embedding=[0.0] * 1536, page=1, source="native"),
        DocumentChunk(document_id=doc.id, chunk_index=1, content="unrelated boilerplate",
                      embedding=[0.0] * 1536, page=1, source="native"),
    ])
    db.flush()

    _, keyword_rows = hybrid_search(db, str(doc.id), "quarterly revenue")
    assert len(keyword_rows) >= 1


def test_tsrank_fallback_matches_on_context_too(db, mocker):
    """Even the fallback searches context + content, so contextual keyword
    recall does not depend on pg_search being installed."""
    from app.services.rag import retrieval
    from app.services.rag.retrieval import hybrid_search
    from app.models import Document, DocumentChunk

    mocker.patch.object(retrieval, "embed_text", return_value=[0.0] * 1536)
    mocker.patch.object(retrieval, "bm25_available", return_value=False)

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.add(DocumentChunk(
        document_id=doc.id, chunk_index=0, content="the rate is 40 percent",
        context="Section 3 on escalation clauses",
        embedding=[0.0] * 1536, page=1, source="native",
    ))
    db.flush()

    # "escalation" appears only in the context, never in content.
    _, keyword_rows = hybrid_search(db, str(doc.id), "escalation")
    assert len(keyword_rows) == 1


@pytest.mark.skipif(
    True,
    reason=(
        "integration test: exercises the real pg_search BM25 path; skipped "
        "by default so the suite does not depend on the ParadeDB image"
    ),
)
def test_bm25_search_matches_on_context(db, mocker):
    """Integration check for the real BM25 path. Flip the skipif to False and
    run with DATABASE_URL pointed at the ParadeDB container's chatbot_test DB."""
    from app.services.rag import retrieval
    from app.services.rag.retrieval import hybrid_search
    from app.models import Document, DocumentChunk

    mocker.patch.object(retrieval, "embed_text", return_value=[0.0] * 1536)
    retrieval.reset_bm25_cache()

    doc = Document(id=uuid.uuid4(), file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    db.add(DocumentChunk(
        document_id=doc.id, chunk_index=0, content="the rate is 40 percent",
        context="Section 3 on escalation clauses",
        embedding=[0.0] * 1536, page=1, source="native",
    ))
    db.flush()

    _, keyword_rows = hybrid_search(db, str(doc.id), "escalation")
    assert len(keyword_rows) == 1


def test_rrf_fuse_weights_vector_above_keyword():
    """At equal rank, a vector-only hit must outrank a keyword-only hit under
    the default 0.8/0.2 split."""
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    result = rrf_fuse([C("V")], [C("K")], k=60)
    ids = [cid for cid, _ in result]

    assert ids[0] == "V"
    scores = dict(result)
    assert scores["V"] == pytest.approx(0.8 / 60)
    assert scores["K"] == pytest.approx(0.2 / 60)


def test_rrf_fuse_explicit_weights_override_settings():
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    result = rrf_fuse([C("V")], [C("K")], k=60, w_vec=0.1, w_keyword=0.9)
    assert [cid for cid, _ in result][0] == "K"


def test_rerank_text_matches_embedding_input_format():
    """_rerank_text and build_embedding_input are deliberately separate
    implementations (reranking and embedding are different concerns), but they
    must format identically or retrieval and reranking would see different
    text. Nothing else catches drift — there is no eval harness."""
    from app.services.rag.reranker import _rerank_text
    from app.services.ingestion import build_embedding_input

    chunk = MagicMock(id="A", content="the rate is 40%", context="Section 3.")
    assert _rerank_text(chunk) == build_embedding_input(chunk.context, chunk.content)

    bare = MagicMock(id="B", content="no context here", context=None)
    assert _rerank_text(bare) == build_embedding_input(None, bare.content)


def test_rrf_fuse_scores_are_additive_across_arms():
    from app.services.rag.retrieval import rrf_fuse

    class C:
        def __init__(self, id):
            self.id = id

    # BOTH is rank 1 in each arm; VEC_ONLY is rank 0 in the vector arm only.
    vec = [C("VEC_ONLY"), C("BOTH")]
    kw = [C("KW_ONLY"), C("BOTH")]

    scores = dict(rrf_fuse(vec, kw, k=1))
    # BOTH: 0.8/2 + 0.2/2 = 0.5 ; VEC_ONLY: 0.8/1 = 0.8 ; KW_ONLY: 0.2/1 = 0.2
    assert scores["BOTH"] == pytest.approx(0.5)
    assert scores["VEC_ONLY"] == pytest.approx(0.8)
    assert scores["KW_ONLY"] == pytest.approx(0.2)
