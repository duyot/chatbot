import pytest
from unittest.mock import MagicMock


def test_get_reranker_returns_configured_model_name():
    from app.services.rag.reranker import get_reranker
    from app.config import settings
    assert get_reranker() == settings.reranker_model


def test_rerank_posts_pairs_to_ollama_and_sorts_by_score(mocker):
    from app.services.rag.reranker import rerank

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "embeddings": [
            [0.5],  # chunk_a (cat) vs query
            [0.9],  # chunk_b (dog) vs query
            [0.1],  # chunk_c (fish) vs query
        ]
    }
    fake_client_cm = MagicMock()
    fake_client_cm.__enter__.return_value.post.return_value = fake_response
    fake_client_cm.__exit__.return_value = None
    mocker.patch("app.services.rag.reranker.httpx.Client", return_value=fake_client_cm)

    chunk_a = MagicMock(id="A", content="cat")
    chunk_b = MagicMock(id="B", content="dog")
    chunk_c = MagicMock(id="C", content="fish")

    result = rerank("pets", [chunk_a, chunk_b, chunk_c], top_n=2)

    assert [c.id for c, _ in result] == ["B", "A"]
    assert result[0][1] == 0.9


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

    c1 = DocumentChunk(document_id=doc.id, parent_id=p1.id, chunk_index=0, content="C1", embedding=[0.0]*2560)
    c2 = DocumentChunk(document_id=doc.id, parent_id=p2.id, chunk_index=1, content="C2", embedding=[0.0]*2560)
    c3 = DocumentChunk(document_id=doc.id, parent_id=p1.id, chunk_index=2, content="C3", embedding=[0.0]*2560)
    db.add_all([c1, c2, c3])
    db.flush()

    # Children ordered [c1 (p1), c2 (p2), c3 (p1)] => expect parents [p1, p2]
    parents = fetch_parents(db, [c1, c2, c3])
    assert [p.id for p in parents] == [p1.id, p2.id]
