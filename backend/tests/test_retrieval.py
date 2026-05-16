import pytest
from unittest.mock import MagicMock, patch


def test_reranker_singleton_returns_same_instance():
    from app.services.rag.reranker import get_reranker
    a = get_reranker()
    b = get_reranker()
    assert a is b


def test_rerank_returns_top_n_sorted(mocker):
    from app.services.rag.reranker import rerank

    fake_ranker = MagicMock()
    fake_ranker.rerank.return_value = [
        {"id": "B", "score": 0.9},
        {"id": "A", "score": 0.5},
        {"id": "C", "score": 0.1},
    ]
    mocker.patch("app.services.rag.reranker.get_reranker", return_value=fake_ranker)

    chunk_a = MagicMock(id="A", content="cat")
    chunk_b = MagicMock(id="B", content="dog")
    chunk_c = MagicMock(id="C", content="fish")

    result = rerank("pets", [chunk_a, chunk_b, chunk_c], top_n=2)

    assert [c.id for c, s in result] == ["B", "A"]
    assert result[0][1] == 0.9


def test_rerank_empty_chunks_returns_empty():
    from app.services.rag.reranker import rerank
    assert rerank("q", [], top_n=5) == []
