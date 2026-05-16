from unittest.mock import MagicMock, AsyncMock


def test_rewrite_query_strips_framing_and_sets_intent(mocker):
    from app.services.rag.nodes import rewrite_query
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    fake_llm.with_structured_output.return_value.ainvoke = AsyncMock(
        return_value=MagicMock(rewritten_query="Corporate Name", intent="lookup")
    )
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    import asyncio
    state = initial_state("doc1", "What is the Corporate Name?")
    out = asyncio.run(rewrite_query(state))

    assert out["rewritten_query"] == "Corporate Name"
    assert out["intent"] == "lookup"


import asyncio
from unittest.mock import MagicMock


def test_retrieve_and_rerank_writes_children_parents_scores(mocker, db):
    from app.models import Document, DocumentParentChunk, DocumentChunk
    from app.services.rag.nodes import retrieve_and_rerank
    from app.services.rag.state import initial_state
    import uuid

    doc_id = uuid.uuid4()
    doc = Document(id=doc_id, file_name="t.pdf", file_path="/tmp/t.pdf", status="done")
    db.add(doc)
    p = DocumentParentChunk(document_id=doc_id, parent_index=0, content="P")
    db.add(p); db.flush()
    c = DocumentChunk(document_id=doc_id, parent_id=p.id, chunk_index=0, content="C",
                      embedding=[0.0] * 2560)
    db.add(c); db.flush()

    mocker.patch(
        "app.services.rag.nodes.retrieve",
        return_value=([c], [p], [0.8]),
    )
    state = initial_state(str(doc_id), "q")
    state["rewritten_query"] = "rewritten"
    out = asyncio.run(retrieve_and_rerank(state, db))

    assert out["retrieved_children"] == [c]
    assert out["parents"] == [p]
    assert out["rerank_scores"] == [0.8]
    assert "rewritten" in out["attempted_queries"]


def test_grade_chunks_fast_path_useful_when_chunks_present(mocker):
    from app.services.rag.nodes import grade_chunks
    from app.services.rag.state import initial_state

    state = initial_state("d", "q")
    state["retrieved_children"] = [MagicMock()]
    state["rerank_scores"] = [-3.5]  # negative score still useful by default

    out = asyncio.run(grade_chunks(state))
    assert out["graded_useful"] is True


def test_grade_chunks_fast_path_not_useful_when_no_chunks(mocker):
    from app.services.rag.nodes import grade_chunks
    from app.services.rag.state import initial_state

    state = initial_state("d", "q")
    state["retrieved_children"] = []
    state["rerank_scores"] = []

    out = asyncio.run(grade_chunks(state))
    assert out["graded_useful"] is False


def test_grade_chunks_fast_path_honors_explicit_floor(mocker):
    from app.services.rag.nodes import grade_chunks
    from app.services.rag.state import initial_state

    mocker.patch("app.services.rag.nodes.settings.rerank_score_floor", 0.5)
    state = initial_state("d", "q")
    state["retrieved_children"] = [MagicMock()]
    state["rerank_scores"] = [0.1]  # below the override floor

    out = asyncio.run(grade_chunks(state))
    assert out["graded_useful"] is False


def test_rewrite_and_retry_produces_new_query(mocker):
    from app.services.rag.nodes import rewrite_and_retry
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    msg = MagicMock(); msg.content = "alternative phrasing"
    fake_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    state = initial_state("d", "q")
    state["attempted_queries"] = ["first", "second"]
    state["retry_count"] = 0

    out = asyncio.run(rewrite_and_retry(state))
    assert out["rewritten_query"] == "alternative phrasing"
    assert out["retry_count"] == 1


def test_faithfulness_check_yes_emits_no_warning(mocker):
    from app.services.rag.nodes import faithfulness_check
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    msg = MagicMock(); msg.content = "YES"
    fake_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    p = MagicMock(); p.content = "context"
    state = initial_state("d", "q")
    state["parents"] = [p]
    state["answer"] = "answer"
    out = asyncio.run(faithfulness_check(state))
    assert out["warnings"] == []


def test_faithfulness_check_no_appends_warning(mocker):
    from app.services.rag.nodes import faithfulness_check
    from app.services.rag.state import initial_state

    fake_llm = MagicMock()
    msg = MagicMock(); msg.content = "NO"
    fake_llm.ainvoke = mocker.AsyncMock(return_value=msg)
    mocker.patch("app.services.rag.nodes._chat_llm", return_value=fake_llm)

    p = MagicMock(); p.content = "context"
    state = initial_state("d", "q")
    state["parents"] = [p]
    state["answer"] = "answer"
    out = asyncio.run(faithfulness_check(state))
    assert len(out["warnings"]) == 1
    assert "warning" in out["warnings"][0]["type"]
