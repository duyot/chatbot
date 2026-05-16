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


def test_grade_chunks_fast_path_yes(mocker):
    from app.services.rag.nodes import grade_chunks
    from app.services.rag.state import initial_state

    state = initial_state("d", "q")
    state["retrieved_children"] = [MagicMock()]
    state["rerank_scores"] = [0.5]  # above default 0.05

    out = asyncio.run(grade_chunks(state))
    assert out["graded_useful"] is True


def test_grade_chunks_fast_path_no(mocker):
    from app.services.rag.nodes import grade_chunks
    from app.services.rag.state import initial_state

    state = initial_state("d", "q")
    state["retrieved_children"] = [MagicMock()]
    state["rerank_scores"] = [0.001]  # below floor

    out = asyncio.run(grade_chunks(state))
    assert out["graded_useful"] is False
