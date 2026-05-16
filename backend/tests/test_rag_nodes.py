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
