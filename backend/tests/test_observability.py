"""Unit tests for the AI trace plumbing.

These assert the guarantees the rest of the instrumentation relies on: `emit`
never raises, the level gate actually gates, credentials are masked, and the
trace id survives a thread-pool fan-out (the contextualizer's failure mode).
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest

from app import observability as obs


@pytest.fixture
def trace_file(tmp_path, mocker):
    """Point the ai.trace logger at a temp file and read events back."""
    path = tmp_path / "ai_trace.jsonl"
    mocker.patch.object(obs.settings, "ai_trace_level", "summary")
    mocker.patch.object(obs.settings, "ai_trace_text_chars", 10)
    obs.configure_ai_trace(str(path))

    def read():
        for handler in logging.getLogger("ai.trace").handlers:
            handler.flush()
        if not path.exists():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    yield read
    obs.configure_ai_trace(str(tmp_path / "unused.jsonl"))


def test_emit_writes_one_json_object_per_event(trace_file):
    obs.bind_trace("abc123")
    obs.emit("unit.test", n=2, note="hello")

    events = trace_file()
    assert len(events) == 1
    assert events[0]["event"] == "unit.test"
    assert events[0]["trace_id"] == "abc123"
    assert events[0]["n"] == 2
    assert "ts" in events[0]


def test_emit_is_silent_when_level_is_off(trace_file, mocker):
    mocker.patch.object(obs.settings, "ai_trace_level", "off")
    obs.emit("unit.test")
    assert trace_file() == []


def test_only_full_events_are_suppressed_at_summary(trace_file):
    obs.emit("unit.detail", only_full=True, x=1)
    obs.emit("unit.always", x=1)

    assert [e["event"] for e in trace_file()] == ["unit.always"]


def test_only_full_events_appear_at_full(trace_file, mocker):
    mocker.patch.object(obs.settings, "ai_trace_level", "full")
    obs.emit("unit.detail", only_full=True, x=1)

    assert [e["event"] for e in trace_file()] == ["unit.detail"]


def test_trunc_caps_text_at_summary_and_reports_the_remainder():
    out = obs.trunc("x" * 25, limit=10)
    assert out.startswith("x" * 10)
    assert "+15 chars" in out


def test_trunc_returns_full_text_at_full_level(mocker):
    mocker.patch.object(obs.settings, "ai_trace_level", "full")
    assert obs.trunc("x" * 25, limit=10) == "x" * 25


def test_trunc_passes_non_strings_through():
    assert obs.trunc(None) is None
    assert obs.trunc(7) == 7


def test_redact_masks_credentials_but_keeps_other_fields():
    out = obs.redact({
        "Authorization": "Bearer sk-secret",
        "X-Api-Key": "k",
        "Content-Type": "application/json",
    })
    assert out["Authorization"] == "***"
    assert out["X-Api-Key"] == "***"
    assert out["Content-Type"] == "application/json"


def test_emit_survives_unserializable_payloads(trace_file):
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    obs.emit("unit.weird", thing=Opaque())
    events = trace_file()
    assert len(events) == 1
    assert events[0]["thing"] == "<opaque>"


def test_invalid_level_degrades_to_summary(mocker):
    mocker.patch.object(obs.settings, "ai_trace_level", "verbose")
    assert obs.trace_level() == "summary"


def test_record_factory_puts_trace_id_on_every_record(caplog):
    obs.install_record_factory()
    obs.bind_trace("f00dcafe1234")
    with caplog.at_level(logging.INFO):
        logging.getLogger("unit.factory").info("hello")
    assert caplog.records[-1].trace_id == "f00dcafe1234"


def test_submit_with_trace_carries_the_id_into_worker_threads():
    """pool.submit() alone loses the ContextVar — that is the contextualizer's
    fan-out failure mode, so it is asserted rather than assumed."""
    obs.bind_trace("threadsafe01")
    with ThreadPoolExecutor(max_workers=2) as pool:
        with_helper = [
            obs.submit_with_trace(pool, obs.current_trace_id) for _ in range(4)
        ]
        plain = pool.submit(obs.current_trace_id)

    assert [f.result() for f in with_helper] == ["threadsafe01"] * 4
    assert plain.result() == obs.NO_TRACE


def test_rerank_emits_request_and_response_events(trace_file, mocker):
    """Wiring check: the reranker's instrumentation is easy to delete in a
    refactor, and nothing else would notice."""
    from types import SimpleNamespace

    from app.services.rag.reranker import rerank

    fake_response = SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
        json=lambda: {"results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]},
    )
    fake_client = mocker.MagicMock()
    fake_client.post.return_value = fake_response
    fake_cm = mocker.MagicMock()
    fake_cm.__enter__ = mocker.MagicMock(return_value=fake_client)
    fake_cm.__exit__ = mocker.MagicMock(return_value=False)
    mocker.patch("app.services.rag.reranker.httpx.Client", return_value=fake_cm)

    chunks = [
        SimpleNamespace(id="a", content="alpha", context=None),
        SimpleNamespace(id="b", content="beta", context=None),
    ]
    out = rerank("q", chunks, top_n=2)
    assert [c.id for c, _ in out] == ["b", "a"]

    events = {e["event"]: e for e in trace_file()}
    assert events["rerank.request"]["n_docs"] == 2
    # The bearer token must never reach the log file.
    assert events["rerank.request"]["headers"]["Authorization"] == "***"
    assert events["rerank.response"]["http_status"] == 200
    assert [r["score"] for r in events["rerank.response"]["results"]] == [0.9, 0.1]


def test_rerank_degradation_is_traced(trace_file, mocker):
    from types import SimpleNamespace

    from app.services.rag.reranker import RERANK_FAILED_SCORE, rerank

    mocker.patch(
        "app.services.rag.reranker.httpx.Client",
        side_effect=RuntimeError("connection refused"),
    )
    chunks = [SimpleNamespace(id="a", content="alpha", context=None)]
    out = rerank("q", chunks, top_n=1)
    assert out == [(chunks[0], RERANK_FAILED_SCORE)]

    events = {e["event"]: e for e in trace_file()}
    assert events["rerank.degraded"]["reason"] == "api_error"
    assert "connection refused" in events["rerank.degraded"]["error"]


def test_chunk_summary_tolerates_objects_without_chunk_attributes():
    class Bare:
        content = "hello"

    row = obs.chunk_summary(Bare(), score=0.5, rank=3)
    assert row["chars"] == 5
    assert row["rank"] == 3
    assert row["score"] == 0.5
    assert row["page"] is None
    # Content is only included at ai_trace_level=full.
    assert "content" not in row
