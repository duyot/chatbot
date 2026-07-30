"""AI-pipeline trace logging: correlation ids + a JSONL event stream.

Two separate concerns live here:

1. **Correlation.** A chat request or ingestion task binds one `trace_id`
   (`bind_trace`) and every log record produced downstream carries it, including
   records from modules that know nothing about tracing. This works via a
   `ContextVar` plus a global `logging` record factory, so no function signature
   in the retrieval/rerank/LLM path had to grow a `trace_id` parameter.

   `ContextVar` values do **not** propagate into `ThreadPoolExecutor` workers.
   Code that fans out (see `services/contextualizer.py`) must submit through
   `contextvars.copy_context().run`, or its per-chunk events lose the id. Use
   `submit_with_trace()` for that.

2. **The trace stream.** `emit()` writes one JSON object per line to a
   dedicated `ai.trace` logger (its own rotating file, `propagate=False`) rather
   than into backend.log/worker.log. A single `generate_answer` prompt is
   10-20 KB; interleaving those with operational logs would make both useless,
   and the 10 MB rotation would hold minutes of traffic.

Verbosity is one setting, `ai_trace_level`:

    off      - emit nothing at all.
    summary  - ids, ranks, scores, token counts, latency; free text truncated
               to `ai_trace_text_chars`.
    full     - complete prompts, complete rerank candidates, complete
               responses. This writes document content to disk: opt-in only.

`emit()` must never raise. A logging call that breaks a chat response is worse
than a missing log line, so serialization falls back to `str` and any failure is
swallowed.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Iterator, Mapping

from .config import settings

logger = logging.getLogger(__name__)

# --- Correlation id ---------------------------------------------------------

NO_TRACE = "-"

TRACE_ID: ContextVar[str] = ContextVar("trace_id", default=NO_TRACE)

# `trace_id` is in every log format string, so every record must have the
# attribute. Set via a record factory rather than a handler filter: a record
# reaching a handler that lacks the filter would otherwise raise on formatting.
_FACTORY_INSTALLED = False


def new_trace_id() -> str:
    """Short random id. 12 hex chars is plenty to disambiguate concurrent runs
    and stays readable in a log line."""
    return uuid.uuid4().hex[:12]


def bind_trace(trace_id: str | None = None) -> str:
    """Bind a trace id to the current context and return it."""
    tid = trace_id or new_trace_id()
    TRACE_ID.set(tid)
    return tid


def current_trace_id() -> str:
    return TRACE_ID.get()


def install_record_factory() -> None:
    """Make every LogRecord carry `trace_id`. Idempotent."""
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous(*args, **kwargs)
        if not hasattr(record, "trace_id"):
            record.trace_id = TRACE_ID.get()
        return record

    logging.setLogRecordFactory(factory)
    _FACTORY_INSTALLED = True


def submit_with_trace(pool, fn: Callable, *args: Any, **kwargs: Any):
    """`pool.submit(fn, ...)` that carries the current context (and therefore
    the trace id) into the worker thread."""
    ctx = copy_context()
    return pool.submit(ctx.run, lambda: fn(*args, **kwargs))


# --- Verbosity level -------------------------------------------------------

LEVEL_OFF = "off"
LEVEL_SUMMARY = "summary"
LEVEL_FULL = "full"
_LEVELS = (LEVEL_OFF, LEVEL_SUMMARY, LEVEL_FULL)

_WARNED_BAD_LEVEL = False


def trace_level() -> str:
    """Normalized `settings.ai_trace_level`. An unrecognized value degrades to
    `summary` (warned once) rather than silently disabling tracing."""
    global _WARNED_BAD_LEVEL
    value = (settings.ai_trace_level or "").strip().lower()
    if value in _LEVELS:
        return value
    if not _WARNED_BAD_LEVEL:
        logger.warning(
            "ai_trace_level=%r is not one of %s; using %r",
            settings.ai_trace_level, _LEVELS, LEVEL_SUMMARY,
        )
        _WARNED_BAD_LEVEL = True
    return LEVEL_SUMMARY


def tracing_enabled() -> bool:
    return trace_level() != LEVEL_OFF


def full_payloads() -> bool:
    """True when complete prompts/documents/responses should be logged."""
    return trace_level() == LEVEL_FULL


# --- Payload shaping -------------------------------------------------------

_REDACT_HINTS = ("authorization", "apikey", "token", "secret", "password", "bearer")
_REDACTED = "***"


def _normalize_key(key: Any) -> str:
    """Lowercase and strip separators so "X-Api-Key", "api_key" and "apiKey"
    all match the same hint."""
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def trunc(text: Any, limit: int | None = None) -> Any:
    """Truncate free text for `summary` mode. Returns the value unchanged at
    `full`, and passes non-strings (None, numbers) straight through so call
    sites don't need to guard."""
    if not isinstance(text, str):
        return text
    if full_payloads():
        return text
    cap = settings.ai_trace_text_chars if limit is None else limit
    if cap < 0 or len(text) <= cap:
        return text
    return f"{text[:cap]}…(+{len(text) - cap} chars)"


def redact(mapping: Mapping[str, Any]) -> dict:
    """Copy of `mapping` with credential-ish values masked. Use on anything
    header- or settings-shaped before it goes near a log line."""
    out: dict = {}
    for key, value in mapping.items():
        normalized = _normalize_key(key)
        out[key] = (
            _REDACTED if any(h in normalized for h in _REDACT_HINTS) else value
        )
    return out


def chunk_summary(
    chunk: Any, score: float | None = None, rank: int | None = None
) -> dict:
    """Compact identity of a retrieved chunk. Attribute-tolerant: `rerank()` is
    called with plain chunk-likes in tests and evals."""
    row: dict = {
        "chunk_id": str(getattr(chunk, "id", "")) or None,
        "chunk_index": getattr(chunk, "chunk_index", None),
        "parent_id": str(getattr(chunk, "parent_id", "") or "") or None,
        "page": getattr(chunk, "page", None),
        "source": getattr(chunk, "source", None),
        "element_type": getattr(chunk, "element_type", None),
        "chars": len(getattr(chunk, "content", "") or ""),
    }
    if rank is not None:
        row["rank"] = rank
    if score is not None:
        row["score"] = round(float(score), 6)
    if full_payloads():
        row["content"] = getattr(chunk, "content", None)
        row["context"] = getattr(chunk, "context", None)
    return row


# --- Trace sink ------------------------------------------------------------

_TRACE_LOGGER_NAME = "ai.trace"
_configured_path: str | None = None


def configure_ai_trace(path: str | None = None) -> logging.Logger:
    """Attach a rotating JSONL handler to the `ai.trace` logger. Idempotent per
    path; call again with a different path to re-point it (tests do this)."""
    global _configured_path
    target = path or settings.ai_trace_file
    trace_logger = logging.getLogger(_TRACE_LOGGER_NAME)
    if _configured_path == target and trace_logger.handlers:
        return trace_logger

    for handler in list(trace_logger.handlers):
        trace_logger.removeHandler(handler)
        handler.close()

    directory = os.path.dirname(target)
    if directory:
        os.makedirs(directory, exist_ok=True)
    handler = RotatingFileHandler(
        target,
        maxBytes=settings.ai_trace_max_bytes,
        backupCount=settings.ai_trace_backups,
        encoding="utf-8",
    )
    # The record message IS the JSON document — no prefix, or `jq` chokes.
    handler.setFormatter(logging.Formatter("%(message)s"))
    trace_logger.addHandler(handler)
    trace_logger.setLevel(logging.INFO)
    # Keep 20 KB prompts out of backend.log / worker.log.
    trace_logger.propagate = False
    _configured_path = target
    return trace_logger


def _trace_logger() -> logging.Logger:
    trace_logger = logging.getLogger(_TRACE_LOGGER_NAME)
    if not trace_logger.handlers:
        # Scripts (reingest_all, render_pages_all) never call configure_*.
        configure_ai_trace()
    return trace_logger


def emit(event: str, only_full: bool = False, **fields: Any) -> None:
    """Write one trace event. Never raises.

    `only_full=True` marks high-volume detail (per-chunk, per-call) that is
    suppressed unless `ai_trace_level=full`.
    """
    level = trace_level()
    if level == LEVEL_OFF:
        return
    if only_full and level != LEVEL_FULL:
        return
    payload = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "trace_id": TRACE_ID.get(),
        "event": event,
    }
    payload.update(fields)
    try:
        line = json.dumps(payload, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - observability must not break the request
        try:
            line = json.dumps(
                {**{k: str(v) for k, v in payload.items()},
                 "trace_error": "serialize"},
                ensure_ascii=False,
            )
        except Exception:  # noqa: BLE001
            return
    try:
        _trace_logger().info(line)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def timed() -> Iterator[Callable[[], float]]:
    """Yield a callable returning elapsed milliseconds. Readable during and
    after the block, so it works for both success and failure paths."""
    start = time.perf_counter()
    yield lambda: round((time.perf_counter() - start) * 1000, 1)
