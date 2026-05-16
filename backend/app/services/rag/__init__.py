"""rag package — public API surface. Phases 2-3 progressively replace the legacy
implementation. Until Phase 3 lands, agentic_rag_stream still comes from the
legacy module."""
from .._rag_legacy import agentic_rag_stream  # noqa: F401
