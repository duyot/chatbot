from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str = ""
    # Embedding model is served via OpenRouter (env var name kept for
    # back-compat; value can be any OpenAI-compatible embedding model slug,
    # e.g. "qwen/qwen3-embedding-8b" on OpenRouter).
    openai_embedding_model: str = "qwen/qwen3-embedding-8b"
    # Output dimension to request. Must match the Vector(N) column dim in
    # document_chunks.embedding. Migration 0005 set N=1536. Models that
    # support OpenAI-style `dimensions` truncation (matryoshka) will honor
    # this; otherwise they return their native dim and you must run a new
    # migration to match.
    embedding_dim: int = 1536
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_chat_model: str = "anthropic/claude-haiku-4.5"
    upload_dir: str = "./uploads"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    # Auth / JWT. jwt_secret_key MUST be set in .env for production; an empty
    # value is only tolerable for local dev/tests.
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24

    # Retrieval tunables
    vector_top_k: int = 75
    # Name retained even though this arm is now BM25: it also sizes the
    # ts_rank fallback, and renaming would break existing .env files.
    fts_top_k: int = 75
    rrf_k: int = 60
    rerank_top_n: int = 6
    # Optional gate: chunks below this rerank score are treated as not-useful.
    # Cross-encoder rerankers via OpenRouter /v1/rerank emit relevance scores
    # roughly in [0, 1] (model-dependent); set e.g. 0.2 to filter weak hits.
    rerank_score_floor: float = -1e9
    # Reranker model (cross-encoder served via OpenRouter /v1/rerank).
    # Default is a chat model since it works without /v1/rerank availability;
    # override to a true reranker like nvidia/llama-nemotron-rerank-vl-1b-v2.
    reranker_model: str = "anthropic/claude-haiku-4.5"

    # --- Contextual embeddings (see docs/superpowers/specs/2026-07-28-contextual-retrieval-design.md)
    # Each child chunk gets an LLM-generated context string situating it within
    # its source document; that context is embedded and indexed alongside the
    # chunk text. Disable to reproduce the pre-contextual pipeline exactly.
    contextual_embeddings_enabled: bool = True
    contextualizer_model: str = "anthropic/claude-haiku-4.5"
    # Concurrent context-generation calls. The first call is always issued alone
    # to warm the prompt cache before fanning out.
    contextualizer_max_workers: int = 8
    # Documents above this token count fall back to
    # (doc summary + the child's own page) instead of the full document.
    # Measured with tiktoken cl100k_base, which undercounts Claude tokens by
    # ~15-20%, so this sits well under the 200k context window on purpose.
    contextualizer_full_doc_token_limit: int = 100_000
    # "1h" costs 2x on the cache write vs 1.25x for the 5-minute default, but
    # break-even is 3 reads and we get 100+ per document — and it stops a long
    # document from re-paying the write when a 5-minute entry expires mid-run.
    contextualizer_cache_ttl: str = "1h"

    # --- BM25 keyword search (ParadeDB pg_search)
    # Auto-detected at runtime: if pg_search is not installed, retrieval falls
    # back to the Postgres ts_rank query. Set False to force the fallback.
    bm25_enabled: bool = True
    # Weighted Reciprocal Rank Fusion. The guideline recommends 80/20
    # semantic/keyword; both are tunable.
    rrf_weight_vector: float = 0.8
    rrf_weight_keyword: float = 0.2

    # Agent loop tunables
    max_retrieval_retries: int = 2
    strict_grader: bool = False

    # OCR (PaddleOCR microservice). The worker stays lean and calls this over HTTP.
    ocr_enabled: bool = True
    ocr_service_url: str = "http://ocr:8080"
    ocr_timeout_s: float = 60.0
    # A PDF page whose native text layer has fewer than this many non-whitespace
    # characters is treated as scanned and sent to OCR. Images are always OCR'd.
    ocr_min_text_chars: int = 20
    # Render DPI for rasterizing scanned PDF pages before OCR.
    ocr_dpi: int = 200

    # --- Page images (document preview). Each PDF page is rasterized once into
    # uploads/pages/{document_id}/{page:04d}.{ext} and served to the UI as a
    # plain image, so the browser never parses the PDF itself. See
    # app/services/page_images.py.
    page_images_enabled: bool = True
    # 150 DPI is ~1240x1754 for A4 — sharper than the pdf.js viewer this
    # replaced (scale 1.5 ~= 108 DPI) at none of the client-side CPU cost.
    page_image_dpi: int = 150
    # "webp" (smallest at equal legibility), "jpg", or "png" (lossless, 3-5x
    # larger). Changing this orphans previously rendered images: the manifest
    # only lists files matching the current format, so previews re-render on
    # next request. Delete uploads/pages/ if you want the old ones gone.
    page_image_format: str = "webp"
    # Lossy quality for webp/jpg; ignored for png.
    page_image_quality: int = 80

    # Metadata-aware rerank boost (Phase 4). Defaults are no-ops so retrieval
    # behaviour is unchanged until tuned via the eval harness (see CLAUDE.md).
    # final_score = rerank_score + native_boost (if source==native)
    #                            - lowconf_penalty (if ocr conf < lowconf_threshold)
    rerank_native_boost: float = 0.0
    rerank_lowconf_penalty: float = 0.0
    rerank_lowconf_threshold: float = 0.5

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
