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

    # Retrieval tunables
    vector_top_k: int = 30
    fts_top_k: int = 30
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

    # Metadata-aware rerank boost (Phase 4). Defaults are no-ops so retrieval
    # behaviour is unchanged until tuned via the eval harness (see CLAUDE.md).
    # final_score = rerank_score + native_boost (if source==native)
    #                            - lowconf_penalty (if ocr conf < lowconf_threshold)
    rerank_native_boost: float = 0.0
    rerank_lowconf_penalty: float = 0.0
    rerank_lowconf_threshold: float = 0.5

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
