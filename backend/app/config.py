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
    # LLM-as-reranker emits scores in 0..10; set above 0.0 to filter weak hits.
    rerank_score_floor: float = -1e9
    # Reranker model (LLM-as-reranker via OpenRouter). Defaults to the chat
    # model. Override to use a stronger model for ranking only.
    reranker_model: str = "anthropic/claude-haiku-4.5"

    # Agent loop tunables
    max_retrieval_retries: int = 2
    strict_grader: bool = False

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
