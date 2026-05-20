from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str = ""
    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_embedding_model: str = "qwen3-embedding:4b"
    ollama_chat_model: str = "qwen3:4b-instruct-2507-q8_0"
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
    # Default is effectively disabled (-1e9) because reranker score ranges vary
    # wildly. With TEI + bge-reranker-v2-m3 and raw_scores=true we get raw
    # logits roughly in [-10, +10]; set to e.g. 0.0 only after observing the
    # grade(fast) log on real queries.
    rerank_score_floor: float = -1e9
    # Reranker is served by HuggingFace TEI (see docker-compose `reranker`
    # service). The model is configured on TEI's CLI (--model-id); this
    # setting is informational and shows up in logs only.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # In-compose service URL. Override to point at a remote TEI box.
    reranker_base_url: str = "http://reranker:80"

    # Agent loop tunables
    max_retrieval_retries: int = 2
    strict_grader: bool = False

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
