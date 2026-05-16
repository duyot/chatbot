from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str = ""
    ollama_base_url: str = "http://host.docker.internal:11434/"
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
    # bge-reranker emits raw logits: positive = relevant, negative = not.
    # Tune after first eval if needed; the rerank log line shows top/bottom scores.
    rerank_score_floor: float = 0.0
    reranker_model: str = "qllama/bge-reranker-v2-m3:q8_0"
    # Reranker can live on a different Ollama instance than chat/embedding
    # (e.g., bigger GPU box). On Mac/Windows Docker Desktop the container
    # reaches the host's Ollama via host.docker.internal; on Linux either
    # use --add-host or set this to the docker bridge IP.
    reranker_base_url: str = "http://host.docker.internal:11434/"

    # Agent loop tunables
    max_retrieval_retries: int = 2
    strict_grader: bool = False

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
