from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434/"
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
    rerank_score_floor: float = 0.05
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = "/tmp/flashrank"

    # Agent loop tunables
    max_retrieval_retries: int = 2
    strict_grader: bool = False

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
