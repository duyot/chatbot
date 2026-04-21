from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str
    redis_url: str = "redis://redis:6379/0"
    openai_api_key: str = ""
    ollama_base_url: str = "http://172.22.6.30:30002"
    ollama_embedding_model: str = "qwen3-embedding:4b"
    upload_dir: str = "./uploads"
    max_upload_mb: int = 20
    cors_origins: str = "http://localhost:3000"

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
