from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    groq_api_key: str
    openai_api_key: str = ""

    groq_model: str = "llama-3.3-70b-versatile"
    app_port: int = 8000

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Elasticsearch
    elasticsearch_host: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_password: str = ""

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "localhost"
    langfuse_port: int = 3000

    # chunker configs
    parent_tokens_min: int = 500
    parent_tokens_max: int = 1000
    child_tokens_min: int = 100
    child_tokens_max: int = 200


    app_env: str = "development"

settings = Settings()