from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

ENV_PATH = Path(__file__).resolve().parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_PATH)

    groq_api_key: str
    openai_api_key: str
    cohere_api_key: str

    groq_llm_model: str = "llama-3.3-70b-versatile"
    app_port: int = 8000

    # Embedder
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    cohere_embedding_model: str = "embed-english-v3.0"
    cohere_embedding_dimensions: int = 1024

    huggingface_embedding_model: str = "BAAI/bge-base-en-v1.5"
    huggingface_embedding_dimensions: int = 768

    embedding_provider: Literal["openai", "cohere", "hf"] = "cohere"
    huggingface_batch_size: int = 256

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
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str = "localhost"
    langfuse_port: int = 3000

    # chunker configs
    parent_chunk_size: int = 1500
    child_chunk_size: int = 250


    app_env: str = "development"

settings = Settings()