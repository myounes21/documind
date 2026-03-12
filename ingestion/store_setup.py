from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config import settings

_VECTOR_SIZES = {
    "openai": settings.openai_embedding_dimensions,
    "cohere": settings.cohere_embedding_dimensions,
    "hf": settings.huggingface_embedding_dimensions,
}

client = QdrantClient(settings.qdrant_host)

def setup_qdrant() -> None:
    vector_size = _VECTOR_SIZES.get(settings.embedding_provider)

    if vector_size is None:
        raise ValueError(
            f"Unknown embedding provider: '{settings.embedding_provider}'. "
            f"Must be one of: {list(_VECTOR_SIZES.keys())}"
        )

    if not client.collection_exists(settings.qdrant_collection_name):
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=vector_size,  # match your embedding model
                distance=Distance.COSINE
            )
        )

