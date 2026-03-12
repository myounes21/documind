from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from config import settings
from elasticsearch import Elasticsearch

_VECTOR_SIZES = {
    "openai": settings.openai_embedding_dimensions,
    "cohere": settings.cohere_embedding_dimensions,
    "hf": settings.huggingface_embedding_dimensions,
}

_ES_MAPPING = {
    "mappings": {
        "properties": {
            "text":        {"type": "text"},
            "filename":    {"type": "keyword"},
            "filetype":    {"type": "keyword"},
            "page_number": {"type": "integer"},
            "is_parent":   {"type": "boolean"},
            "parent_id":   {"type": "keyword"},
            "chunk_id":    {"type": "keyword"},
        }
    }
}

qdrant_client = QdrantClient(settings.qdrant_host)
elastic_client = Elasticsearch(f"http://{settings.elasticsearch_host}:{settings.elasticsearch_port}")


def setup_qdrant() -> None:
    vector_size = _VECTOR_SIZES.get(settings.embedding_provider)

    if vector_size is None:
        raise ValueError(
            f"Unknown embedding provider: '{settings.embedding_provider}'. "
            f"Must be one of: {list(_VECTOR_SIZES.keys())}"
        )

    if not qdrant_client.collection_exists(settings.qdrant_collection_name):
        qdrant_client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


def setup_elasticsearch() -> None:
    if not elastic_client.indices.exists(index=settings.elasticsearch_index_name):
        elastic_client.indices.create(
            index=settings.elasticsearch_index_name,
            body=_ES_MAPPING
        )