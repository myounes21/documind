from functools import cache
from typing import Literal

from config import settings
from schemas import Chunk


@cache
def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=settings.openai_api_key)


@cache
def _get_cohere_client():
    import cohere
    return cohere.Client(api_key=settings.cohere_api_key)


@cache
def _get_huggingface_client():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.huggingface_embedding_model)


# OpenAI
def _openai_embed_texts(texts: list[str]) -> list[list[float]]:
    response = _get_openai_client().embeddings.create(
        model=settings.openai_embedding_model,
        input=texts
    )
    return [item.embedding for item in response.data]


# Cohere
CohereInputType = Literal["search_document", "search_query"]

def _cohere_embed_texts(texts: list[str], input_type: CohereInputType = "search_document") -> list[list[float]]:
    response = _get_cohere_client().embed(
        texts=texts,
        model=settings.cohere_embedding_model,
        input_type=input_type,
        embedding_types=["float"]
    )
    return response.embeddings.float


# HuggingFace
def _huggingface_embed_texts(texts: list[str]) -> list[list[float]]:
    return _get_huggingface_client().encode(texts).tolist()


_PROVIDERS = {
    "openai": _openai_embed_texts,
    "cohere": _cohere_embed_texts,
    "hf": _huggingface_embed_texts,
}
_BATCH_LIMITS = {
    "openai": 2048,
    "cohere": 96,
    "hf": settings.huggingface_batch_size,
}


def embed_chunks(chunks: list[Chunk]) -> list[Chunk]:
    if not chunks:
        raise ValueError("chunks must not be empty")

    embed_fn = _PROVIDERS.get(settings.embedding_provider)
    if embed_fn is None:
        raise ValueError(
            f"Unknown embedding provider: '{settings.embedding_provider}'. "
            f"Must be one of: {list(_PROVIDERS.keys())}"
        )

    texts = [chunk.text for chunk in chunks]
    batch_size = _BATCH_LIMITS[settings.embedding_provider]
    embeddings = []
    for i in range(0, len(texts), batch_size):
        embeddings.extend(embed_fn(texts[i:i + batch_size]))

    embedded_chunks = []
    for i, vector in enumerate(embeddings):
        chunk = chunks[i].model_copy(update={"vector": vector})
        embedded_chunks.append(chunk)

    return embedded_chunks


def embed_query(text: str) -> list[float]:
    if settings.embedding_provider == "cohere":
        return _cohere_embed_texts([text], input_type="search_query")[0]

    embed_fn = _PROVIDERS.get(settings.embedding_provider)
    if embed_fn is None:
        raise ValueError(
            f"Unknown embedding provider: '{settings.embedding_provider}'. "
            f"Must be one of: {list(_PROVIDERS.keys())}"
        )

    return embed_fn([text])[0]
