from .schemas import Chunk
from config import settings


# lazy load all clients
_openai_client = None
_cohere_client = None
_hf_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_cohere_client():
    global _cohere_client
    if _cohere_client is None:
        import cohere
        _cohere_client = cohere.Client(api_key=settings.cohere_api_key)
    return _cohere_client


def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from sentence_transformers import SentenceTransformer
        _hf_client = SentenceTransformer(settings.huggingface_embedding_model)
    return _hf_client


# OpenAI
def _openai_embed_texts(texts: list[str]) -> list[list[float]]:
    response = _get_openai_client().embeddings.create(
        model=settings.openai_embedding_model,
        input=texts
    )
    return [item.embedding for item in response.data]


# Cohere
def _cohere_embed_texts(texts: list[str]) -> list[list[float]]:
    response = _get_cohere_client().embed(
        texts=texts,
        model=settings.cohere_embedding_model,
        input_type="search_document",
        embedding_types=["float"]
    )
    return response.embeddings.float


# HuggingFace
def _huggingface_embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_hf_client()
    return model.encode(texts).tolist()



