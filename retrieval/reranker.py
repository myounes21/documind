from functools import cache
from config import settings
from schemas import RetrievedChunk


@cache
def _get_cohere_client():
    import cohere
    return cohere.Client(api_key=settings.cohere_api_key)

@cache
def _get_huggingface_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(settings.huggingface_reranker_model)


def _cohere_rerank(query: str, chunks: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, float]]:
    client = _get_cohere_client()
    response = client.rerank(
        model=settings.cohere_reranker_model,
        query=query,
        documents=[c.text for c in chunks],
        top_n=len(chunks)
    )
    return [(chunks[r.index], r.relevance_score) for r in response.results]

def _huggingface_rerank(query: str, chunks: list[RetrievedChunk]) -> list[tuple[RetrievedChunk, float]]:
    model = _get_huggingface_reranker()
    pairs = [(query, chunk.text) for chunk in chunks]
    scores = model.predict(pairs)
    return list(zip(chunks, scores))


_RERANK_PROVIDERS = {
    "cohere": _cohere_rerank,
    "hf": _huggingface_rerank,
}


def rerank_chunks(query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
    if not chunks:
        return []

    rerank_fn = _RERANK_PROVIDERS.get(settings.rerank_provider)
    if rerank_fn is None:
        raise ValueError(
            f"Unknown reranker provider: '{settings.rerank_provider}'. "
            f"Must be one of: {list(_RERANK_PROVIDERS.keys())}"
        )

    ranked = rerank_fn(query, chunks)
    ranked.sort(key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]

    return [chunk.model_copy(update={"score": score}) for chunk, score in top]