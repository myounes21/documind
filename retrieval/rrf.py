from schemas import RetrievedChunk


def rrf(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    k: int = 60,
    top_k: int = 5
) -> list[RetrievedChunk]:
    scores = {}  # chunk_id → rrf score
    chunks = {}  # chunk_id → RetrievedChunk object

    # score dense results
    for rank, chunk in enumerate(dense_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (rank + k)
        chunks[chunk.chunk_id] = chunk

    # score sparse results
    for rank, chunk in enumerate(sparse_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (rank + k)
        chunks[chunk.chunk_id] = chunk

    # sort by score
    sorted_ids = sorted(scores, key=lambda id: scores[id], reverse=True)

    return [
        chunks[id].model_copy(update={"score": scores[id]})  # ← update score
        for id in sorted_ids[:top_k]
    ]