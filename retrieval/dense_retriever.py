from config import settings
from schemas import RetrievedChunk, ChunkMetadata
from db_setup import qdrant_client as client

def retrieve_dense(query_vector: list[float], top_k: int) -> list[RetrievedChunk]:
    results = client.search(
        collection_name=settings.qdrant_collection_name,
        query_vector=query_vector,
        limit=top_k,
    )

    return [
        RetrievedChunk(
            chunk_id=point.id,
            text=point.payload["text"],
            score=point.score,
            metadata=ChunkMetadata(
                filename=point.payload["filename"],
                filetype=point.payload["filetype"],
                page_number=point.payload["page_number"],
            ),
            parent_id=point.payload["parent_id"],
        )
        for point in results
    ]

