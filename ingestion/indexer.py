from config import settings
from .schemas import Chunk
from qdrant_client.models import PointStruct
from .store_setup import qdrant_client


def store_in_qdrant(chunks: list[Chunk]) -> None:

    if any(chunk.vector is None for chunk in chunks):
        raise ValueError("all chunks must have a vector before indexing")

    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=chunk.vector,
            payload={
                "text": chunk.text,
                "filename": chunk.metadata.filename,
                "filetype": chunk.metadata.filetype,
                "page_number": chunk.metadata.page_number,
                "is_parent": chunk.is_parent,
                "parent_id": chunk.parent_id,
            }
        )
        for chunk in chunks
    ]

    qdrant_client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=points,
    )