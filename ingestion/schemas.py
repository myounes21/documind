from pydantic import BaseModel
from typing import Optional
from uuid import UUID


class ChunkMetadata(BaseModel):
    page_number: int
    filename: str
    filetype: str

class Chunk(BaseModel):
    text: str
    chunk_id: UUID
    is_parent: bool
    parent_id: Optional[UUID] = None
    metadata: ChunkMetadata

    vector: Optional[list[float]] = None
