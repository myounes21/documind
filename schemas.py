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
    metadata: ChunkMetadata
    parent_id: Optional[UUID] = None
    vector: Optional[list[float]] = None

class RetrievedChunk(BaseModel):
    text: str
    chunk_id: UUID
    metadata: ChunkMetadata
    parent_id: Optional[UUID] = None
    vector: Optional[list[float]] = None
    score: float

