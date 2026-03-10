from pydantic import BaseModel
from typing import Optional


class ChunkMetadata(BaseModel):
    page_number: int
    filename: str
    filetype: str

class Chunk(BaseModel):
    text: str
    chunk_id: str
    is_parent: bool
    metadata: ChunkMetadata
    parent_id: Optional[str] = None
