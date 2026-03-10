from pydantic import BaseModel
from typing import Optional


class ChunkMetadata(BaseModel):
    page_number: int
    filename: str
    filetype: str

class Chunk(BaseModel):
    chunk_id: str
    parent_id: Optional[str] = None
    text: str
    metadata: ChunkMetadata
    is_parent: bool