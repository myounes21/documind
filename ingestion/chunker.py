from .schemas import Chunk, ChunkMetadata
from unstructured.documents.elements import Element
import tiktoken
from config import settings
import uuid
from typing import Optional

ENCODER = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

def _build_chunk(
        buffer_text: str,
        chunk_metadata: ChunkMetadata,
        parent_id: Optional[str] = None,
        is_parent: bool = True
) -> Chunk:
    return Chunk(
        text=buffer_text,
        chunk_id=str(uuid.uuid4()),
        parent_id=parent_id,
        is_parent=is_parent,
        metadata=chunk_metadata
    )

def _build_metadata(element: Element) -> ChunkMetadata:
    return  ChunkMetadata(
        filename=element.metadata.filename,
        filetype=element.metadata.filetype,
        page_number=element.metadata.page_number,
    )

def chunk_parent(elements: list[Element]) -> list[Chunk]:
    if not elements:
        raise ValueError("elements must not be empty.")

    chunks = []
    buffer_text = ""
    first_element = None

    for element in elements:
        if buffer_text == "":
            first_element = element

        buffer_text += ("\n\n" if buffer_text else "") + element.text

        if settings.parent_chunk_size <= _count_tokens(buffer_text):
            metadata = _build_metadata(first_element)
            chunks.append(_build_chunk(buffer_text, metadata))
            buffer_text = ""

    if buffer_text:
        metadata = _build_metadata(first_element)
        chunks.append(_build_chunk(buffer_text, metadata))

    return chunks


def chunk_children(parent_chunk: Chunk) -> list[Chunk]:
    chunks = []
    token_ids = ENCODER.encode(parent_chunk.text)

    for i in range(0, len(token_ids), settings.child_chunk_size):
        child_token_ids = token_ids[i:i + settings.child_chunk_size]
        child_text = ENCODER.decode(child_token_ids)
        chunks.append(_build_chunk(
            buffer_text=child_text,
            chunk_metadata=parent_chunk.metadata,
            parent_id=parent_chunk.chunk_id,
            is_parent=False
        ))

    return chunks
