from .models import Chunk, ChunkMetadata
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
        first_element: Element,
        parent_id: Optional[str] = None,
        is_parent: bool = True
) -> Chunk:
    return Chunk(
        text=buffer_text,
        chunk_id=str(uuid.uuid4()),
        parent_id=parent_id,
        is_parent=is_parent,
        metadata=ChunkMetadata(
            filename=first_element.metadata.filename,
            filetype=first_element.metadata.filetype,
            page_number=first_element.metadata.page_number,
        )
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

        if settings.parent_tokens_min <= _count_tokens(buffer_text):
            chunks.append(_build_chunk(buffer_text, first_element))
            buffer_text = ""

    if buffer_text:
        chunks.append(_build_chunk(buffer_text, first_element))

    return chunks


# def chunk_children(parent_chunk: Chunk) -> list[Chunk]:
#     chunks = []
#     token_ids = ENCODER.encode(parent_chunk.text)
#
#     for i in range(0, len(token_ids), settings.child_tokens_min):
#         child_token_ids = token_ids[i:i + settings.child_tokens_min]
#         child_text = ENCODER.decode(child_token_ids)
#         chunks.append(_build_chunk(
#             buffer_text=child_text,
#             first_element=parent_chunk.metadata,
#             parent_id=parent_chunk.chunk_id,
#             is_parent=False
#         ))
#
#     return chunks
