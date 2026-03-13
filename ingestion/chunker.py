from typing import Optional
from uuid import UUID
import uuid

import tiktoken
from unstructured.documents.elements import Element

from config import settings
from schemas import Chunk, ChunkMetadata

ENCODER = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def _build_chunk(
        text: str,
        chunk_metadata: ChunkMetadata,
        parent_id: Optional[UUID] = None,
        is_parent: bool = True
) -> Chunk:
    return Chunk(
        text=text,
        chunk_id=uuid.uuid4(),
        parent_id=parent_id,
        is_parent=is_parent,
        metadata=chunk_metadata
    )


def _build_metadata(element: Element) -> ChunkMetadata:
    return ChunkMetadata(
        filename=element.metadata.filename,
        filetype=element.metadata.filetype,
        page_number=element.metadata.page_number,
    )


def _exceeds_token_limit(text: str) -> bool:
    return _count_tokens(text) > settings.parent_chunk_size

def _split_text(text: str, size: int) -> list[str]:
    segments = []
    token_ids = ENCODER.encode(text)

    for i in range(0, len(token_ids), size):
        segment_token_ids = token_ids[i:i + size]
        segment_text = ENCODER.decode(segment_token_ids)
        segments.append(segment_text)

    return segments

def _flush_buffer(buffer_text: str, anchor_element: Element) -> tuple[str, Optional[Chunk]]:
    if _count_tokens(buffer_text) >= settings.parent_chunk_size:
        metadata = _build_metadata(anchor_element)
        return "", _build_chunk(buffer_text, metadata)
    return buffer_text, None


def chunk_parent(elements: list[Element]) -> list[Chunk]:
    if not elements:
        raise ValueError("elements must not be empty.")

    chunks = []
    buffer_text = ""
    anchor_element = None

    for element in elements:
        if buffer_text == "":
            anchor_element = element

        # split if text oversized
        text_segments = _split_text(element.text, settings.parent_chunk_size) if _exceeds_token_limit(element.text) else [element.text]
        for segment in text_segments:
            buffer_text += ("\n\n" if buffer_text else "") + segment
            buffer_text, chunk = _flush_buffer(buffer_text, anchor_element)
            if chunk:
                chunks.append(chunk)

    if buffer_text:
        metadata = _build_metadata(anchor_element)
        chunks.append(_build_chunk(buffer_text, metadata))

    return chunks


def chunk_children(parent_chunk: Chunk) -> list[Chunk]:
    chunks = []
    text_segments = _split_text(parent_chunk.text, size=settings.child_chunk_size)

    for segment in text_segments:
        chunks.append(_build_chunk(
            text=segment,
            chunk_metadata=parent_chunk.metadata,
            parent_id=parent_chunk.chunk_id,
            is_parent=False
        ))

    return chunks
