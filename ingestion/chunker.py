from .models import Chunk
from unstructured.documents.elements import Element
import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text:str) -> int:
    tokens = ENCODER.encode(text)
    return len(tokens)


def chunk_parent(elements:list[Element]) -> list[Chunk]:
    pass

def chunk_children(parent_chunk :Chunk) -> list[Chunk]:
    pass

