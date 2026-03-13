from .parser import parse_document
from .chunker import chunk_parent, chunk_children
from .embedder import embed_chunks
from .indexer import store_in_elasticsearch, store_in_qdrant


def ingest_document(file_path: str) -> None:
    # parse
    elements = parse_document(file_path)

    # chunk
    parents = chunk_parent(elements)
    children = [child for parent in parents for child in chunk_children(parent)]

    # embed
    embedded_children = embed_chunks(children)

    # index
    store_in_elasticsearch(parents + children)
    store_in_qdrant(embedded_children)
