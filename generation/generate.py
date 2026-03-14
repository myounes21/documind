from typing import Generator
from retrieval.retrieve import retrieve
from .context_formatter import format_context
from .streamer import stream


def generate(query_text: str) -> Generator[str, None, None]:
    retrieved_chunks = retrieve(query_text)
    formatted_context = format_context(retrieved_chunks)

    yield from stream(formatted_context, query_text)