from typing import Generator
from llm_client import get_chain


def stream(formatted_context: str, query: str) -> Generator[str, None, None]:
    chain = get_chain()
    for chunk in chain.stream({
        "context": formatted_context,
        "question": query
        }):
        yield chunk
