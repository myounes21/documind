from typing import Generator
from retrieval.retrieve import retrieve
from .context_formatter import format_context
from .streamer import stream
from .memory import get_history, save_turn


def generate(query_text: str, session_id: str) -> Generator[str, None, None]:

    retrieved_chunks = retrieve(query_text)
    formatted_context = format_context(retrieved_chunks)

    chat_history = get_history(session_id)

    response_tokens: list[str] = []

    for token in stream(formatted_context, query_text, chat_history):
        response_tokens.append(token)
        yield token

    answer = "".join(response_tokens)

    save_turn(session_id, query_text, answer)