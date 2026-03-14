from langchain_core.messages import HumanMessage, AIMessage

memory_store: dict[str, list[HumanMessage | AIMessage]] = {}

def get_history(session_id: str) -> list[HumanMessage | AIMessage]:
    if session_id not in memory_store:
        memory_store[session_id] = []
    return memory_store[session_id]


def save_turn(session_id: str, question: str, answer: str) -> None:
    memory_store[session_id].append(HumanMessage(content=question))
    memory_store[session_id].append(AIMessage(content=answer))


def clear_history(session_id: str) -> None:
    if session_id in memory_store:
        memory_store[session_id] = []
