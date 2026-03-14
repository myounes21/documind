from schemas import RetrievedChunk

def format_context(chunks: list[RetrievedChunk]) -> str:
    formatted_contexts = []
    for i, chunk in enumerate(chunks):
        text = chunk.text
        file_name = chunk.metadata.filename
        page_number = chunk.metadata.page_number

        data = f"[S{i+1}] {text} (source: {file_name}, page {page_number})"
        formatted_contexts.append(data)

    return "\n\n".join(formatted_contexts)
