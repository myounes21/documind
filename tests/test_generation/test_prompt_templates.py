from uuid import uuid4

from generation.prompt_templates import (
    SYSTEM_PROMPT,
    build_langchain_messages,
    build_messages,
    build_user_prompt,
    format_context,
    format_history,
)
from schemas import ChunkMetadata, RetrievedChunk


def _make_chunk(text: str, filename: str = "a.pdf", page_number: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        chunk_id=uuid4(),
        metadata=ChunkMetadata(
            filename=filename,
            filetype="application/pdf",
            page_number=page_number,
        ),
        parent_id=None,
        vector=None,
        score=0.9,
    )


class TestPromptTemplates:
    def test_format_context_renders_sources_with_indexes(self):
        chunks = [
            _make_chunk("Revenue increased in Q3.", filename="report.pdf", page_number=3),
            _make_chunk("Net income was stable.", filename="report.pdf", page_number=4),
        ]

        rendered = format_context(chunks)

        assert "[S1] report.pdf (page 3)" in rendered
        assert "[S2] report.pdf (page 4)" in rendered
        assert "Revenue increased in Q3." in rendered
        assert "Net income was stable." in rendered

    def test_format_context_empty_returns_no_context_marker(self):
        assert format_context([]) == "[NO_CONTEXT]"

    def test_format_history_empty_returns_no_history_marker(self):
        assert format_history(None) == "[NO_HISTORY]"

    def test_build_user_prompt_includes_sections(self):
        prompt = build_user_prompt(
            query_text="  What   happened in Q3?  ",
            chunks=[_make_chunk("Revenue increased.", page_number=7)],
            history=[("  hi  ", " hello there  ")],
        )

        assert "Conversation history:" in prompt
        assert "Retrieved context:" in prompt
        assert "Question:" in prompt
        assert "Answer requirements:" in prompt
        assert "What happened in Q3?" in prompt
        assert "[S1]" in prompt

    def test_build_messages_returns_chat_style_messages(self):
        messages = build_messages(
            query_text="What happened?",
            chunks=[_make_chunk("Revenue increased.")],
            history=None,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "Retrieved context:" in messages[1]["content"]

    def test_build_langchain_messages_returns_system_and_human(self):
        messages = build_langchain_messages(
            query_text="What happened?",
            chunks=[_make_chunk("Revenue increased.")],
            history=None,
        )

        assert len(messages) == 2
        assert messages[0].type == "system"
        assert messages[0].content == SYSTEM_PROMPT
        assert messages[1].type == "human"
        assert "Retrieved context:" in messages[1].content

