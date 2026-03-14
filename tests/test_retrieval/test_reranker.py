import pytest
from uuid import uuid4
from unittest.mock import MagicMock, patch

from schemas import RetrievedChunk, ChunkMetadata
from retrieval.reranker import (
    _cohere_rerank,
    _huggingface_rerank,
    _RERANK_PROVIDERS,
    rerank_chunks,
)


def _make_retrieved_chunk(
    text: str = "hello",
    score: float = 0.1,
    is_parent: bool = False,
    parent_id=None,
    filename: str = "test.pdf",
    filetype: str = "application/pdf",
    page_number: int = 1,
):
    return RetrievedChunk(
        text=text,
        chunk_id=uuid4(),
        is_parent=is_parent,
        metadata=ChunkMetadata(
            filename=filename,
            filetype=filetype,
            page_number=page_number,
        ),
        parent_id=parent_id,
        vector=None,
        score=score,
    )


class TestCohereRerank:
    @patch("retrieval.reranker._get_cohere_client")
    def test_uses_all_chunk_texts_and_returns_pairs(self, mock_get_client):
        mock_result1 = MagicMock()
        mock_result1.index = 1
        mock_result1.relevance_score = 0.9
        mock_result2 = MagicMock()
        mock_result2.index = 0
        mock_result2.relevance_score = 0.5

        mock_response = MagicMock()
        mock_response.results = [mock_result1, mock_result2]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        mock_get_client.return_value = mock_client

        chunks = [_make_retrieved_chunk(text="A"), _make_retrieved_chunk(text="B")]

        with patch("retrieval.reranker.settings") as mock_settings:
            mock_settings.cohere_reranker_model = "cohere-model"
            scores = _cohere_rerank("query", chunks)

        # client is called with expected parameters
        mock_client.rerank.assert_called_once()
        call_kwargs = mock_client.rerank.call_args.kwargs
        assert call_kwargs["model"] == "cohere-model"
        assert call_kwargs["query"] == "query"
        assert call_kwargs["documents"] == ["A", "B"]
        assert call_kwargs["top_n"] == len(chunks)

        # results are mapped back using indices
        assert scores == [
            (chunks[1], 0.9),
            (chunks[0], 0.5),
        ]


class TestHuggingFaceRerank:
    @patch("retrieval.reranker._get_huggingface_reranker")
    def test_builds_query_pairs_and_returns_pairs(self, mock_get_model):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.8]
        mock_get_model.return_value = mock_model

        chunks = [_make_retrieved_chunk(text="A"), _make_retrieved_chunk(text="B")]
        scores = _huggingface_rerank("query", chunks)

        mock_model.predict.assert_called_once_with([("query", "A"), ("query", "B")])
        assert scores == list(zip(chunks, [0.2, 0.8]))


class TestRerankChunks:
    def test_empty_chunks_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr("retrieval.reranker.settings.rerank_provider", "cohere")
        assert rerank_chunks("q", [], top_k=5) == []

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("retrieval.reranker.settings.rerank_provider", "unknown")
        with pytest.raises(ValueError, match="Unknown reranker provider"):
            rerank_chunks("q", [_make_retrieved_chunk()], top_k=1)

    def test_cohere_provider_is_used_and_results_are_sorted(self, monkeypatch):
        chunks = [
            _make_retrieved_chunk(text="A", score=0.0),
            _make_retrieved_chunk(text="B", score=0.0),
            _make_retrieved_chunk(text="C", score=0.0),
        ]

        def fake_rerank(query, chunks_arg):
            # Return out of order scores to ensure sorting happens
            return [
                (chunks_arg[1], 0.7),
                (chunks_arg[2], 0.9),
                (chunks_arg[0], 0.3),
            ]

        monkeypatch.setattr("retrieval.reranker.settings.rerank_provider", "cohere")
        monkeypatch.setattr("retrieval.reranker._RERANK_PROVIDERS", {"cohere": fake_rerank})

        result = rerank_chunks("q", chunks, top_k=2)

        assert len(result) == 2
        # Highest score first, then next
        assert result[0].text == "C"
        assert result[0].score == 0.9
        assert result[1].text == "B"
        assert result[1].score == 0.7

    def test_top_k_limits_results(self, monkeypatch):
        chunks = [
            _make_retrieved_chunk(text="A", score=0.0),
            _make_retrieved_chunk(text="B", score=0.0),
            _make_retrieved_chunk(text="C", score=0.0),
        ]

        def fake_rerank(query, chunks_arg):
            return [(c, float(i)) for i, c in enumerate(chunks_arg)]

        monkeypatch.setattr("retrieval.reranker.settings.rerank_provider", "hf")
        monkeypatch.setattr("retrieval.reranker._RERANK_PROVIDERS", {"hf": fake_rerank})

        result = rerank_chunks("q", chunks, top_k=1)

        assert len(result) == 1
        # After sorting desc by score, last element should be returned
        assert result[0].text == "C"

