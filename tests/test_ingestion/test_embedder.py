import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4
from ingestion.embedder import (
    _openai_embed_texts,
    _cohere_embed_texts,
    _huggingface_embed_texts,
    _PROVIDERS,
    _BATCH_LIMITS,
    embed_chunks,
    embed_query,
)
from schemas import Chunk, ChunkMetadata


# ── helpers ──────────────────────────────────────────────────────

def _make_chunk(text: str = "some text", vector=None):
    return Chunk(
        text=text,
        chunk_id=uuid4(),
        is_parent=True,
        metadata=ChunkMetadata(
            filename="test.pdf",
            filetype="application/pdf",
            page_number=1,
        ),
        vector=vector,
    )


def _make_chunks(n: int):
    return [_make_chunk(text=f"text {i}") for i in range(n)]


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Returns a unique fake vector per text."""
    return [[float(i)] * 3 for i in range(len(texts))]


# ── _PROVIDERS / _BATCH_LIMITS registries ────────────────────────

class TestProviders:

    def test_openai_registered(self):
        assert "openai" in _PROVIDERS

    def test_cohere_registered(self):
        assert "cohere" in _PROVIDERS

    def test_hf_registered(self):
        assert "hf" in _PROVIDERS

    def test_all_providers_are_callable(self):
        for name, fn in _PROVIDERS.items():
            assert callable(fn), f"Provider '{name}' is not callable"


class TestBatchLimits:

    def test_openai_limit(self):
        assert _BATCH_LIMITS["openai"] == 2048

    def test_cohere_limit(self):
        assert _BATCH_LIMITS["cohere"] == 96

    def test_hf_limit_exists(self):
        assert "hf" in _BATCH_LIMITS
        assert isinstance(_BATCH_LIMITS["hf"], int)

    def test_providers_and_batch_limits_have_same_keys(self):
        assert set(_PROVIDERS.keys()) == set(_BATCH_LIMITS.keys())


# ── _openai_embed_texts ─────────────────────────────────────────

class TestOpenAIEmbedTexts:

    @patch("ingestion.embedder._get_openai_client")
    def test_returns_embeddings(self, mock_get_client):
        mock_item_1 = MagicMock()
        mock_item_1.embedding = [0.1, 0.2, 0.3]
        mock_item_2 = MagicMock()
        mock_item_2.embedding = [0.4, 0.5, 0.6]

        mock_response = MagicMock()
        mock_response.data = [mock_item_1, mock_item_2]

        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = _openai_embed_texts(["hello", "world"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @patch("ingestion.embedder._get_openai_client")
    def test_passes_correct_model(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.data = []
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        _openai_embed_texts(["test"])
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert "model" in call_kwargs
        assert call_kwargs["input"] == ["test"]


# ── _cohere_embed_texts ─────────────────────────────────────────

class TestCohereEmbedTexts:

    @patch("ingestion.embedder._get_cohere_client")
    def test_returns_embeddings(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.embeddings.float = [[0.1, 0.2], [0.3, 0.4]]

        mock_client = MagicMock()
        mock_client.embed.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = _cohere_embed_texts(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @patch("ingestion.embedder._get_cohere_client")
    def test_default_input_type_is_search_document(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.embeddings.float = [[0.1]]
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_response
        mock_get_client.return_value = mock_client

        _cohere_embed_texts(["test"])
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["input_type"] == "search_document"

    @patch("ingestion.embedder._get_cohere_client")
    def test_accepts_search_query_input_type(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.embeddings.float = [[0.1]]
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_response
        mock_get_client.return_value = mock_client

        _cohere_embed_texts(["test"], input_type="search_query")
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["input_type"] == "search_query"

    @patch("ingestion.embedder._get_cohere_client")
    def test_requests_float_embedding_type(self, mock_get_client):
        mock_response = MagicMock()
        mock_response.embeddings.float = [[0.1]]
        mock_client = MagicMock()
        mock_client.embed.return_value = mock_response
        mock_get_client.return_value = mock_client

        _cohere_embed_texts(["test"])
        call_kwargs = mock_client.embed.call_args[1]
        assert call_kwargs["embedding_types"] == ["float"]


# ── _huggingface_embed_texts ────────────────────────────────────

class TestHuggingFaceEmbedTexts:

    @patch("ingestion.embedder._get_huggingface_client")
    def test_returns_embeddings_as_list(self, mock_get_client):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
        mock_get_client.return_value = mock_model

        result = _huggingface_embed_texts(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    @patch("ingestion.embedder._get_huggingface_client")
    def test_passes_texts_to_encode(self, mock_get_client):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1]])
        mock_get_client.return_value = mock_model

        _huggingface_embed_texts(["hello"])
        mock_model.encode.assert_called_once_with(["hello"])


# ── embed_document ───────────────────────────────────────────────

class TestEmbedDocumentValidInput:

    def test_returns_chunks_with_vectors(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": _fake_embed_fn})
        monkeypatch.setattr("ingestion.embedder._BATCH_LIMITS", {"openai": 2048})
        chunks = _make_chunks(3)
        result = embed_chunks(chunks)
        assert len(result) == 3
        for chunk in result:
            assert chunk.vector is not None

    def test_does_not_mutate_original_chunks(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": _fake_embed_fn})
        monkeypatch.setattr("ingestion.embedder._BATCH_LIMITS", {"openai": 2048})
        chunks = _make_chunks(2)
        result = embed_chunks(chunks)
        for original in chunks:
            assert original.vector is None
        for embedded in result:
            assert embedded.vector is not None

    def test_preserves_chunk_text(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": _fake_embed_fn})
        monkeypatch.setattr("ingestion.embedder._BATCH_LIMITS", {"openai": 2048})
        chunks = _make_chunks(2)
        result = embed_chunks(chunks)
        for i, chunk in enumerate(result):
            assert chunk.text == chunks[i].text

    def test_preserves_chunk_metadata(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": _fake_embed_fn})
        monkeypatch.setattr("ingestion.embedder._BATCH_LIMITS", {"openai": 2048})
        chunks = _make_chunks(1)
        result = embed_chunks(chunks)
        assert result[0].metadata == chunks[0].metadata
        assert result[0].chunk_id == chunks[0].chunk_id


class TestEmbedDocumentBatching:

    def test_batches_calls_correctly(self, monkeypatch):
        call_log = []

        def tracking_embed_fn(texts):
            call_log.append(len(texts))
            return [[0.1]] * len(texts)

        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": tracking_embed_fn})
        monkeypatch.setattr("ingestion.embedder._BATCH_LIMITS", {"openai": 3})
        chunks = _make_chunks(7)
        result = embed_chunks(chunks)

        assert len(result) == 7
        assert call_log == [3, 3, 1]

    def test_single_batch_when_under_limit(self, monkeypatch):
        call_log = []

        def tracking_embed_fn(texts):
            call_log.append(len(texts))
            return [[0.1]] * len(texts)

        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": tracking_embed_fn})
        monkeypatch.setattr("ingestion.embedder._BATCH_LIMITS", {"openai": 100})
        chunks = _make_chunks(5)
        embed_chunks(chunks)

        assert call_log == [5]


class TestEmbedDocumentInvalidInput:

    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="chunks must not be empty"):
            embed_chunks([])

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "nonexistent")
        chunks = _make_chunks(1)
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            embed_chunks(chunks)

    def test_error_message_includes_provider_name(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "bad_provider")
        chunks = _make_chunks(1)
        with pytest.raises(ValueError, match="bad_provider"):
            embed_chunks(chunks)


# ── embed_query ──────────────────────────────────────────────────

class TestEmbedQueryValidInput:

    def test_returns_single_vector(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": _fake_embed_fn})
        result = embed_query("what is AI?")
        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    def test_cohere_uses_search_query_input_type(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "cohere")
        mock_cohere = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        monkeypatch.setattr("ingestion.embedder._cohere_embed_texts", mock_cohere)

        embed_query("test query")
        mock_cohere.assert_called_once_with(["test query"], input_type="search_query")

    def test_non_cohere_uses_providers_dict(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "openai")
        monkeypatch.setattr("ingestion.embedder._PROVIDERS", {"openai": _fake_embed_fn})
        result = embed_query("hello")
        assert isinstance(result, list)


class TestEmbedQueryInvalidInput:

    def test_unknown_provider_raises_value_error(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "nonexistent")
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            embed_query("test")

    def test_error_message_lists_valid_providers(self, monkeypatch):
        monkeypatch.setattr("ingestion.embedder.settings.embedding_provider", "bad")
        with pytest.raises(ValueError, match="Must be one of"):
            embed_query("test")
    