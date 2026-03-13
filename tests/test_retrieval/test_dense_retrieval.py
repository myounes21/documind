import pytest
from uuid import uuid4
from unittest.mock import MagicMock

from retrieval.dense_retriever import dense_retrieve


def _make_point(
    *,
    text: str = "hello",
    score: float = 0.99,
    is_parent: bool = False,
    parent_id=None,
    filename: str = "x.pdf",
    filetype: str = "application/pdf",
    page_number: int = 1,
):
    point = MagicMock()
    point.id = uuid4()
    point.score = score
    point.payload = {
        "text": text,
        "filename": filename,
        "filetype": filetype,
        "page_number": page_number,
        "is_parent": is_parent,
        "parent_id": parent_id,
    }
    return point


class TestRetrieveDense:
    def test_calls_qdrant_search_with_expected_args(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.search.return_value = []
        monkeypatch.setattr("retrieval.dense_retriever.client", mock_client)
        monkeypatch.setattr("retrieval.dense_retriever.settings.qdrant_collection_name", "chunks_test")

        query_vector = [0.1, 0.2, 0.3]
        dense_retrieve(query_vector=query_vector, top_k=5)

        mock_client.search.assert_called_once_with(
            collection_name="chunks_test",
            query_vector=query_vector,
            limit=5,
        )

    def test_empty_results_returns_empty_list(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.search.return_value = []
        monkeypatch.setattr("retrieval.dense_retriever.client", mock_client)

        result = dense_retrieve(query_vector=[0.1], top_k=3)

        assert result == []

    def test_maps_qdrant_points_to_retrieved_chunks(self, monkeypatch):
        parent_id = uuid4()
        p1 = _make_point(text="A", score=0.5, is_parent=False, parent_id=parent_id, page_number=2)
        p2 = _make_point(text="B", score=0.4, is_parent=True, parent_id=None, filename="y.pdf")

        mock_client = MagicMock()
        mock_client.search.return_value = [p1, p2]
        monkeypatch.setattr("retrieval.dense_retriever.client", mock_client)

        result = dense_retrieve(query_vector=[0.0], top_k=2)

        assert len(result) == 2
        assert result[0].chunk_id == p1.id
        assert result[0].text == "A"
        assert result[0].score == 0.5
        assert result[0].metadata.filename == p1.payload["filename"]
        assert result[0].metadata.page_number == 2
        assert result[0].parent_id == parent_id

        assert result[1].chunk_id == p2.id
        assert result[1].text == "B"
        assert result[1].score == 0.4
        assert result[1].metadata.filename == "y.pdf"
        assert result[1].parent_id is None

    def test_bubbles_up_client_exceptions(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.search.side_effect = RuntimeError("qdrant down")
        monkeypatch.setattr("retrieval.dense_retriever.client", mock_client)

        with pytest.raises(RuntimeError, match="qdrant down"):
            dense_retrieve(query_vector=[0.0], top_k=1)

