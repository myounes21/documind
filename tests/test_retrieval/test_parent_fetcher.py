import pytest
from uuid import uuid4
from unittest.mock import MagicMock

from schemas import RetrievedChunk, ChunkMetadata
from retrieval.parent_fetcher import parent_fetch


def _make_retrieved_chunk(
    text: str = "child",
    score: float = 0.5,
    parent_id=None,
    filename: str = "child.pdf",
    filetype: str = "application/pdf",
    page_number: int = 1,
):
    return RetrievedChunk(
        text=text,
        chunk_id=uuid4(),
        metadata=ChunkMetadata(
            filename=filename,
            filetype=filetype,
            page_number=page_number,
        ),
        parent_id=parent_id,
        vector=None,
        score=score,
    )


class TestParentFetch:
    def test_no_parent_ids_returns_original_chunks_without_mget(self, monkeypatch):
        mock_client = MagicMock()
        monkeypatch.setattr("retrieval.parent_fetcher.client", mock_client)

        rerank_result = [
            _make_retrieved_chunk(parent_id=None),
            _make_retrieved_chunk(parent_id=None),
        ]

        result = parent_fetch(rerank_result)

        assert result == rerank_result
        mock_client.mget.assert_not_called()

    def test_calls_mget_with_ordered_deduplicated_parent_ids(self, monkeypatch):
        pid1 = uuid4()
        pid2 = uuid4()

        mock_client = MagicMock()
        mock_client.mget.return_value = {
            "docs": [
                {
                    "_id": str(pid1),
                    "found": True,
                    "_source": {
                        "text": "parent one",
                        "filename": "a.pdf",
                        "filetype": "application/pdf",
                        "page_number": 3,
                        "parent_id": None,
                    },
                },
                {
                    "_id": str(pid2),
                    "found": True,
                    "_source": {
                        "text": "parent two",
                        "filename": "b.pdf",
                        "filetype": "application/pdf",
                        "page_number": 9,
                        "parent_id": None,
                    },
                },
            ]
        }

        monkeypatch.setattr("retrieval.parent_fetcher.client", mock_client)
        monkeypatch.setattr("retrieval.parent_fetcher.settings.elasticsearch_index_name", "chunks_test")

        rerank_result = [
            _make_retrieved_chunk(text="c1", parent_id=pid1),
            _make_retrieved_chunk(text="c2", parent_id=pid2),
            _make_retrieved_chunk(text="c3", parent_id=pid1),  # duplicate
            _make_retrieved_chunk(text="c4", parent_id=None),
        ]

        result = parent_fetch(rerank_result)

        mock_client.mget.assert_called_once_with(
            index="chunks_test",
            body={"ids": [str(pid1), str(pid2)]},
        )
        assert len(result) == 2
        assert str(result[0].chunk_id) == str(pid1)
        assert str(result[1].chunk_id) == str(pid2)

    def test_maps_found_docs_to_parent_chunks(self, monkeypatch):
        pid = uuid4()
        mock_client = MagicMock()
        mock_client.mget.return_value = {
            "docs": [
                {
                    "_id": str(pid),
                    "found": True,
                    "_source": {
                        "text": "parent text",
                        "filename": "report.pdf",
                        "filetype": "application/pdf",
                        "page_number": 7,
                        "parent_id": None,
                    },
                }
            ]
        }

        monkeypatch.setattr("retrieval.parent_fetcher.client", mock_client)

        rerank_result = [_make_retrieved_chunk(parent_id=pid)]
        result = parent_fetch(rerank_result)

        assert len(result) == 1
        assert str(result[0].chunk_id) == str(pid)
        assert result[0].text == "parent text"
        assert result[0].metadata.filename == "report.pdf"
        assert result[0].metadata.filetype == "application/pdf"
        assert result[0].metadata.page_number == 7
        assert result[0].parent_id is None

    def test_non_int_page_number_defaults_to_zero(self, monkeypatch):
        pid = uuid4()
        mock_client = MagicMock()
        mock_client.mget.return_value = {
            "docs": [
                {
                    "_id": str(pid),
                    "found": True,
                    "_source": {
                        "text": "parent text",
                        "filename": "report.pdf",
                        "filetype": "application/pdf",
                        "page_number": "unknown",
                        "parent_id": None,
                    },
                }
            ]
        }

        monkeypatch.setattr("retrieval.parent_fetcher.client", mock_client)

        rerank_result = [_make_retrieved_chunk(parent_id=pid)]
        result = parent_fetch(rerank_result)

        assert len(result) == 1
        assert result[0].metadata.page_number == 0

    def test_mget_exception_returns_original_chunks(self, monkeypatch):
        pid = uuid4()
        mock_client = MagicMock()
        mock_client.mget.side_effect = RuntimeError("es down")
        monkeypatch.setattr("retrieval.parent_fetcher.client", mock_client)

        rerank_result = [_make_retrieved_chunk(parent_id=pid)]
        result = parent_fetch(rerank_result)

        assert result == rerank_result

    def test_no_found_parents_returns_original_chunks(self, monkeypatch):
        pid = uuid4()
        mock_client = MagicMock()
        mock_client.mget.return_value = {
            "docs": [
                {"_id": str(pid), "found": False}
            ]
        }
        monkeypatch.setattr("retrieval.parent_fetcher.client", mock_client)

        rerank_result = [_make_retrieved_chunk(parent_id=pid)]
        result = parent_fetch(rerank_result)

        assert result == rerank_result

