import pytest
from unittest.mock import MagicMock
from uuid import uuid4

from ingestion.indexer import store_in_elasticsearch, store_in_qdrant
from ingestion.schemas import Chunk, ChunkMetadata


def _make_metadata(filename: str = "test.pdf", filetype: str = "application/pdf", page_number: int = 1):
    return ChunkMetadata(filename=filename, filetype=filetype, page_number=page_number)


def _make_chunk(
    *,
    text: str = "hello",
    vector: list[float] | None = None,
    is_parent: bool = True,
    parent_id=None,
    metadata: ChunkMetadata | None = None,
):
    return Chunk(
        text=text,
        chunk_id=uuid4(),
        is_parent=is_parent,
        parent_id=parent_id,
        metadata=metadata or _make_metadata(),
        vector=vector,
    )


class TestStoreInQdrant:

    def test_missing_vector_raises_value_error(self):
        chunks = [_make_chunk(vector=None)]
        with pytest.raises(ValueError, match="must have a vector"):
            store_in_qdrant(chunks)

    def test_empty_list_is_noop(self, monkeypatch):
        mock_qdrant = MagicMock()
        monkeypatch.setattr("ingestion.indexer.qdrant_client", mock_qdrant)

        store_in_qdrant([])

        mock_qdrant.upsert.assert_not_called()

    def test_upserts_points_with_expected_payload(self, monkeypatch):
        mock_qdrant = MagicMock()
        monkeypatch.setattr("ingestion.indexer.qdrant_client", mock_qdrant)
        monkeypatch.setattr("ingestion.indexer.settings.qdrant_collection_name", "test_chunks")

        chunk = _make_chunk(
            text="abc",
            vector=[0.1, 0.2, 0.3],
            is_parent=False,
            parent_id=uuid4(),
            metadata=_make_metadata(filename="x.pdf", filetype="application/pdf", page_number=7),
        )
        store_in_qdrant([chunk])

        mock_qdrant.upsert.assert_called_once()
        call_kwargs = mock_qdrant.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_chunks"
        assert len(call_kwargs["points"]) == 1

        point = call_kwargs["points"][0]
        assert point.id == chunk.chunk_id
        assert point.vector == chunk.vector
        assert point.payload == {
            "text": chunk.text,
            "filename": chunk.metadata.filename,
            "filetype": chunk.metadata.filetype,
            "page_number": chunk.metadata.page_number,
            "is_parent": chunk.is_parent,
            "parent_id": chunk.parent_id,
        }

    def test_client_exception_bubbles_up(self, monkeypatch):
        mock_qdrant = MagicMock()
        mock_qdrant.upsert.side_effect = RuntimeError("qdrant down")
        monkeypatch.setattr("ingestion.indexer.qdrant_client", mock_qdrant)

        chunk = _make_chunk(text="abc", vector=[0.1])

        with pytest.raises(RuntimeError, match="qdrant down"):
            store_in_qdrant([chunk])


class TestStoreInElasticsearch:

    def test_empty_list_is_noop(self, monkeypatch):
        mock_bulk = MagicMock()
        mock_es = MagicMock()
        monkeypatch.setattr("ingestion.indexer.bulk", mock_bulk)
        monkeypatch.setattr("ingestion.indexer.elastic_client", mock_es)

        store_in_elasticsearch([])

        mock_bulk.assert_not_called()

    def test_bulk_indexes_actions(self, monkeypatch):
        mock_bulk = MagicMock()
        mock_es = MagicMock()
        monkeypatch.setattr("ingestion.indexer.bulk", mock_bulk)
        monkeypatch.setattr("ingestion.indexer.elastic_client", mock_es)
        monkeypatch.setattr("ingestion.indexer.settings.elasticsearch_index_name", "test_chunks")

        chunk = _make_chunk(
            text="hello world",
            vector=[0.5, 0.6],
            is_parent=True,
            parent_id=None,
            metadata=_make_metadata(filename="z.pdf", filetype="application/pdf", page_number=2),
        )

        store_in_elasticsearch([chunk])

        mock_bulk.assert_called_once()
        args, _kwargs = mock_bulk.call_args
        assert args[0] is mock_es

        actions = args[1]
        assert isinstance(actions, list)
        assert len(actions) == 1
        assert actions[0] == {
            "_index": "test_chunks",
            "_id": chunk.chunk_id,
            "_source": {
                "text": chunk.text,
                "filename": chunk.metadata.filename,
                "filetype": chunk.metadata.filetype,
                "page_number": chunk.metadata.page_number,
                "is_parent": chunk.is_parent,
                "parent_id": chunk.parent_id,
            },
        }

    def test_bulk_exception_bubbles_up(self, monkeypatch):
        mock_bulk = MagicMock()
        mock_bulk.side_effect = RuntimeError("es down")
        mock_es = MagicMock()
        monkeypatch.setattr("ingestion.indexer.bulk", mock_bulk)
        monkeypatch.setattr("ingestion.indexer.elastic_client", mock_es)
        monkeypatch.setattr("ingestion.indexer.settings.elasticsearch_index_name", "test_chunks")

        chunk = _make_chunk(text="hello", vector=[0.5])

        with pytest.raises(RuntimeError, match="es down"):
            store_in_elasticsearch([chunk])

