import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from retrieval.retrieve import retrieve


class TestRetrieve:
    def test_happy_path_calls_all_stages_with_expected_args(self, monkeypatch):
        fake_settings = SimpleNamespace(retrieval_top_k=10, rrf_top_k=6, rerank_top_k=4)
        monkeypatch.setattr("retrieval.retrieve.settings", fake_settings)

        embed_query_mock = MagicMock(return_value=[0.11, 0.22])
        dense_retrieve_mock = MagicMock(return_value=["dense_1", "dense_2"])
        sparse_retrieve_mock = MagicMock(return_value=["sparse_1"])
        rrf_mock = MagicMock(return_value=["rrf_1", "rrf_2"])
        rerank_chunks_mock = MagicMock(return_value=["rerank_1"])
        parent_fetch_mock = MagicMock(return_value=["parent_1"])

        monkeypatch.setattr("retrieval.retrieve.embed_query", embed_query_mock)
        monkeypatch.setattr("retrieval.retrieve.dense_retrieve", dense_retrieve_mock)
        monkeypatch.setattr("retrieval.retrieve.sparse_retrieve", sparse_retrieve_mock)
        monkeypatch.setattr("retrieval.retrieve.rrf", rrf_mock)
        monkeypatch.setattr("retrieval.retrieve.rerank_chunks", rerank_chunks_mock)
        monkeypatch.setattr("retrieval.retrieve.parent_fetch", parent_fetch_mock)

        result = retrieve("what is the revenue?")

        assert result == ["parent_1"]
        embed_query_mock.assert_called_once_with("what is the revenue?")
        dense_retrieve_mock.assert_called_once_with([0.11, 0.22], 10)
        sparse_retrieve_mock.assert_called_once_with("what is the revenue?", 10)
        rrf_mock.assert_called_once_with(["dense_1", "dense_2"], ["sparse_1"], 6)
        rerank_chunks_mock.assert_called_once_with("what is the revenue?", ["rrf_1", "rrf_2"], 4)
        parent_fetch_mock.assert_called_once_with(["rerank_1"])

    def test_returns_exact_parent_fetch_output(self, monkeypatch):
        fake_settings = SimpleNamespace(retrieval_top_k=5, rrf_top_k=3, rerank_top_k=2)
        monkeypatch.setattr("retrieval.retrieve.settings", fake_settings)

        expected = ["final_parent_a", "final_parent_b"]

        monkeypatch.setattr("retrieval.retrieve.embed_query", MagicMock(return_value=[0.1]))
        monkeypatch.setattr("retrieval.retrieve.dense_retrieve", MagicMock(return_value=["d"]))
        monkeypatch.setattr("retrieval.retrieve.sparse_retrieve", MagicMock(return_value=["s"]))
        monkeypatch.setattr("retrieval.retrieve.rrf", MagicMock(return_value=["fused"]))
        monkeypatch.setattr("retrieval.retrieve.rerank_chunks", MagicMock(return_value=["reranked"]))
        monkeypatch.setattr("retrieval.retrieve.parent_fetch", MagicMock(return_value=expected))

        result = retrieve("hello")

        assert result is expected

    def test_embed_query_exception_bubbles_and_stops_pipeline(self, monkeypatch):
        fake_settings = SimpleNamespace(retrieval_top_k=5, rrf_top_k=3, rerank_top_k=2)
        monkeypatch.setattr("retrieval.retrieve.settings", fake_settings)

        embed_query_mock = MagicMock(side_effect=RuntimeError("embedding failed"))
        dense_retrieve_mock = MagicMock()
        sparse_retrieve_mock = MagicMock()

        monkeypatch.setattr("retrieval.retrieve.embed_query", embed_query_mock)
        monkeypatch.setattr("retrieval.retrieve.dense_retrieve", dense_retrieve_mock)
        monkeypatch.setattr("retrieval.retrieve.sparse_retrieve", sparse_retrieve_mock)

        with pytest.raises(RuntimeError, match="embedding failed"):
            retrieve("test")

        dense_retrieve_mock.assert_not_called()
        sparse_retrieve_mock.assert_not_called()

    def test_dense_retrieve_exception_bubbles_and_stops_pipeline(self, monkeypatch):
        fake_settings = SimpleNamespace(retrieval_top_k=5, rrf_top_k=3, rerank_top_k=2)
        monkeypatch.setattr("retrieval.retrieve.settings", fake_settings)

        embed_query_mock = MagicMock(return_value=[0.1])
        dense_retrieve_mock = MagicMock(side_effect=RuntimeError("qdrant down"))
        sparse_retrieve_mock = MagicMock()

        monkeypatch.setattr("retrieval.retrieve.embed_query", embed_query_mock)
        monkeypatch.setattr("retrieval.retrieve.dense_retrieve", dense_retrieve_mock)
        monkeypatch.setattr("retrieval.retrieve.sparse_retrieve", sparse_retrieve_mock)

        with pytest.raises(RuntimeError, match="qdrant down"):
            retrieve("test")

        sparse_retrieve_mock.assert_not_called()

    def test_rerank_exception_bubbles_and_parent_fetch_not_called(self, monkeypatch):
        fake_settings = SimpleNamespace(retrieval_top_k=8, rrf_top_k=4, rerank_top_k=2)
        monkeypatch.setattr("retrieval.retrieve.settings", fake_settings)

        monkeypatch.setattr("retrieval.retrieve.embed_query", MagicMock(return_value=[0.2]))
        monkeypatch.setattr("retrieval.retrieve.dense_retrieve", MagicMock(return_value=["d1"]))
        monkeypatch.setattr("retrieval.retrieve.sparse_retrieve", MagicMock(return_value=["s1"]))
        monkeypatch.setattr("retrieval.retrieve.rrf", MagicMock(return_value=["f1", "f2"]))

        rerank_chunks_mock = MagicMock(side_effect=RuntimeError("rerank failed"))
        parent_fetch_mock = MagicMock()
        monkeypatch.setattr("retrieval.retrieve.rerank_chunks", rerank_chunks_mock)
        monkeypatch.setattr("retrieval.retrieve.parent_fetch", parent_fetch_mock)

        with pytest.raises(RuntimeError, match="rerank failed"):
            retrieve("test")

        parent_fetch_mock.assert_not_called()

    def test_empty_fusion_results_still_flows_to_parent_fetch(self, monkeypatch):
        fake_settings = SimpleNamespace(retrieval_top_k=7, rrf_top_k=5, rerank_top_k=3)
        monkeypatch.setattr("retrieval.retrieve.settings", fake_settings)

        monkeypatch.setattr("retrieval.retrieve.embed_query", MagicMock(return_value=[0.3]))
        monkeypatch.setattr("retrieval.retrieve.dense_retrieve", MagicMock(return_value=[]))
        monkeypatch.setattr("retrieval.retrieve.sparse_retrieve", MagicMock(return_value=[]))
        monkeypatch.setattr("retrieval.retrieve.rrf", MagicMock(return_value=[]))

        rerank_chunks_mock = MagicMock(return_value=[])
        parent_fetch_mock = MagicMock(return_value=[])
        monkeypatch.setattr("retrieval.retrieve.rerank_chunks", rerank_chunks_mock)
        monkeypatch.setattr("retrieval.retrieve.parent_fetch", parent_fetch_mock)

        result = retrieve("nothing")

        assert result == []
        rerank_chunks_mock.assert_called_once_with("nothing", [], 3)
        parent_fetch_mock.assert_called_once_with([])

