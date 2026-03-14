from config import settings
from .dense_retriever import dense_retrieve
from .sparse_retriever import sparse_retrieve
from .rrf import rrf
from .reranker import rerank_chunks
from .parent_fetcher import parent_fetch
from ingestion.embedder import embed_query
from schemas import RetrievedChunk

def retrieve(query_text: str) -> list[RetrievedChunk]:
    query_vector = embed_query(query_text)
    dense_results = dense_retrieve(query_vector, settings.retrieval_top_k)
    sparse_results = sparse_retrieve(query_text, settings.retrieval_top_k)
    rrf_results = rrf(dense_results, sparse_results, settings.rrf_top_k)
    rerank_results = rerank_chunks(query_text, rrf_results, settings.rerank_top_k)
    parent_chunks = parent_fetch(rerank_results)

    return parent_chunks

