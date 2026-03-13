from config import settings
from schemas import RetrievedChunk, ChunkMetadata
from db_setup import elastic_client as client


def sparse_retrieve(query_text: str, top_k: int) -> list[RetrievedChunk]:
    results = client.search(
        index=settings.elasticsearch_index_name,
        body={
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {"match": {"text": query_text}}
                    ],
                    "filter": [
                        {"exists": {"field": "parent_id"}}  # only children
                    ]
                }
            }
        }
    )

    hits = results["hits"]["hits"]

    return [
        RetrievedChunk(
            chunk_id=hit["_id"],
            text=hit["_source"]["text"],
            score=hit["_score"],
            metadata=ChunkMetadata(
                filename=hit["_source"]["filename"],
                filetype=hit["_source"]["filetype"],
                page_number=hit["_source"]["page_number"],
            ),
            parent_id=hit["_source"]["parent_id"],
        )
        for hit in hits
    ]
