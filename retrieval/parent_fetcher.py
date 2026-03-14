import logging
from config import settings
from schemas import RetrievedChunk, ChunkMetadata
from db_setup import elastic_client as client

logger = logging.getLogger(__name__)


def parent_fetch(rerank_result: list[RetrievedChunk]) -> list[RetrievedChunk]:
    # Keep first-seen order from reranked children while deduplicating
    ordered_parent_ids: list[str] = []
    seen: set[str] = set()
    for chunk in rerank_result:
        if not chunk.parent_id:
            continue
        pid = str(chunk.parent_id)
        if pid not in seen:
            seen.add(pid)
            ordered_parent_ids.append(pid)

    if not ordered_parent_ids:
        return rerank_result

    try:
        response = client.mget(
            index=settings.elasticsearch_index_name,
            body={"ids": ordered_parent_ids},
        )
    except Exception as exc:
        logger.warning(
            "parent_fetch: mget failed for ids=%s; returning original chunks. error=%s",
            ordered_parent_ids,
            exc,
        )
        return rerank_result

    docs_by_id = {doc["_id"]: doc for doc in response.get("docs", []) if doc.get("found")}

    parents: list[RetrievedChunk] = []
    for pid in ordered_parent_ids:
        doc = docs_by_id.get(pid)
        if not doc:
            continue

        source = doc["_source"]

        # `page_number` is required in ChunkMetadata; default defensively.
        raw_page = source.get("page_number")
        page_number = raw_page if isinstance(raw_page, int) else 0

        parents.append(
            RetrievedChunk(
                chunk_id=doc["_id"],
                text=source["text"],
                metadata=ChunkMetadata(
                    filename=source["filename"],
                    filetype=source["filetype"],
                    page_number=page_number,
                ),
                parent_id=source.get("parent_id"),
            )
        )

    if not parents:
        logger.warning(
            "parent_fetch: no parent docs found for ids=%s, returning original chunks as fallback",
            ordered_parent_ids,
        )
        return rerank_result

    return parents
