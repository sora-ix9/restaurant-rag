"""Milvus service: corpus ingestion and dense vector search."""
import logging
from functools import lru_cache
from glob import glob
from pathlib import Path

from pymilvus import MilvusClient

from app.config import settings
from app.services.loaders import load_file
from app.services.openai_service import embed_text, embed_texts

logger = logging.getLogger(__name__)

# Documents are embedded in batches to keep the number of API calls low.
EMBED_BATCH_SIZE = 64


@lru_cache(maxsize=None)
def get_client(database: str | None = None) -> MilvusClient:
    """Return a per-database Milvus client, creating the database if absent.

    Cached on the database name so alternating between databases does not
    reconnect, and so a cached client's database can never drift (the name is
    bound at construction rather than via `use_database`).
    """

    name = database or settings.milvus_database

    # A client bound to a non-existent db_name fails on first use rather than
    # at construction, so ensure the database exists via a default-scoped one.
    if name != "default":
        admin = MilvusClient(uri=settings.milvus_uri)
        if name not in admin.list_databases():
            logger.info("Creating database %s", name)
            admin.create_database(name)

    return MilvusClient(uri=settings.milvus_uri, db_name=name)


def load_corpus(pattern: str | None = None, **loader_kwargs) -> list[dict]:
    """Load every file matching the glob, dispatching each by extension.

    Returns a list of `{"text": ..., "source": ..., "meta": ...}` chunks.
    """

    pattern = pattern or settings.corpus_glob
    chunks: list[dict] = []

    for file_path in sorted(glob(pattern, recursive=True)):
        path = Path(file_path)
        # A recursive glob returns directories too; `suffix` on one would miss
        # the loader table and log a spurious unsupported-extension warning.
        if path.is_file():
            chunks.extend(load_file(path, **loader_kwargs))

    logger.info("Loaded %d chunks from %s", len(chunks), pattern)

    return chunks


def create_collection(
    dimension: int,
    collection: str | None = None,
    database: str | None = None,
    drop_existing: bool = True,
) -> None:
    """Create the Milvus collection, optionally dropping any existing one."""

    client = get_client(database)
    name = collection or settings.milvus_collection

    if client.has_collection(name):
        if not drop_existing:
            logger.info("Collection %s already exists; skipping creation", name)
            return
        logger.info("Dropping existing collection %s", name)
        client.drop_collection(name)

    client.create_collection(
        collection_name=name,
        dimension=dimension,
        metric_type=settings.milvus_metric_type,
        consistency_level="Bounded",
        auto_id=False,
    )

    logger.info("Created collection %s (dim=%d)", name, dimension)


def ingest(
    pattern: str | None = None,
    collection: str | None = None,
    database: str | None = None,
    drop_existing: bool = True,
    limit: int | None = None,
    **loader_kwargs,
) -> int:
    """Ingest the corpus into Milvus. Returns the number of chunks inserted."""

    name = collection or settings.milvus_collection

    chunks = load_corpus(pattern, **loader_kwargs)
    if not chunks:
        logger.warning("No chunks found for pattern %s; nothing to ingest", pattern)
        return 0

    # Capped after loading, before embedding: the whole file is still parsed
    # (so malformed rows anywhere surface), but only `limit` chunks are paid for.
    if limit:
        logger.info("Limiting to %d of %d chunks", limit, len(chunks))
        chunks = chunks[:limit]

    # Probe the embedding dimension so the collection schema matches the model.
    dimension = len(embed_text(chunks[0]["text"]))
    create_collection(dimension, name, database, drop_existing=drop_existing)

    client = get_client(database)

    # Appending must not reuse primary keys — Milvus upserts on ID collision,
    # so restarting IDs at 0 would overwrite rows 0..N of the existing data.
    next_id = 0
    if not drop_existing and client.has_collection(name):
        next_id = client.get_collection_stats(name).get("row_count", 0)
        logger.info("Appending to %s from id %d", name, next_id)

    inserted = 0

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[start : start + EMBED_BATCH_SIZE]
        vectors = embed_texts([chunk["text"] for chunk in batch])

        rows = [
            {
                "id": next_id + start + offset,
                "vector": vector,
                "text": chunk["text"],
                "source": chunk["source"],
                "meta": chunk.get("meta", {}),
            }
            for offset, (chunk, vector) in enumerate(zip(batch, vectors))
        ]

        client.insert(collection_name=name, data=rows)
        inserted += len(rows)
        logger.info("Inserted %d/%d chunks", inserted, len(chunks))

    return inserted


def search(query: str, top_k: int | None = None) -> list[dict]:
    """Embed the query and return the top-K nearest chunks.

    Each result is `{"text": ..., "source": ..., "score": ...}`, best first.
    """

    top_k = top_k or settings.top_k

    results = get_client().search(
        collection_name=settings.milvus_collection,
        data=[embed_text(query)],
        limit=top_k,
        search_params={"metric_type": settings.milvus_metric_type, "params": {}},
        output_fields=["text", "source"],
    )

    if not results:
        return []

    return [
        {
            "text": hit["entity"]["text"],
            "source": hit["entity"].get("source", ""),
            "score": hit["distance"],
        }
        for hit in results[0]
    ]
