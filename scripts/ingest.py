"""Ingest a document corpus into Milvus.

Usage:
    python -m scripts.ingest
    python -m scripts.ingest --pattern "./data/**/*.md" --keep-existing
    python -m scripts.ingest --pattern "./data/reviews.csv" \\
        --database retail_db --collection reviews \\
        --columns review,rating --meta-columns rating --limit 500
"""
import argparse
import logging
import sys

from app.config import settings
from app.services import milvus_service


def _split_columns(value: str | None) -> list[str] | None:
    """Parse a comma-separated column list, ignoring blanks."""

    if not value:
        return None

    return [name.strip() for name in value.split(",") if name.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into Milvus.")
    parser.add_argument(
        "--pattern",
        default=settings.corpus_glob,
        help="Glob pattern for the corpus (.md, .csv, .json, .jsonl, .pdf).",
    )
    parser.add_argument(
        "--database",
        default=None,
        help=f"Milvus database, created if absent (default: {settings.milvus_database}).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=f"Milvus collection (default: {settings.milvus_collection}).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Append to the existing collection instead of dropping it first.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed at most N chunks. The whole corpus is still parsed.",
    )
    parser.add_argument(
        "--columns",
        default=None,
        help="CSV only: comma-separated column names. Supplying these treats "
        "the file as headerless, preserving row 0 as data.",
    )
    parser.add_argument(
        "--text-columns",
        default=None,
        help="CSV only: columns to embed (default: all except --meta-columns).",
    )
    parser.add_argument(
        "--meta-columns",
        default=None,
        help="CSV only: columns stored as metadata but never embedded.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    database = args.database or settings.milvus_database
    collection = args.collection or settings.milvus_collection

    count = milvus_service.ingest(
        pattern=args.pattern,
        collection=collection,
        database=database,
        drop_existing=not args.keep_existing,
        limit=args.limit,
        columns=_split_columns(args.columns),
        text_columns=_split_columns(args.text_columns),
        meta_columns=_split_columns(args.meta_columns),
    )

    print(f"Ingested {count} chunks into '{database}/{collection}'.")

    return 0 if count else 1


if __name__ == "__main__":
    sys.exit(main())
