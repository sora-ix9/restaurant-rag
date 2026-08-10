"""Per-format document loaders.

Every loader takes a `Path` and returns a list of chunks shaped as
`{"text": str, "source": str, "meta": dict}`. `meta` is optional and defaults
to `{}` — only the CSV loader populates it (see `--meta-columns`).
"""
import csv
import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Candidate CSV delimiters. Comma covers most exports; semicolon is what Excel
# writes in EU locales, where the comma is the decimal separator.
SNIFF_DELIMITERS = ",;"
CSV_SAMPLE_BYTES = 8192


def sniff_delimiter(sample: str) -> str:
    """Detect the CSV delimiter, falling back to whichever candidate dominates.

    `csv.Sniffer` handles quoted fields correctly but gives up on prose-heavy
    and single-column files. Counting is cruder, but it never silently picks a
    delimiter the file does not actually contain.
    """

    try:
        return csv.Sniffer().sniff(sample, delimiters=SNIFF_DELIMITERS).delimiter
    except csv.Error:
        counts = {d: sample.count(d) for d in SNIFF_DELIMITERS}
        best = max(counts, key=counts.get)
        if counts[best]:
            logger.info("Sniffer failed; counted delimiter %r", best)
            return best
        # No candidate present at all — a single-column file reads fine as CSV.
        return ","


def _render_pairs(pairs: dict) -> str:
    """Render `{key: value}` as `key: value` lines, dropping empty entries.

    Column names carry real semantic signal for an embedding model, so a row
    reads as `product_name: Blue Scarf` rather than a bare delimited line.
    """

    return "\n".join(
        f"{str(key).strip()}: {str(value).strip()}"
        for key, value in pairs.items()
        if key and value is not None and str(value).strip()
    )


def load_markdown(path: Path) -> list[dict]:
    """Split a markdown file on `# ` headings, one chunk per section."""

    file_text = path.read_text(encoding="utf-8")

    return [
        {"text": text, "source": path.name, "meta": {}}
        for part in file_text.split("# ")
        if (text := part.strip())
    ]


def load_csv(
    path: Path,
    columns: list[str] | None = None,
    text_columns: list[str] | None = None,
    meta_columns: list[str] | None = None,
) -> list[dict]:
    """One chunk per row.

    `columns` supplies field names for a headerless file — passing it also
    stops `DictReader` from consuming row 0 as a header. `meta_columns` are
    stored on the chunk but never embedded.
    """

    meta_columns = meta_columns or []

    with path.open(encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(CSV_SAMPLE_BYTES)
        fh.seek(0)

        reader = csv.DictReader(
            fh, fieldnames=columns, delimiter=sniff_delimiter(sample)
        )

        chunks: list[dict] = []
        for i, row in enumerate(reader, start=1):
            embedded = text_columns or [
                key for key in row if key and key not in meta_columns
            ]
            text = _render_pairs({key: row.get(key) for key in embedded})
            if not text:
                continue

            chunks.append(
                {
                    "text": text,
                    "source": f"{path.name}#row{i}",
                    "meta": {key: row.get(key, "") for key in meta_columns},
                }
            )

    return chunks


def load_json(path: Path) -> list[dict]:
    """Load JSONL (one object per line) or a JSON array / single object."""

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        parsed = json.loads(raw)
        records = parsed if isinstance(parsed, list) else [parsed]

    return [
        {"text": text, "source": f"{path.name}#{i}", "meta": {}}
        for i, record in enumerate(records, start=1)
        if (text := _render_record(record))
    ]


def _render_record(record: object) -> str:
    """Render a record as `key: value` lines; non-dicts stringify directly."""

    if not isinstance(record, dict):
        return str(record).strip()

    return _render_pairs(record)


def load_pdf(path: Path) -> list[dict]:
    """One chunk per page. Pages without extractable text are skipped."""

    # Imported lazily so projects that never ingest PDFs need not install pypdf.
    from pypdf import PdfReader

    reader = PdfReader(str(path))

    return [
        {"text": text, "source": f"{path.name}#page{i}", "meta": {}}
        for i, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    ]


LOADERS: dict[str, Callable[..., list[dict]]] = {
    ".md": load_markdown,
    ".csv": load_csv,
    ".json": load_json,
    ".jsonl": load_json,
    ".pdf": load_pdf,
}

# Options that only the CSV loader accepts; ignored for every other format.
CSV_ONLY_KWARGS = {"columns", "text_columns", "meta_columns"}


def load_file(path: Path, **kwargs) -> list[dict]:
    """Load one file into chunks, dispatching on its extension.

    An unsupported or unreadable file logs and yields nothing rather than
    raising: a malformed file 200 of 300 must not discard the 199 files whose
    embeddings have already been paid for.
    """

    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        logger.warning("Skipping %s: unsupported extension %s", path.name, path.suffix)
        return []

    if loader is not load_csv:
        kwargs = {k: v for k, v in kwargs.items() if k not in CSV_ONLY_KWARGS}

    try:
        return loader(path, **kwargs)
    except Exception:
        logger.exception("Skipping %s: loader failed", path.name)
        return []
