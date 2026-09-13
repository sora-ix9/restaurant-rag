"""Loader tests. No Milvus, no network — every loader is a pure function."""
import json

import pytest

from app.services.loaders import (
    load_csv,
    load_file,
    load_json,
    load_markdown,
    sniff_delimiter,
)
from app.services.openai_service import MAX_EMBED_TOKENS, get_encoding, truncate_to_tokens


def write(tmp_path, name: str, content: str, encoding: str = "utf-8"):
    """Write a temp file and return its path."""
    path = tmp_path / name
    path.write_text(content, encoding=encoding)
    return path


# --------------------------------------------------------------------------
# Delimiter detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample,expected",
    [
        ("a,b\n1,2\n", ","),
        ("a;b\n1;2\n", ";"),
        # Sniffer fails on these; the counting fallback must still be right.
        ("long thai prose;5\nmore prose;4\n", ";"),
        ("single\ncolumn\nvalues\n", ","),
    ],
)
def test_sniff_delimiter(sample, expected):
    assert sniff_delimiter(sample) == expected


def test_sniff_delimiter_survives_sniffer_failure():
    """The real Wongnai shape: semicolons, no header, prose fields."""
    sample = "".join(f"review text number {i} with words;{i % 5 + 1}\n" for i in range(50))
    assert sniff_delimiter(sample) == ";"


# --------------------------------------------------------------------------
# Markdown — must reproduce pre-refactor behaviour exactly
# --------------------------------------------------------------------------


def test_load_markdown_splits_on_headings(tmp_path):
    path = write(tmp_path, "doc.md", "# One\nalpha\n# Two\nbeta\n")
    chunks = load_markdown(path)

    assert len(chunks) == 2
    assert chunks[0]["text"] == "One\nalpha"
    assert chunks[0]["source"] == "doc.md"
    assert chunks[0]["meta"] == {}


def test_load_markdown_drops_empty_sections(tmp_path):
    path = write(tmp_path, "doc.md", "# \n\n# Real\ncontent\n")
    assert len(load_markdown(path)) == 1


# --------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------


def test_load_csv_comma(tmp_path):
    path = write(tmp_path, "d.csv", "a,b\n1,2\n")
    chunks = load_csv(path)

    assert len(chunks) == 1
    assert chunks[0]["text"] == "a: 1\nb: 2"
    assert chunks[0]["source"] == "d.csv#row1"


def test_load_csv_semicolon_matches_comma(tmp_path):
    """Same data, different delimiter, identical chunks."""
    comma = load_csv(write(tmp_path, "c.csv", "a,b\n1,2\n"))
    semi = load_csv(write(tmp_path, "s.csv", "a;b\n1;2\n"))

    assert comma[0]["text"] == semi[0]["text"]


def test_load_csv_strips_bom(tmp_path):
    path = write(tmp_path, "bom.csv", "a,b\n1,2\n", encoding="utf-8-sig")
    assert load_csv(path)[0]["text"].startswith("a: 1")


def test_load_csv_drops_empty_cells(tmp_path):
    path = write(tmp_path, "d.csv", "a,b\n1,\n")
    assert load_csv(path)[0]["text"] == "a: 1"


def test_load_csv_single_column(tmp_path):
    path = write(tmp_path, "d.csv", "a\n1\n2\n")
    assert len(load_csv(path)) == 2


def test_load_csv_header_only(tmp_path):
    path = write(tmp_path, "d.csv", "a,b\n")
    assert load_csv(path) == []


def test_load_csv_headerless_preserves_row_zero(tmp_path):
    """--columns names the fields AND stops row 0 being eaten as a header."""
    path = write(tmp_path, "d.csv", "first review;5\nsecond review;3\n")
    chunks = load_csv(path, columns=["review", "rating"])

    assert len(chunks) == 2
    assert "review: first review" in chunks[0]["text"]
    assert "rating: 5" in chunks[0]["text"]


def test_load_csv_meta_columns_excluded_from_text(tmp_path):
    path = write(tmp_path, "d.csv", "great food;5\n")
    chunks = load_csv(path, columns=["review", "rating"], meta_columns=["rating"])

    assert chunks[0]["text"] == "review: great food"
    assert "rating" not in chunks[0]["text"]
    assert chunks[0]["meta"] == {"rating": "5"}


def test_load_csv_meta_defaults_to_empty(tmp_path):
    path = write(tmp_path, "d.csv", "a,b\n1,2\n")
    chunks = load_csv(path)

    assert chunks[0]["meta"] == {}
    assert "b: 2" in chunks[0]["text"]


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def test_load_json_array(tmp_path):
    path = write(tmp_path, "d.json", json.dumps([{"x": 1}, {"x": 2}]))
    chunks = load_json(path)

    assert len(chunks) == 2
    assert chunks[0]["source"] == "d.json#1"


def test_load_json_single_object(tmp_path):
    path = write(tmp_path, "d.json", json.dumps({"x": 1}))
    assert len(load_json(path)) == 1


def test_load_jsonl_skips_blank_lines(tmp_path):
    path = write(tmp_path, "d.jsonl", '{"x": 1}\n\n{"x": 2}\n')
    assert len(load_json(path)) == 2


def test_load_json_drops_empty_values(tmp_path):
    path = write(tmp_path, "d.json", json.dumps({"x": 1, "y": None, "z": ""}))
    assert load_json(path)[0]["text"] == "x: 1"


def test_load_json_empty_file(tmp_path):
    assert load_json(write(tmp_path, "d.json", "")) == []


# --------------------------------------------------------------------------
# Dispatch — must never raise (R4)
# --------------------------------------------------------------------------


def test_load_file_unsupported_extension(tmp_path):
    assert load_file(write(tmp_path, "notes.txt", "hello")) == []


def test_load_file_malformed_json_does_not_raise(tmp_path):
    assert load_file(write(tmp_path, "bad.json", '{"x": ')) == []


def test_load_file_routes_by_extension(tmp_path):
    md = load_file(write(tmp_path, "d.md", "# A\nbody"))
    csv_ = load_file(write(tmp_path, "d.csv", "a,b\n1,2\n"))

    assert md[0]["source"] == "d.md"
    assert csv_[0]["source"] == "d.csv#row1"


def test_load_file_ignores_csv_kwargs_for_other_formats(tmp_path):
    """CSV-only options must not blow up the markdown loader."""
    chunks = load_file(write(tmp_path, "d.md", "# A\nbody"), meta_columns=["x"])
    assert len(chunks) == 1


# --------------------------------------------------------------------------
# Token truncation (D13)
# --------------------------------------------------------------------------


def test_truncate_leaves_short_text_untouched():
    assert truncate_to_tokens("hello world") == "hello world"


def test_truncate_caps_long_text():
    long_text = "word " * 20000
    result = truncate_to_tokens(long_text)

    assert len(get_encoding().encode(result)) == MAX_EMBED_TOKENS
