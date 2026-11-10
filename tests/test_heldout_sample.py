"""Held-out sampler tests (D5, C11). No Milvus, no network, no credentials —
sampling is pure, and S5 reads the local corpus CSVs if they exist."""
from pathlib import Path

import pytest

from evaluation.heldout import (
    TEST_FILE,
    TRAIN_FILE,
    load_test_reviews,
    load_train_texts,
    sample_heldout,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]


def rows(count: int) -> list[dict]:
    """Synthetic test_file.csv rows, as csv.DictReader yields them."""
    return [{"reviewID": str(i), "review": f"review text {i}"} for i in range(1, count + 1)]


# --------------------------------------------------------------------------
# Sampling — S1-S3
# --------------------------------------------------------------------------


def test_sample_heldout_is_deterministic():
    """S1: the sample IS the eval set, so the seed must pin it."""
    pool = rows(50)

    first = [r["review_id"] for r in sample_heldout(pool, set(), 10, seed=7)]
    again = [r["review_id"] for r in sample_heldout(pool, set(), 10, seed=7)]
    other = [r["review_id"] for r in sample_heldout(pool, set(), 10, seed=8)]

    assert first == again
    assert first != other


def test_sample_heldout_filters_before_sampling():
    """S2: an indexed review would retrieve itself at ~1.0 (C11)."""
    pool = rows(20)
    train_texts = {"review text 3", "review text 11"}

    # n equals the filtered pool, so only filter-then-sample can succeed.
    picked = sample_heldout(pool, train_texts, len(pool) - len(train_texts))

    assert len(picked) == 18
    assert not {r["text"] for r in picked} & train_texts


def test_sample_heldout_schema():
    """S3: the D5 data model, text verbatim."""
    pool = rows(10)
    pool[4]["review"] = "  line one\nline two  "

    picked = sample_heldout(pool, set(), len(pool))

    assert [r["id"] for r in picked] == list(range(1, len(pool) + 1))
    for r in picked:
        assert set(r) == {"id", "text", "review_id", "source"}
        assert r["source"] == f"test_file.csv#{r['review_id']}"
    assert "  line one\nline two  " in {r["text"] for r in picked}


# --------------------------------------------------------------------------
# Writing — S4
# --------------------------------------------------------------------------


def test_write_jsonl_never_overwrites(tmp_path):
    """S4: a re-sample under the same name must fail loudly (D9)."""
    path = tmp_path / "sample.jsonl"
    write_jsonl([{"id": 1}], path)

    with pytest.raises(FileExistsError):
        write_jsonl([{"id": 2}], path)
    assert path.read_text(encoding="utf-8") == '{"id": 1}\n'


# --------------------------------------------------------------------------
# The real corpus — S5
# --------------------------------------------------------------------------


def test_load_test_reviews_real_source():
    """S5: 6,203 held-out rows; 8 exact training matches, measured in C11."""
    test_file, train_file = ROOT / TEST_FILE, ROOT / TRAIN_FILE
    if not (test_file.exists() and train_file.exists()):
        pytest.skip("Wongnai corpus absent under data/ -- untracked by design")

    test_rows = load_test_reviews(test_file)
    train_texts = load_train_texts(train_file)
    excluded = sum(r["review"].strip() in train_texts for r in test_rows)

    assert len(test_rows) == 6203
    assert all(set(r) == {"reviewID", "review"} for r in test_rows)
    assert excluded == 8
