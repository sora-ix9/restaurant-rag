"""Sample the held-out review eval set from the corpus's never-ingested split (D5)."""
import csv
import json
import random
from pathlib import Path

REVIEW_DIR = Path("data/wongnai-corpus-master/review/review_dataset")
TEST_FILE = REVIEW_DIR / "test_file.csv"           # reviewID;review -- WITH a header
TRAIN_FILE = REVIEW_DIR / "w_review_train.csv"     # review;rating  -- headerless
EVAL_FILE = Path("data/eval/reviews_heldout_sample.jsonl")

SAMPLE_SEED = 20261019      # pinned: the sample IS the eval set


def load_test_reviews(path: Path = TEST_FILE) -> list[dict]:
    """Rows of test_file.csv with non-empty review text."""

    with path.open(encoding="utf-8", newline="") as fh:
        return [r for r in csv.DictReader(fh, delimiter=";") if (r["review"] or "").strip()]


def load_train_texts(path: Path = TRAIN_FILE) -> set[str]:
    """Stripped review texts of the indexed split, for exact-match exclusion."""

    with path.open(encoding="utf-8", newline="") as fh:
        return {row[0].strip() for row in csv.reader(fh, delimiter=";") if row}


def sample_heldout(rows: list[dict], train_texts: set[str], n: int,
                   seed: int = SAMPLE_SEED) -> list[dict]:
    """Filter out reviews the index already holds, THEN sample.

    Filtering after sampling would return fewer than n rows, or a different
    sample depending on how the shortfall is refilled. A review that is in the
    index retrieves itself at ~1.0 and makes Precedent trivially "recurring".
    """

    pool = [r for r in rows if r["review"].strip() not in train_texts]
    picked = random.Random(seed).sample(pool, n)
    return [
        {"id": i, "text": r["review"], "review_id": r["reviewID"],
         "source": f"test_file.csv#{r['reviewID']}"}
        for i, r in enumerate(picked, 1)
    ]


def write_jsonl(rows: list[dict], path: Path) -> None:
    """Write once. Mode "x" raises if the file exists -- a re-sample under the
    same name is the silent failure eval_set_sha exists to catch (D10)."""

    with path.open("x", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
