"""Sample the held-out review eval set, or check it for self-retrieval (D5).

Usage (from the repo root):
    python scripts/sample_heldout_reviews.py --n 100            # offline, free
    python scripts/sample_heldout_reviews.py --check-retrieval  # Milvus + OpenAI, ~$0.001

--n never touches a service: it reads two local CSVs and writes one JSONL, and
refuses to overwrite an existing one. --check-retrieval embeds every sampled
review and searches the index once; a top-1 score of ~1.0 means the review is
already indexed and would retrieve itself, making Precedent trivially
"recurring".
"""
import argparse
import json
import sys
from pathlib import Path

# Run as a file, sys.path[0] is scripts/ -- put the repo root there instead so
# `evaluation` and `app` resolve the same way they do under `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.heldout import (  # noqa: E402
    EVAL_FILE,
    SAMPLE_SEED,
    TEST_FILE,
    TRAIN_FILE,
    load_test_reviews,
    load_train_texts,
    sample_heldout,
    write_jsonl,
)

SELF_RETRIEVAL_SCORE = 0.99


def sample(n: int, out: Path) -> int:
    """Filter, sample and write the eval set. Offline."""

    rows = load_test_reviews(TEST_FILE)
    train_texts = load_train_texts(TRAIN_FILE)
    picked = sample_heldout(rows, train_texts, n)
    excluded = sum(r["review"].strip() in train_texts for r in rows)

    write_jsonl(picked, out)

    print(f"pool {len(rows) - excluded:,} of {len(rows):,} "
          f"({excluded} exact training matches excluded)")
    print(f"seed {SAMPLE_SEED}")
    print(f"wrote {len(picked)} rows -> {out.as_posix()}")
    print(f"first review_ids {[r['review_id'] for r in picked[:5]]}")
    return 0


def check_retrieval(path: Path) -> int:
    """Search each sampled review against the index. Needs Milvus and OpenAI."""

    # Imported here, not at the top: constructing these clients needs a running
    # Milvus and credentials, and the offline mode must never do either.
    from app.services import milvus_service

    rows = [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    scores = {}
    for row in rows:
        hits = milvus_service.search(row["text"], top_k=1)
        scores[row["id"]] = hits[0]["score"] if hits else 0.0

    flagged = [i for i, s in scores.items() if s >= SELF_RETRIEVAL_SCORE]

    print(f"checked {len(scores)} rows from {path.as_posix()}")
    print(f"max top-1 score {max(scores.values()):.4f}")
    print(f"ids >= {SELF_RETRIEVAL_SCORE}: {flagged}")
    return 1 if flagged else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Held-out review eval set (D5).")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n", type=int, help="Sample N reviews and write the eval set.")
    mode.add_argument("--check-retrieval", action="store_true",
                      help="Search each sampled review; exit 1 if any retrieves itself.")
    parser.add_argument("--out", type=Path, default=EVAL_FILE,
                        help=f"Eval set path (default: {EVAL_FILE.as_posix()}).")
    args = parser.parse_args()

    if args.check_retrieval:
        return check_retrieval(args.out)
    return sample(args.n, args.out)


if __name__ == "__main__":
    sys.exit(main())
