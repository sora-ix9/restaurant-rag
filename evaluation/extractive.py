"""Slice the 5-line analyst report down to its extractive lines (D1, D4)."""
import re

# Lines 4-5 (Actions, Priority) are prescriptive: a recommendation cannot be
# entailed by a source, because sources describe what happened and these state
# what should be done. Measured at 55.5% of generated text over the 100 rows in
# data/eval/rag_responses.jsonl, capping faithfulness at ~0.445 regardless of
# how well-grounded lines 1-3 are. See docs/part_1/faithfulness_diagnosis.md 6.1.
EXTRACTIVE = ("sentiment", "issues", "precedent")

# "1." and "1)" both occur in the saved responses; ^ stops a mid-line
# "Precedent:" inside an Actions line from being kept. Mirrored in
# tests/load/query_assess.k6.js -- T10 pins the two together.
_EXTRACTIVE_RE = re.compile(rf"^\s*(\d[.)]\s*)?({'|'.join(EXTRACTIVE)})\s*:", re.I)

_SENTIMENT_RE = re.compile(r"^\s*(\d[.)]\s*)?sentiment\s*:\s*([A-Za-z]+)", re.I | re.M)


def extractive_only(answer: str) -> str:
    """Keep the Sentiment/Issues/Precedent lines; drop Actions/Priority."""

    return "\n".join(ln for ln in answer.splitlines() if _EXTRACTIVE_RE.match(ln))


def sentiment_label(answer: str) -> str:
    """The generated Sentiment label, lower-cased, or "unparsed".

    Used only to record the realized sentiment mix of an unstratified sample
    (D5, Risk 17) -- it describes the set, it does not stratify it.
    """

    match = _SENTIMENT_RE.search(answer)
    return match.group(2).lower() if match else "unparsed"
