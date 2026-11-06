"""Extractive-slice tests (D1, D4). No Milvus, no network, no credentials —
the slice is a pure function, and T9 reads one local JSONL if it exists."""
import json
from pathlib import Path

import pytest

from evaluation.extractive import _EXTRACTIVE_RE, extractive_only, sentiment_label

ROOT = Path(__file__).resolve().parents[1]
RESPONSE_FILE = ROOT / "data" / "eval" / "rag_responses.jsonl"
K6_SCRIPT = ROOT / "tests" / "load" / "query_assess.k6.js"

# The conforming shape: row 1 of data/eval/rag_responses.jsonl.
CONFORMING = (
    "Sentiment: Positive / 0.85 / driving phrase: บรรยากาศดี, นั่งชิลล์\n"
    "Issues: none\n"
    "Precedent: not applicable\n"
    "Actions: protect ambience consistency; maintain attentive service\n"
    "Priority: low — focus on sustaining ambience as a differentiator"
)

# The shape of ids 67 and 86: Thai line labels, so nothing English-keyed survives.
THAI_LABELLED = (
    "1. บวก, ความมั่นใจ 0.8, ขับเคลื่อนด้วย เบียร์เย็น\n"
    "2. none\n"
    "3. ไม่สามารถอ้างอิงได้ (ไม่มีประเด็นในชุดข้อมูล) แหล่งที่มา: N/A\n"
    "4. Actions: รักษาคุณภาพเบียร์และบรรยากาศอิซากายะ\n"
    "5. Low priority: ไม่มีประเด็นด้านความปลอดภัยอาหารหรือปัญหาซ้ำซ้อน"
)


# --------------------------------------------------------------------------
# The slice — T1-T8
# --------------------------------------------------------------------------


def test_extractive_only_keeps_extractive_lines():
    """T1."""
    assert extractive_only(CONFORMING).splitlines() == CONFORMING.splitlines()[:3]


def test_extractive_only_drops_prescriptive_lines():
    """T2."""
    kept = extractive_only(CONFORMING)

    assert "Actions:" not in kept
    assert "Priority:" not in kept


def test_extractive_only_accepts_both_numbering_styles():
    """T3: "1." and "1)" both occur in the saved responses."""
    answer = "1. Sentiment: negative 0.7\n2) Issues: - [wait] slow\n4. Actions: fix it"

    assert extractive_only(answer) == "1. Sentiment: negative 0.7\n2) Issues: - [wait] slow"


def test_extractive_only_is_anchored():
    """T4: a label mid-sentence is not a line label."""
    answer = "Actions: check the Precedent: line before acting\nIssues: none"

    assert extractive_only(answer) == "Issues: none"


def test_extractive_only_is_case_insensitive():
    """T5."""
    answer = "SENTIMENT: mixed 0.5\nsentiment: mixed 0.5"

    assert extractive_only(answer) == answer


def test_extractive_only_preserves_order():
    """T6."""
    answer = "Precedent: one case\nActions: none\nSentiment: neutral 0.5\nIssues: none"

    assert extractive_only(answer) == "Precedent: one case\nSentiment: neutral 0.5\nIssues: none"


def test_extractive_only_thai_labelled_response_is_empty():
    """T7: Risk 3 — dropped and counted, never scored 0."""
    assert extractive_only(THAI_LABELLED) == ""


def test_extractive_only_empty_input_is_safe():
    """T8."""
    assert extractive_only("") == ""


# --------------------------------------------------------------------------
# The known corpus — T9
# --------------------------------------------------------------------------


def test_extractive_only_known_corpus_drop_count():
    """T9: D4's measured 98/2 over the 100 saved analyst responses."""
    if not RESPONSE_FILE.exists():
        pytest.skip("data/eval/rag_responses.jsonl absent -- untracked by design (Q1=B)")

    rows = [json.loads(line) for line
            in RESPONSE_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    dropped = [r["id"] for r in rows if not extractive_only(r["response"]).strip()]

    assert len(rows) == 100
    assert len(rows) - len(dropped) == 98
    assert dropped == [67, 86]


# --------------------------------------------------------------------------
# Parity and parsing — T10, T11
# --------------------------------------------------------------------------


def test_extractive_regex_matches_k6_script():
    """T10: the k6 script carries a JS copy of the regex; the two must not drift."""
    js_literal = "/" + _EXTRACTIVE_RE.pattern + "/i"

    assert js_literal in K6_SCRIPT.read_text(encoding="utf-8")


def test_sentiment_label_parses_generated_label():
    """T11: records the realized sentiment mix (D5, Risk 17)."""
    assert sentiment_label("Sentiment: Positive / 0.85 / driving phrase: x") == "positive"
    assert sentiment_label("1) Sentiment: negative, 0.6") == "negative"
    assert sentiment_label(THAI_LABELLED) == "unparsed"
