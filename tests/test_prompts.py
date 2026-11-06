"""Prompt registry tests. No Milvus, no network — the registry is pure data."""
from dataclasses import FrozenInstanceError

import pytest

from app.services.prompts import PROMPT_SETS, PromptSet, build_messages

# D7's system prompt, transcribed from the rename ADR's current wording
# (docs/part_1/2-rename-retail-to-restaurant/adr-rename_retail_to_restaurant.md), NOT copied from
# prompts.py. That is the whole point of N2: `default` is the control in every
# comparison, so the pin must come from the specification. Pinning whatever the
# code happens to hold would bless a typo as the baseline (Risk 11).
D7_SYSTEM_PROMPT = """\
You are a restaurant intelligence assistant. Answer the user's question using only the
contextual passage snippets provided. If the context does not contain enough
information to answer, say so explicitly instead of guessing."""

EXPECTED_SETS = {"default", "assess", "strict", "thai"}


# --------------------------------------------------------------------------
# N1-N2 — the registry exists, and `default` is the control
# --------------------------------------------------------------------------


def test_registry_holds_exactly_the_four_named_sets():
    assert set(PROMPT_SETS) == EXPECTED_SETS


@pytest.mark.parametrize("name", sorted(EXPECTED_SETS))
def test_registry_entries_are_prompt_sets_with_both_halves_filled(name):
    prompt_set = PROMPT_SETS[name]

    assert isinstance(prompt_set, PromptSet)
    assert prompt_set.system.strip()
    assert prompt_set.user_template.strip()


def test_default_set_is_byte_identical_to_d7():
    """The control. An innocuous edit here invalidates every prior measurement."""

    assert PROMPT_SETS["default"].system == D7_SYSTEM_PROMPT


# --------------------------------------------------------------------------
# N3-N4, N9 — selection and resolution
# --------------------------------------------------------------------------


def test_build_messages_uses_the_named_set():
    messages = build_messages("q", "c", "strict")

    assert messages[0]["content"] == PROMPT_SETS["strict"].system


def test_build_messages_without_a_name_resolves_to_the_configured_set(monkeypatch):
    monkeypatch.setattr("app.services.prompts.settings.prompt_set", "strict")

    messages = build_messages("q", "c")

    assert messages[0]["content"] == PROMPT_SETS["strict"].system


def test_build_messages_explicit_name_beats_the_configured_set(monkeypatch):
    monkeypatch.setattr("app.services.prompts.settings.prompt_set", "strict")

    messages = build_messages("q", "c", "default")

    assert messages[0]["content"] == PROMPT_SETS["default"].system


def test_build_messages_rejects_an_unknown_set():
    """A KeyError in-service; turning it into a 422 is Pydantic's job (D2)."""

    with pytest.raises(KeyError):
        build_messages("q", "c", "nope")


# --------------------------------------------------------------------------
# N6-N8 — the invariants every set must hold
# --------------------------------------------------------------------------


def test_prompt_sets_are_immutable():
    """Read from module scope on every request — one request must not be able to
    change what a later one sees (D10)."""

    with pytest.raises(FrozenInstanceError):
        PROMPT_SETS["default"].system = "rewritten"


@pytest.mark.parametrize("name", sorted(EXPECTED_SETS))
def test_every_set_instructs_refusal(name):
    """D7's refusal bias is the cheapest guard against a helpful model answering
    anyway; a variant that drops it is not a variant of D7."""

    system = PROMPT_SETS[name].system.lower()
    refusal_markers = ("say so", "don't", "doesn't", "never", "only", "ไม่")

    assert any(marker in system for marker in refusal_markers)


@pytest.mark.parametrize("name", sorted(EXPECTED_SETS))
def test_every_user_template_fills_both_slots(name):
    filled = PROMPT_SETS[name].user_template.format(
        context="CONTEXT-SENTINEL", question="QUESTION-SENTINEL"
    )

    assert "CONTEXT-SENTINEL" in filled
    assert "QUESTION-SENTINEL" in filled


@pytest.mark.parametrize("name", sorted(EXPECTED_SETS))
def test_every_user_template_delimits_its_slots_with_tags(name):
    """Undelimited, a retrieved chunk containing question-like text can read as
    an instruction — not hypothetical on a 40,000-review corpus (D7)."""

    template = PROMPT_SETS[name].user_template

    assert template.count("<") >= 4
    assert template.count(">") >= 4


# --------------------------------------------------------------------------
# N10 — `assess` survived the move (C1)
# --------------------------------------------------------------------------


def test_assess_is_the_analyst_prompt_that_generated_the_saved_responses():
    """C1: `assess` is not a new variant. It is the prompt openai_service already
    ran, moved verbatim. Retyping it makes new scores incomparable to the old."""

    prompt_set = PROMPT_SETS["assess"]

    assert "restaurant operations analyst" in prompt_set.system
    assert "<past_reviews>" in prompt_set.user_template
    assert "<review>" in prompt_set.user_template
    assert "exactly these 5 lines" in prompt_set.user_template


# --------------------------------------------------------------------------
# N5 — the drift-1 test
# --------------------------------------------------------------------------


def test_both_providers_send_the_same_system_message(monkeypatch):
    """The only test that can catch drift 1 reopening.

    Expected red before Step 2 is a TWO-DOCUMENT diff — "restaurant intelligence
    assistant" versus "restaurant operations analyst". A short diff (a stray
    newline) means someone partially closed the drift by pasting text between
    files, which is the failure mode C1 warns against.
    """

    from unittest.mock import MagicMock, patch

    from app.services import openai_service, qwen_service

    question, context = "Q-SENTINEL", "C-SENTINEL"

    openai_client = MagicMock()
    openai_client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="ok"))
    ]
    with patch.object(openai_service, "get_client", return_value=openai_client):
        openai_service.generate_answer(question, context, prompt="assess")
    openai_messages = openai_client.chat.completions.create.call_args.kwargs["messages"]

    service = qwen_service.QwenService()
    service.api_key = "test-key"
    qwen_client = MagicMock()
    qwen_client.post.return_value.json.return_value = {
        "choices": [{"message": {"content": "ok"}}]
    }
    with patch.object(service, "_get_client", return_value=qwen_client):
        service.generate_answer(question, context, prompt="assess")
    qwen_messages = qwen_client.post.call_args.kwargs["json"]["messages"]

    assert openai_messages[0]["content"] == qwen_messages[0]["content"]
    assert openai_messages[1]["content"] == qwen_messages[1]["content"]
