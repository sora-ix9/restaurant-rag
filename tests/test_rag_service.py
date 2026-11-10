"""Pipeline and API-contract tests. No Milvus, no network — both service
boundaries are patched."""
from typing import Literal, get_args, get_origin
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.models.requests import QueryRequest
from app.services import rag_service
from app.services.prompts import PROMPT_SETS
from app.services.rag_service import NO_CONTEXT_ANSWER, answer_query

DOCUMENT = {"text": "a review", "source": "corpus.csv", "score": 0.9}


def prompt_literal_names() -> set[str]:
    """The names the `prompt` Literal on QueryRequest admits."""

    annotation = QueryRequest.model_fields["prompt"].annotation

    # `Literal[...] | None` — pull the Literal out of the union.
    if get_origin(annotation) is not Literal:
        annotation = next(
            arg for arg in get_args(annotation) if get_origin(arg) is Literal
        )

    return set(get_args(annotation))


# --------------------------------------------------------------------------
# N11, N14 — the empty-retrieval path (D4, D12)
# --------------------------------------------------------------------------


def test_empty_retrieval_short_circuits_without_generating():
    """D4: no documents means no generation call at all — not a call with an
    empty context, which would invite the model to answer from memory."""

    with patch.object(rag_service.milvus_service, "search", return_value=[]):
        with patch.object(rag_service.openai_service, "generate_answer") as generate:
            result = answer_query("q")

    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["sources"] == []
    generate.assert_not_called()


def test_empty_retrieval_still_reports_a_prompt_set(monkeypatch):
    """D12: QueryResponse is one shape on every path. Omitting the key here is a
    500 on the one path that was meant to be free."""

    monkeypatch.setattr(rag_service.settings, "prompt_set", "strict")

    with patch.object(rag_service.milvus_service, "search", return_value=[]):
        result = answer_query("q")

    assert result["prompt_set"] == "strict"


# --------------------------------------------------------------------------
# N13, N15 — `prompt` in, `prompt_set` out (D11, D12)
# --------------------------------------------------------------------------


def test_response_reports_the_resolved_prompt_set(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "prompt_set", "default")

    with patch.object(rag_service.milvus_service, "search", return_value=[DOCUMENT]):
        with patch.object(
            rag_service.openai_service, "generate_answer", return_value="answer"
        ):
            result = answer_query("q", prompt="assess")

    assert result["prompt_set"] == "assess"


def test_response_reports_the_configured_set_when_none_is_requested(monkeypatch):
    """D12: the name that resolved, never the requested None — that is what makes
    D11's resolution observable from outside."""

    monkeypatch.setattr(rag_service.settings, "prompt_set", "thai")

    with patch.object(rag_service.milvus_service, "search", return_value=[DOCUMENT]):
        with patch.object(
            rag_service.openai_service, "generate_answer", return_value="answer"
        ):
            result = answer_query("q")

    assert result["prompt_set"] == "thai"


def test_requested_prompt_reaches_the_provider():
    with patch.object(rag_service.milvus_service, "search", return_value=[DOCUMENT]):
        with patch.object(
            rag_service.openai_service, "generate_answer", return_value="answer"
        ) as generate:
            answer_query("q", prompt="strict")

    assert generate.call_args.kwargs["prompt"] == "strict"


def test_provider_receives_the_resolved_name_not_none(monkeypatch):
    """Resolve once at the top and pass the name down. Resolving again inside
    build_messages is harmless but is how the two drift apart later."""

    monkeypatch.setattr(rag_service.settings, "prompt_set", "strict")

    with patch.object(rag_service.milvus_service, "search", return_value=[DOCUMENT]):
        with patch.object(
            rag_service.openai_service, "generate_answer", return_value="answer"
        ) as generate:
            answer_query("q")

    assert generate.call_args.kwargs["prompt"] == "strict"


# --------------------------------------------------------------------------
# N16, N17 — the provider switch (D5)
# --------------------------------------------------------------------------


def test_provider_override_wins_over_config(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "llm_provider", "openai")

    assert rag_service._get_llm_provider("qwen") is rag_service.qwen_service.qwen_service


def test_provider_defaults_to_config(monkeypatch):
    monkeypatch.setattr(rag_service.settings, "llm_provider", "qwen")

    assert rag_service._get_llm_provider() is rag_service.qwen_service.qwen_service


def test_both_providers_accept_the_same_generate_answer_signature():
    """D5/D6: the pipeline calls one interface. A provider that differs makes the
    switch a special case at the call site."""

    import inspect

    from app.services import openai_service, qwen_service

    openai_params = list(
        inspect.signature(openai_service.generate_answer).parameters
    )
    qwen_params = [
        p
        for p in inspect.signature(
            qwen_service.QwenService.generate_answer
        ).parameters
        if p != "self"
    ]

    assert openai_params == qwen_params == ["question", "context", "prompt"]


# --------------------------------------------------------------------------
# N12 — registry / Literal parity (Risk 12)
# --------------------------------------------------------------------------


def test_request_literal_matches_the_registry_exactly():
    """The duplication is deliberate — deriving the Literal from PROMPT_SETS
    loses the OpenAPI enum (D11). This is what makes it checked, not trusted:
    a name in the Literal only is a dead variant, and one in the registry only
    is a KeyError inside the service that D2 exists to prevent."""

    assert prompt_literal_names() == set(PROMPT_SETS)


# --------------------------------------------------------------------------
# N18 — request validation (D2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "q", "prompt": "nonexistent"},
        {"question": ""},
        {"question": "q", "top_k": 0},
        {"question": "q", "top_k": 51},
        {"question": "q", "provider": "gemini"},
    ],
)
def test_invalid_requests_are_rejected_before_any_service_call(payload):
    """D2: Pydantic rejects at the boundary, so a bad request costs no embedding
    round-trip. The 422 is the framework's; the service never sees these."""

    with pytest.raises(ValidationError):
        QueryRequest(**payload)


@pytest.mark.parametrize("name", sorted(PROMPT_SETS))
def test_every_registry_name_is_accepted_by_the_request_model(name):
    assert QueryRequest(question="q", prompt=name).prompt == name
