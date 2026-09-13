"""Request models."""
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """A natural-language question against the retail knowledge base."""

    question: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Number of chunks to retrieve. Defaults to the configured top_k.",
    )
    provider: Literal["openai", "qwen"] | None = Field(
        default=None,
        description="LLM provider to use. Overrides config default if specified.",
    )
    # The Literal duplicates PROMPT_SETS.keys() deliberately: deriving it would
    # lose the OpenAPI enum, which is what renders a dropdown in /docs. A parity
    # test keeps the duplication checked rather than trusted (D11, Risk 12).
    prompt: Literal["default", "assess", "strict", "thai"] | None = Field(
        default=None,
        description=(
            "Named prompt set. Overrides config default if specified. "
            '"assess" returns a fixed 5-line analyst report rather than prose.'
        ),
    )
