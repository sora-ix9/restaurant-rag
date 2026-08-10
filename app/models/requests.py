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
