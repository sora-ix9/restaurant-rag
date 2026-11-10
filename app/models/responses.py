"""Response models."""
from pydantic import BaseModel


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str


class SourceDocument(BaseModel):
    """A retrieved chunk used to ground the answer."""

    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    """Grounded answer plus the documents it was built from."""

    answer: str
    sources: list[SourceDocument]
    # The name that resolved, never the requested None — that is what makes
    # D11's resolution observable from outside (D12).
    prompt_set: str
