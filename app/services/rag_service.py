"""RAG pipeline: retrieve from Milvus, then generate a grounded answer."""
import logging
from typing import Literal

from app.config import settings
from app.services import milvus_service, openai_service, qwen_service

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = (
    "I could not find any relevant information in the knowledge base "
    "to answer this question."
)


def build_context(documents: list[dict]) -> str:
    """Join retrieved chunks into a single context block for the LLM."""

    return "\n\n".join(doc["text"] for doc in documents)


def _get_llm_provider(provider: Literal["openai", "qwen"] | None = None):
    """Get the appropriate LLM provider based on config or override."""

    effective_provider = provider or settings.llm_provider
    logger.info(f"Using LLM provider: {effective_provider}")

    if effective_provider == "qwen":
        return qwen_service.qwen_service
    return openai_service


def answer_query(
    question: str, top_k: int | None = None, provider: Literal["openai", "qwen"] | None = None
) -> dict:
    """Run the retrieval-augmented generation pipeline for one question."""

    documents = milvus_service.search(question, top_k=top_k)
    logger.info("Retrieved %d documents for query: %s", len(documents), question)

    if not documents:
        return {"answer": NO_CONTEXT_ANSWER, "sources": []}

    llm = _get_llm_provider(provider)
    answer = llm.generate_answer(question, build_context(documents))

    return {"answer": answer, "sources": documents}
