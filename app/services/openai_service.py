"""OpenAI client wrapper: embeddings and grounded answer generation."""
import logging
from functools import lru_cache

import tiktoken
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# Input cap for text-embedding-3-*. Over-long input is rejected with a 400,
# which would discard the whole batch it was sent in — so truncate instead.
MAX_EMBED_TOKENS = 8191
EMBED_ENCODING = "cl100k_base"

SYSTEM_PROMPT = """\
You are a retail intelligence assistant. Answer the user's question using only the
contextual passage snippets provided. If the context does not contain enough
information to answer, say so explicitly instead of guessing."""

USER_PROMPT_TEMPLATE = """\
Use the following pieces of information enclosed in <context> tags to provide an
answer to the question enclosed in <question> tags.

<context>
{context}
</context>

<question>
{question}
</question>"""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Return a process-wide OpenAI client."""

    if settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key)

    # Falls back to the OPENAI_API_KEY environment variable.
    return OpenAI()


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    """Return the tokenizer used by the embedding models."""

    return tiktoken.get_encoding(EMBED_ENCODING)


def truncate_to_tokens(text: str, max_tokens: int = MAX_EMBED_TOKENS) -> str:
    """Truncate text to the embedding model's input cap.

    A no-op for normal-length text. Applied here rather than in each loader so
    one guard covers every format.
    """

    encoding = get_encoding()
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    logger.warning("Truncating %d tokens to %d", len(tokens), max_tokens)

    return encoding.decode(tokens[:max_tokens])


def embed_text(text: str) -> list[float]:
    """Embed a single string."""

    return embed_texts([text])[0]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings, preserving input order."""

    if not texts:
        return []

    response = get_client().embeddings.create(
        input=[truncate_to_tokens(text) for text in texts],
        model=settings.embedding_model,
    )

    # The API may return items out of order; sort by index to be safe.
    return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


def generate_answer(question: str, context: str) -> str:
    """Generate an answer grounded in the retrieved context."""

    response = get_client().chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    context=context,
                    question=question,
                ),
            },
        ],
    )

    return (response.choices[0].message.content or "").strip()
