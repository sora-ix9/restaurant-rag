"""Shared prompt templates for LLM providers."""

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


def build_messages(question: str, context: str) -> list[dict]:
    """Build the system+user message list for a grounded-answer call."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(context=context, question=question),
        },
    ]
