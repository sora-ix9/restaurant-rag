"""Shared prompt templates and the named prompt-set registry (D7, D10).

Provider modules own no prompt text: both `openai_service` and `qwen_service`
build their messages here, so one named set means the same instruction on both
sides of the provider switch. That is what makes a GPT-vs-Qwen difference
attributable to the model rather than to the prompt (D6).
"""
from dataclasses import dataclass

from app.config import settings


# ── default ─── D7's QA prompt. The control in every comparison; pinned
#                byte-for-byte by tests/test_prompts.py. Changing it invalidates
#                every prior measurement — add a new named set instead (Risk 11).
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


# ── assess ──── The 5-line analyst report. Lived in openai_service.py until
#                drift 1 was closed; moved here VERBATIM because it generated
#                every row of data/eval/rag_responses.jsonl and is what D13
#                scores. Not one of D10's three seed names — see the plan's C1.
ASSESS_SYSTEM_PROMPT = """\
You are a restaurant operations analyst. You get one customer review plus similar
past reviews from this restaurant's corpus.

Ground every claim. Sentiment and issues come from the new review. Any claim that
an issue is recurring must cite a retrieved review; if none support it, say the
corpus doesn't show it. Never invent facts about staffing, suppliers, or prices.
If retrieved reviews are unrelated, say so and assess the review alone.

Reply in the review's language. Be brief — an operator reads this."""

ASSESS_USER_TEMPLATE = """\
<past_reviews>
{context}
</past_reviews>

<review>
{question}
</review>

Assess in exactly these 5 lines:

1. Sentiment: positive/negative/mixed/neutral + confidence 0-1 + the phrase driving it
2. Issues: bullets, each tagged [taste|portion|freshness|service|wait|price|cleanliness|ambience]. "none" if none
3. Precedent: per issue — recurring / one case / not in corpus, + source id
4. Actions: if negative/mixed, 2-3 fixes tied to issue numbers. Tag guesses "verify:". If positive, what to protect
5. Priority: high/med/low + one clause. High = food safety or repeated failure"""

# ── strict ──── Refusal made mandatory rather than merely encouraged, and the
#                passage relied on must be quoted. Exists to answer: is D7's
#                refusal clause strong enough? (Risk 8). Shares D7's user
#                template — only the system half differs.
STRICT_SYSTEM_PROMPT = """\
You are a retail intelligence assistant. Answer only what the contextual passage
snippets state. Quote the passage you relied on verbatim before answering from it.

If the context does not state the answer, reply exactly "The context does not
contain this information." and nothing else. Do not infer, extrapolate, or draw
on knowledge from outside the passages — an unsupported answer is worse than no
answer."""


# ── thai ────── The same instruction written in Thai, answering in the question's
#                language. Exists to answer: does an English instruction frame
#                degrade Thai answers on a Thai corpus? Needs its own template so
#                the text around the tags is Thai too.
THAI_SYSTEM_PROMPT = """\
คุณเป็นผู้ช่วยวิเคราะห์ข้อมูลร้านค้า ตอบคำถามของผู้ใช้โดยอ้างอิงจากข้อความที่ให้มาเท่านั้น
หากข้อมูลที่ให้มาไม่เพียงพอต่อการตอบ ให้บอกตรง ๆ ว่าไม่พบข้อมูล อย่าเดาหรือแต่งขึ้นเอง

ตอบเป็นภาษาเดียวกับคำถาม"""

THAI_USER_TEMPLATE = """\
ใช้ข้อมูลในแท็ก <context> เพื่อตอบคำถามในแท็ก <question>

<context>
{context}
</context>

<question>
{question}
</question>"""


@dataclass(frozen=True)
class PromptSet:
    """One switchable instruction: a system message and a user template.

    Both halves switch together. The user template's tags are what the system
    prompt's "contextual passage snippets" refers to, so swapping one without
    the other can break that reference silently (D10).

    Frozen because the registry is read from module scope on every request — a
    mutable set would let one request change what a later one sees.
    """

    system: str
    user_template: str


PROMPT_SETS: dict[str, PromptSet] = {
    "default": PromptSet(SYSTEM_PROMPT, USER_PROMPT_TEMPLATE),
    "assess": PromptSet(ASSESS_SYSTEM_PROMPT, ASSESS_USER_TEMPLATE),
    "strict": PromptSet(STRICT_SYSTEM_PROMPT, USER_PROMPT_TEMPLATE),
    "thai": PromptSet(THAI_SYSTEM_PROMPT, THAI_USER_TEMPLATE),
}


def build_messages(question: str, context: str, prompt: str | None = None) -> list[dict]:
    """Build the system+user message list for a grounded-answer call.

    Args:
        question: The user's question.
        context: Retrieved document chunks joined together.
        prompt: Named prompt set. Falls back to `settings.prompt_set` (D11).

    Raises:
        KeyError: If `prompt` names no set in the registry. Callers on the API
            path never see this — `QueryRequest.prompt` is a Literal, so an
            unknown name is a 422 at the boundary (D2).
    """

    prompt_set = PROMPT_SETS[prompt or settings.prompt_set]
    return [
        {"role": "system", "content": prompt_set.system},
        {
            "role": "user",
            "content": prompt_set.user_template.format(context=context, question=question),
        },
    ]
