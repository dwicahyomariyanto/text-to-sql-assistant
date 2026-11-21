"""The Text-to-SQL prompt template, used CONSISTENTLY in training, evaluation, and the UI."""
from __future__ import annotations

import re
from typing import Iterable, Optional

# The dataset (b-mc2/sql-create-context) is in English, so the model prompt is in English too.
SYSTEM_PROMPT = (
    "You are a Text-to-SQL assistant. Given database table definitions (CREATE TABLE statements) "
    "and a question, reply with a single valid SQLite SELECT query that answers the question. "
    "Output only the SQL query, with no explanation."
)


def build_user_message(schema: str, question: str) -> str:
    return f"### Schema\n{schema.strip()}\n\n### Question\n{question.strip()}"


def build_messages(
    schema: str,
    question: str,
    answer: Optional[str] = None,
    history: Optional[Iterable[tuple[str, str]]] = None,
) -> list[dict]:
    """Build the chat messages.

    - answer given -> training example (includes the assistant turn).
    - answer None  -> inference prompt (no assistant turn).
    - history      -> earlier (question, sql) pairs for multi-turn chat (UI).
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past_question, past_sql in history or []:
        messages.append({"role": "user", "content": build_user_message(schema, past_question)})
        messages.append({"role": "assistant", "content": past_sql})
    messages.append({"role": "user", "content": build_user_message(schema, question)})
    if answer is not None:
        messages.append({"role": "assistant", "content": answer.strip()})
    return messages


_FENCE = re.compile(r"```(?:sql|sqlite)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def extract_sql(text: str) -> str:
    """Extract the SQL from model output (drop code fences and any explanation after the semicolon)."""
    if not text:
        return ""
    text = text.strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    else:
        text = text.replace("```", "").strip()
    text = re.sub(r"^\s*sql\s*:\s*", "", text, flags=re.IGNORECASE)
    # the model sometimes adds an explanation after ';' -> keep only the first SQL statement
    cut = _first_statement_end(text)
    return text[:cut].strip()


def _first_statement_end(text: str) -> int:
    quote = None
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"`":
            quote = ch
        elif ch == ";":
            return i
    return len(text)
