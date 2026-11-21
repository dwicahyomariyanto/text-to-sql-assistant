"""Helpers that make the assistant robust for free-form questions.

Everything here is pure Python (no torch), so it can be unit-tested anywhere:
  * language detection (English / Indonesian) and prompts for translating the question to English,
  * prompts for LLM-written answers grounded in the query result, with number verification,
  * self-correction feedback prompts for a SQL query that failed,
  * a case-insensitive retry for string comparisons that returned no rows.
"""
from __future__ import annotations

import random
import re
from typing import Optional, Sequence

MAX_ROWS_IN_PROMPT = 20
MAX_ANSWER_CHARS = 700

# ------------------------------------------------------------------ language
_ID_WORDS = {
    "apa", "berapa", "siapa", "dimana", "kapan", "bagaimana", "kenapa", "mengapa", "yang", "dengan", "dari", "untuk",
    "adalah", "ada", "semua", "tampilkan", "daftar", "jumlah", "rata-rata", "tertinggi", "terendah", "terbanyak",
    "tersedikit", "paling", "dan", "atau", "tidak", "pada", "berdasarkan", "setiap", "tiap", "urutkan", "cari",
    "sebutkan", "lebih", "kurang", "antara", "sampai", "hingga", "bulan", "tahun", "hari", "nama", "berasal", "punya",
    "memiliki", "dokter", "pasien", "mahasiswa", "pelanggan", "produk", "harga", "kota", "tolong", "sebanyak", "besar",
    "kecil", "banyak", "hitung", "kelompokkan", "terakhir", "pertama", "di", "ke",
}
_EN_WORDS = {
    "the", "what", "how", "many", "much", "list", "show", "which", "are", "is", "of", "with", "for", "and", "each", "all",
    "average", "total", "where", "who", "when", "does", "do", "have", "has", "from", "in", "by", "than", "more", "less",
    "most", "least", "name", "names", "count", "number", "highest", "lowest", "there",
}


def detect_language(text: str) -> str:
    """Return 'id' for Indonesian-looking text, otherwise 'en'."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]*", (text or "").lower())
    id_hits = sum(t in _ID_WORDS for t in tokens)
    en_hits = sum(t in _EN_WORDS for t in tokens)
    return "id" if id_hits >= 1 and id_hits > en_hits else "en"


LANG_NAME = {"en": "English", "id": "Indonesian (Bahasa Indonesia)"}


def build_translate_messages(question: str, schema: str) -> list[dict]:
    return [
        {"role": "system", "content": (
            "Translate the user's question into English so it can be turned into a database query. "
            "Keep names, places, and values exactly as written. If a table or column name from the schema is meant, use that exact name. "
            "Output only the English question, nothing else.")},
        {"role": "user", "content": f"Schema (for reference):\n{schema.strip()}\n\nQuestion:\n{question.strip()}"},
    ]


def clean_translation(text: str, fallback: str) -> str:
    t = (text or "").strip().strip('"').strip()
    t = re.sub(r"^\s*(english|translation)\s*:\s*", "", t, flags=re.IGNORECASE).split("\n")[0].strip()
    return t if 3 <= len(t) <= 400 else fallback


# ------------------------------------------------------------------ SQL self-correction
_HARD_REFUSAL_PREFIXES = ("Only SELECT", "The '")


_WRITE_VERBS = r"(?:delete|drop|truncate|erase|wipe|destroy|remove|update|insert|alter|modify|hapus|musnahkan|ubah|tambahkan)"
_WRITE_INTENT = re.compile(
    rf"^\s*(?:(?:please|pls|tolong|kindly|can you|could you|would you|will you|i want you to|i want to|i need to|let'?s)\s+)*{_WRITE_VERBS}\b"
    rf"|\b(?:delete|drop|truncate|erase|wipe|destroy|hapus)\s+(?:all|every|the|my|this|these|semua|seluruh|tabel|table|data|records?)\b",
    re.IGNORECASE,
)


def has_write_intent(question: str) -> bool:
    """True when the QUESTION itself asks to change data ("Delete all customers"). Reading questions such as
    "How many orders were deleted?" are not matched."""
    return bool(_WRITE_INTENT.search(question or ""))


def is_hard_refusal(reason: str) -> bool:
    """Safety refusals (writes, DDL) are final: the query is never retried or 'fixed' into something runnable."""
    return (reason or "").startswith(_HARD_REFUSAL_PREFIXES)


def feedback_messages(previous_sql: str, error: str) -> list[dict]:
    return [
        {"role": "assistant", "content": previous_sql.strip()},
        {"role": "user", "content": (
            f"That SQL did not work. Problem: {str(error)[:300]}\n"
            "Write a corrected single SQLite SELECT query for the same question. Use only tables and columns from the schema. Output only the SQL.")},
    ]


_EQ_LITERAL = re.compile(r"(=|!=|<>)\s*('(?:[^']|'')*'|\"(?:[^\"]|\"\")*\")(?!\s*COLLATE)", re.IGNORECASE)


def relax_string_equality(sql: str) -> tuple[str, bool]:
    """Make string comparisons case-insensitive ('Jakarta' vs 'jakarta'). Used only when a query returned no rows."""
    new, n = _EQ_LITERAL.subn(lambda m: f"{m.group(1)} {m.group(2)} COLLATE NOCASE", sql or "")
    return new, n > 0


# ------------------------------------------------------------------ LLM-written answers
STYLES = [
    "friendly and brief, like a helpful colleague",
    "straightforward like a data analyst: state the key number first",
    "casual but polite; you may add one helpful closing sentence",
    "clear and structured: the main finding first, then one supporting detail",
]

SYSTEM_ANSWER = (
    "You are a friendly data assistant. Answer the user's question based on the QUERY RESULT provided. "
    "Rules: (1) use only numbers and names that appear in the result; do not recompute, sum, or guess; "
    "(2) do not mention SQL, table names, or database jargon; (3) at most 3 sentences; "
    "(4) if the result has many rows, summarize it and give a few examples only; "
    "(5) do not claim anything the result does not show (for example, do not say names are unique or duplicated); "
    "(6) if the result has no rows, say that no matching data was found; (7) do not use markdown."
)

SYSTEM_FAILURE = (
    "You are a friendly data assistant. You could not answer the user's question from the available data. "
    "Explain this politely in at most 3 sentences, without technical details or error messages. "
    "You may suggest rephrasing the question using the available table and column names that are listed. "
    "Do not invent data or numbers. Do not use markdown."
)


def _cell(v) -> str:
    if v is None:
        return "empty"
    if isinstance(v, float):
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return str(v)


def build_explain_messages(question: str, sql: str, columns: Sequence[str], rows: Sequence[Sequence],
                           rng: Optional[random.Random] = None, language: str = "en", strict: bool = False) -> list[dict]:
    rng = rng or random
    shown = rows[:MAX_ROWS_IN_PROMPT]
    lines = [" | ".join(f"{c}={_cell(v)}" for c, v in zip(columns, r)) for r in shown] or ["(no rows)"]
    extra = f"\n(Only {len(shown)} of {len(rows)} rows are shown.)" if len(rows) > len(shown) else ""
    style = "plain and literal: restate the result values without any extra claim" if strict else rng.choice(STYLES)
    user = (
        f"User question: {question}\n"
        f"Query that was run (only to understand the column meanings): {sql}\n"
        f"Number of result rows: {len(rows)}\n"
        f"QUERY RESULT:\n" + "\n".join(lines) + extra + "\n\n"
        f"Write the answer in {LANG_NAME.get(language, 'English')}. Style: {style}."
    )
    return [{"role": "system", "content": SYSTEM_ANSWER}, {"role": "user", "content": user}]


def build_failure_messages(question: str, schema_tables: dict, language: str = "en") -> list[dict]:
    tables = "; ".join(f"{t}({', '.join(c for c, _ in cols)})" for t, cols in schema_tables.items())
    user = (f"User question: {question}\nAvailable data: {tables}\n"
            f"Write the reply in {LANG_NAME.get(language, 'English')}.")
    return [{"role": "system", "content": SYSTEM_FAILURE}, {"role": "user", "content": user}]


# ------------------------------------------------------------------ number verification
_NUM = re.compile(r"\d[\d.,]*\d|\d")


def _candidates(tok: str) -> set[float]:
    """All plausible readings of a number token (1,234.5 and 1.234,5 styles)."""
    out: set[float] = set()
    t = tok.strip(".,")
    for f in {t, t.replace(".", "").replace(",", ""), t.replace(".", "").replace(",", "."), t.replace(",", "")}:
        try:
            out.add(float(f))
        except ValueError:
            pass
    return out


def _numbers_in(text: str) -> set[float]:
    nums: set[float] = set()
    for m in _NUM.finditer(text or ""):
        nums |= _candidates(m.group(0))
    return nums


def allowed_numbers(question: str, columns: Sequence[str], rows: Sequence[Sequence]) -> set[float]:
    ok: set[float] = {float(len(rows))} | {float(i) for i in range(1, min(len(rows), MAX_ROWS_IN_PROMPT) + 1)}
    ok |= _numbers_in(question)
    for r in rows[:MAX_ROWS_IN_PROMPT]:
        for v in r:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                ok.update({float(v), float(round(v)), round(float(v), 1), round(float(v), 2)})
            elif v is not None:
                ok |= _numbers_in(str(v))
    return ok


def numbers_grounded(answer: str, question: str, columns: Sequence[str], rows: Sequence[Sequence]) -> bool:
    """True if every number in the answer can be traced to the query result (or the question)."""
    ok = allowed_numbers(question, columns, rows)
    for m in _NUM.finditer(answer or ""):
        if not any(abs(c - a) <= 0.0051 for c in _candidates(m.group(0)) for a in ok):
            return False
    return True


_BAD = re.compile(r"(?i)\b(select|from|where)\b.*\b(from|where|group|limit)\b|```")


def clean_answer(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^\s*(answer|jawaban)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = t.replace("**", "").replace("__", "").replace("#", "")
    t = re.sub(r"\n{2,}", "\n", t).strip()
    if len(t) > MAX_ANSWER_CHARS:
        t = t[:MAX_ANSWER_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "..."
    return t


def accept_answer(text: str, question: str, columns: Sequence[str], rows: Sequence[Sequence]) -> Optional[str]:
    """Return the cleaned answer if it passes the checks, else None."""
    t = clean_answer(text)
    if len(t) < 8 or _BAD.search(t):
        return None
    if not numbers_grounded(t, question, columns, rows):
        return None
    return t


def accept_plain(text: str) -> Optional[str]:
    """For replies that contain no data claims (failure explanations): only basic hygiene checks."""
    t = clean_answer(text)
    if len(t) < 8 or _BAD.search(t) or re.search(r"\d", t):
        return None
    return t
