"""Inference guardrail: SELECT only, reject DDL/DML, validate syntax, limit the number of rows."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

from sql_utils import check_syntax

FORBIDDEN_KEYWORDS = (
    "drop", "delete", "update", "insert", "alter", "create", "truncate", "attach", "detach",
    "pragma", "vacuum", "reindex", "grant", "revoke", "exec", "execute", "merge", "load_extension",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
DEFAULT_MAX_ROWS = 100


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str
    sanitized_sql: str = ""
    max_rows: int = DEFAULT_MAX_ROWS


def _scan(sql: str) -> tuple[str, str, bool]:
    """Scan SQL: strip comments, mask literal contents. Returns (clean, masked, has_unterminated_quote)."""
    clean, masked, i, n, unterminated = [], [], 0, len(sql), False
    while i < n:
        c = sql[i]
        if c in "'\"`":
            j = i + 1
            while j < n:
                if sql[j] == c:
                    if j + 1 < n and sql[j + 1] == c:
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                unterminated, j = True, n - 1
            clean.append(sql[i:j + 1])
            masked.append(c + c)
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j
            clean.append(" ")
            masked.append(" ")
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            i = n if j == -1 else j + 2
            clean.append(" ")
            masked.append(" ")
        else:
            clean.append(c)
            masked.append(c)
            i += 1
    return "".join(clean), "".join(masked), unterminated


def validate_sql(
    sql: str, max_rows: int = DEFAULT_MAX_ROWS, conn: Optional[sqlite3.Connection] = None
) -> GuardrailResult:
    """Check whether the SQL is safe to run. If conn is given, the syntax is validated against that schema."""
    if not sql or not sql.strip():
        return GuardrailResult(False, "Empty query.", "", max_rows)

    clean, masked, unterminated = _scan(sql)
    if unterminated:
        return GuardrailResult(False, "Unclosed quotation mark.", "", max_rows)

    masked = masked.strip().rstrip(";").strip()
    clean = clean.strip().rstrip(";").strip()
    if not masked:
        return GuardrailResult(False, "Empty query.", "", max_rows)
    if ";" in masked:
        return GuardrailResult(False, "Only one SQL statement is allowed.", "", max_rows)

    first_word = re.match(r"\s*\(*\s*(\w+)", masked)
    if not first_word or first_word.group(1).lower() not in {"select", "with"}:
        return GuardrailResult(False, "Only SELECT queries are allowed.", "", max_rows)

    bad = _FORBIDDEN_RE.search(masked)
    if bad:
        return GuardrailResult(False, f"The '{bad.group(1).upper()}' command is not allowed (read-only).", "", max_rows)

    if masked.count("(") != masked.count(")"):
        return GuardrailResult(False, "Invalid syntax: unbalanced parentheses.", "", max_rows)

    if conn is not None:
        error = check_syntax(conn, clean)
        if error:
            return GuardrailResult(False, f"Invalid syntax or schema: {error}", "", max_rows)

    return GuardrailResult(True, "Safe: read-only query (SELECT).", clean, max_rows)
