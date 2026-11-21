"""SQL utilities: normalization, complexity, schema parsing, synthetic SQLite DB, safe execution."""
from __future__ import annotations

import random
import re
import sqlite3
import time
from collections import Counter, OrderedDict
from typing import Iterable, Optional

# ---------------------------------------------------------------- literals & normalization
_LITERAL = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")


def mask_literals(sql: str) -> str:
    """Replace the content of string literals / quoted identifiers with '' so the SQL can be analyzed safely."""
    return _LITERAL.sub("''", sql)


def normalize_sql(sql: str) -> str:
    """Normalization for exact match: lowercase, whitespace, quote style, trailing semicolon."""
    s = (sql or "").strip().rstrip(";").strip()
    s = s.replace('"', "'")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*([,=<>!])\s*", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+\(", "(", s)
    return s.lower()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_sql(pred) == normalize_sql(gold)


# ---------------------------------------------------------------- kompleksitas
def sql_features(sql: str) -> set[str]:
    s = mask_literals(sql).lower()
    feats: set[str] = set()
    if re.search(r"\bjoin\b", s):
        feats.add("join")
    if re.search(r"\bgroup\s+by\b", s):
        feats.add("group_by")
    if re.search(r"\bhaving\b", s):
        feats.add("having")
    if re.search(r"\b(count|sum|avg|min|max)\s*\(", s):
        feats.add("aggregate")
    if re.search(r"\border\s+by\b", s):
        feats.add("order_by")
    n_set_ops = len(re.findall(r"\b(union|intersect|except)\b", s))
    if n_set_ops:
        feats.add("set_op")
    if len(re.findall(r"\bselect\b", s)) - 1 - n_set_ops > 0:
        feats.add("subquery")
    return feats


CATEGORIES = ["simple", "aggregation", "join", "nested"]


def complexity_category(sql: str) -> str:
    f = sql_features(sql)
    if f & {"subquery", "set_op"}:
        return "nested"
    if "join" in f:
        return "join"
    if f & {"aggregate", "group_by", "having"}:
        return "aggregation"
    return "simple"


# ---------------------------------------------------------------- schema parsing
def _unquote(name: str) -> str:
    return name.strip().strip("`\"'[]").strip()


def _split_top_level(body: str) -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return parts


def parse_schema(schema: str) -> "OrderedDict[str, list[tuple[str, str]]]":
    """Parse CREATE TABLE statements into {table: [(column, type), ...]}."""
    tables: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for stmt in (schema or "").split(";"):
        stmt = stmt.strip()
        m = re.match(r"create\s+table\s+(?:if\s+not\s+exists\s+)?(.+?)\s*\(", stmt, re.I | re.S)
        if not m or ")" not in stmt:
            continue
        name = _unquote(m.group(1))
        body = stmt[m.end(): stmt.rindex(")")]
        cols: list[tuple[str, str]] = []
        for part in _split_top_level(body):
            part = part.strip()
            if not part or re.match(r"(primary|foreign|unique|constraint|key|check)\b", part, re.I):
                continue
            if part[0] in "`\"[":
                close = {"`": "`", '"': '"', "[": "]"}[part[0]]
                end = part.find(close, 1)
                col, rest = part[1:end], part[end + 1:].strip()
            else:
                pieces = part.split(None, 1)
                col, rest = pieces[0], (pieces[1] if len(pieces) > 1 else "")
            ctype = rest.split()[0] if rest else "TEXT"
            cols.append((col, ctype))
        if cols:
            tables[name] = _dedupe_columns(cols)
    return tables


def _dedupe_columns(cols: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """SQLite column names are case-insensitive and duplicates are rejected; add a _dup suffix."""
    seen: set[str] = set()
    out = []
    for name, ctype in cols:
        new = name
        while new.lower() in seen:
            new += "_dup"
        seen.add(new.lower())
        out.append((new, ctype))
    return out


def _kind(ctype: str) -> str:
    t = re.sub(r"\(.*?\)", "", ctype).upper()
    if "INT" in t:
        return "int"
    if any(k in t for k in ("REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC")):
        return "real"
    return "text"


# ---------------------------------------------------------------- DB sintetis
def extract_literals(sql: str) -> list:
    """Extract string/number literals from a query so the synthetic data contains those values."""
    out: list = []
    for m in _LITERAL.finditer(sql or ""):
        inner = m.group(0)[1:-1].replace("''", "'").strip("%")
        if inner:
            out.append(inner)
    for m in re.finditer(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", mask_literals(sql or "")):
        txt = m.group(0)
        out.append(float(txt) if "." in txt else int(txt))
    return out


def build_sqlite_db(
    schema: str, rows_per_table: int = 25, seed: int = 0, literals: Iterable = ()
) -> sqlite3.Connection:
    """Build an in-memory SQLite DB from a schema, filled with deterministic synthetic data.

    Value pools are shared across tables (so JOINs return rows) and extended with
    literals from the query (so WHERE clauses are not automatically empty).
    """
    rng = random.Random(seed)
    lits = list(literals)
    str_pool = [f"item{i}" for i in range(6)] + [str(x) for x in lits]
    nums = [x for x in lits if isinstance(x, (int, float)) and not isinstance(x, bool)]
    near = lambda x: [x - 1, x, x + 1, x * 2]  # noqa: E731 - values around the literal so >, < comparisons return rows
    int_pool = list(dict.fromkeys(list(range(0, 12)) + [int(v) for x in nums if isinstance(x, int) for v in near(x)]))
    real_pool = [round(rng.uniform(0, 100), 1) for _ in range(10)] + [float(v) for x in nums for v in near(x)]
    conn = sqlite3.connect(":memory:")
    for tname, cols in parse_schema(schema).items():
        q = lambda s: s.replace('"', '""')  # noqa: E731 - escape double quotes in names
        ddl = ", ".join(f'"{q(c)}" {"INTEGER" if _kind(t) == "int" else "REAL" if _kind(t) == "real" else "TEXT"}' for c, t in cols)
        try:
            conn.execute(f'CREATE TABLE "{q(tname)}" ({ddl})')
        except sqlite3.Error:
            continue   # table cannot be created: queries using it will fail -> example is excluded from the count
        pools = []
        for _, t in cols:
            base = {"int": int_pool, "real": real_pool, "text": str_pool}[_kind(t)][:]
            rng.shuffle(base)
            pools.append(base)
        for r in range(rows_per_table):
            row = [p[r] if r < len(p) else rng.choice(p) for p in pools]
            conn.execute(f'INSERT INTO "{q(tname)}" VALUES ({",".join("?" * len(cols))})', row)
    conn.commit()
    return conn


# ---------------------------------------------------------------- safe (read-only) execution
_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _authorizer(action, *_args):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def safe_execute(conn: sqlite3.Connection, sql: str, max_rows: int = 1000, timeout_s: float = 5.0):
    """Run a SELECT query read-only with a time limit and a row limit."""
    deadline = time.monotonic() + timeout_s
    conn.set_authorizer(_authorizer)
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10000)
    try:
        cur = conn.execute(sql)
        cols = [d[0] for d in cur.description] if cur.description else []
        return cols, [tuple(r) for r in cur.fetchmany(max_rows)]
    finally:
        conn.set_progress_handler(None, 0)
        conn.set_authorizer(None)


def check_syntax(conn: sqlite3.Connection, sql: str) -> Optional[str]:
    """Return None if the SQL compiles on this schema, otherwise the error message."""
    conn.set_authorizer(_authorizer)
    try:
        conn.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
        return None
    except sqlite3.Error as exc:
        return str(exc)
    finally:
        conn.set_authorizer(None)


# ---------------------------------------------------------------- execution match
def _norm_value(v):
    return round(v, 4) if isinstance(v, float) else v


def _same_rows(a: list[tuple], b: list[tuple], ordered: bool) -> bool:
    a = [tuple(_norm_value(v) for v in r) for r in a]
    b = [tuple(_norm_value(v) for v in r) for r in b]
    return a == b if ordered else Counter(map(repr, a)) == Counter(map(repr, b))


def execution_match(schema: str, pred_sql: str, gold_sql: str, rows_per_table: int = 25, seed: int = 0) -> dict:
    """Compare the execution result of the predicted SQL vs the reference on the synthetic DB."""
    literals = extract_literals(gold_sql) + extract_literals(pred_sql)
    try:
        conn = build_sqlite_db(schema, rows_per_table, seed, literals)
    except Exception as exc:  # odd schema that cannot be built -> example excluded (gold_ok=False)
        return {"match": False, "gold_ok": False, "gold_nonempty": False, "error": f"schema: {exc}"}
    try:
        try:
            _, gold_rows = safe_execute(conn, gold_sql)
        except Exception as exc:  # the reference cannot run on the synthetic schema
            return {"match": False, "gold_ok": False, "gold_nonempty": False, "error": f"gold: {exc}"}
        try:
            _, pred_rows = safe_execute(conn, pred_sql)
        except Exception as exc:
            return {"match": False, "gold_ok": True, "gold_nonempty": bool(gold_rows), "error": str(exc)}
        ordered = bool(re.search(r"\border\s+by\b", mask_literals(gold_sql), re.I))
        return {
            "match": _same_rows(gold_rows, pred_rows, ordered),
            "gold_ok": True,
            "gold_nonempty": bool(gold_rows),
            "error": None,
        }
    finally:
        conn.close()
