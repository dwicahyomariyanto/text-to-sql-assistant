"""Import the user's own data (CSV or SQLite file) into a temporary SQLite database for the UI.

Result: the path of a .db file (opened READ-ONLY when querying) and a compact CREATE TABLE schema for the model prompt.
Size, table count, and column count are limited because the model has a small context window (3B, 1024 tokens).
"""
from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

MAX_FILE_MB = 25
MAX_TABLES = 6
MAX_COLS = 25
MAX_ROWS = 200_000
MAX_SCHEMA_CHARS = 2000
SQLITE_EXT = {".db", ".sqlite", ".sqlite3"}
_SQLITE_MAGIC = b"SQLite format 3\x00"


class UploadError(ValueError):
    """A message that is safe to show directly to the user."""


@dataclass
class UploadResult:
    db_path: str
    schema: str
    tables: dict = field(default_factory=dict)   # table name -> row count
    message: str = ""


def sanitize_name(name: str, fallback: str = "table") -> str:
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(name).strip()).strip("_")
    if not s:
        s = fallback
    if s[0].isdigit():
        s = f"t_{s}"
    return s


def _dedupe(names: Iterable[str]) -> list[str]:
    seen, out = {}, []
    for n in names:
        k = n.lower()
        seen[k] = seen.get(k, 0) + 1
        out.append(n if seen[k] == 1 else f"{n}_{seen[k]}")
    return out


def _type_label(decl: str) -> str:
    d = (decl or "").upper()
    if "INT" in d:
        return "INTEGER"
    if any(k in d for k in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "REAL"
    return "VARCHAR"


def _check_size(path: Path) -> None:
    if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
        raise UploadError(f"File '{path.name}' is larger than {MAX_FILE_MB} MB. Please make the data smaller.")


def _csv_to_sqlite(csv_path: Path, conn: sqlite3.Connection, used: set) -> None:
    import pandas as pd

    df = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python", encoding=enc, nrows=MAX_ROWS)
            break
        except UnicodeDecodeError:
            continue
        except Exception as exc:  # noqa: BLE001
            raise UploadError(f"Could not read '{csv_path.name}' as CSV: {exc}") from exc
    if df is None or df.shape[1] == 0:
        raise UploadError(f"'{csv_path.name}' is empty or not a valid CSV.")
    df.columns = _dedupe(sanitize_name(c, f"column_{i + 1}") for i, c in enumerate(df.columns))
    name = sanitize_name(csv_path.stem)
    base, k = name, 2
    while name.lower() in used:
        name, k = f"{base}_{k}", k + 1
    used.add(name.lower())
    df.to_sql(name, conn, index=False)


def _describe(conn: sqlite3.Connection) -> tuple[str, dict]:
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
    if not tables:
        raise UploadError("No tables found in the uploaded data.")
    if len(tables) > MAX_TABLES:
        raise UploadError(f"Found {len(tables)} tables; the maximum is {MAX_TABLES} so the schema fits the model context. Upload fewer tables.")
    parts, counts = [], {}
    for t in tables:
        info = conn.execute(f'PRAGMA table_info("{t}")').fetchall()
        if len(info) > MAX_COLS:
            raise UploadError(f"Table '{t}' has {len(info)} columns; the maximum is {MAX_COLS}. Remove columns you do not need.")
        cols = ", ".join(f"{c[1]} {_type_label(c[2])}" for c in info)
        parts.append(f"CREATE TABLE {t} ({cols})")
        counts[t] = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    schema = "; ".join(parts)
    if len(schema) > MAX_SCHEMA_CHARS:
        raise UploadError("The schema is too long for the model. Reduce the number of tables or columns.")
    return schema, counts


def import_files(paths: Iterable[str], workdir: Optional[str] = None) -> UploadResult:
    """Turn one SQLite file OR several CSV files into a single ready-to-query SQLite database."""
    files = [Path(p) for p in paths if p]
    if not files:
        raise UploadError("No file selected.")
    for f in files:
        if not f.exists():
            raise UploadError(f"File '{f.name}' was not found.")
        _check_size(f)
    out_dir = Path(workdir or tempfile.mkdtemp(prefix="t2s_upload_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "data_pengguna.db"
    if db_path.exists():
        db_path.unlink()

    sqlite_files = [f for f in files if f.suffix.lower() in SQLITE_EXT]
    csv_files = [f for f in files if f.suffix.lower() in {".csv", ".tsv", ".txt"}]
    other = [f.name for f in files if f not in sqlite_files and f not in csv_files]
    if other:
        raise UploadError("Unsupported format: " + ", ".join(other) + ". Use .csv or .db/.sqlite.")
    if sqlite_files and csv_files:
        raise UploadError("Upload one SQLite file OR several CSV files, not a mix.")
    if len(sqlite_files) > 1:
        raise UploadError("Upload only one SQLite file.")

    if sqlite_files:
        with open(sqlite_files[0], "rb") as fh:
            if fh.read(16) != _SQLITE_MAGIC:
                raise UploadError(f"'{sqlite_files[0].name}' is not a valid SQLite file.")
        shutil.copyfile(sqlite_files[0], db_path)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(db_path)
        used: set = set()
        try:
            for f in csv_files:
                _csv_to_sqlite(f, conn, used)
            conn.commit()
        except Exception:
            conn.close()
            raise
    try:
        schema, counts = _describe(conn)
    finally:
        conn.close()
    first = next(iter(counts))
    detail = "; ".join(f"`{t}` ({n:,} rows)" for t, n in counts.items())
    msg = (f"Loaded {len(counts)} table(s): {detail}. "
           f"Try asking, for example: *How many rows are in {first}?*")
    return UploadResult(str(db_path), schema, counts, msg)


def open_readonly(db_path: str) -> sqlite3.Connection:
    """Open the uploaded database in read-only mode."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
