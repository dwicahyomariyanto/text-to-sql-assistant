import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from guardrail import validate_sql  # noqa: E402
from sql_utils import safe_execute  # noqa: E402


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE city (name TEXT, population INTEGER)")
    c.executemany("INSERT INTO city VALUES (?, ?)", [("a", 1), ("b", 2), ("c", 3)])
    return c


@pytest.mark.parametrize("sql", [
    "SELECT * FROM city",
    "select name from city where population > 1;",
    "SELECT name FROM city WHERE name = 'drop table x'",       # kata terlarang hanya di dalam literal
    "WITH t AS (SELECT * FROM city) SELECT COUNT(*) FROM t",
    "SELECT name FROM city -- komentar biasa",
])
def test_select_diizinkan(sql, conn):
    assert validate_sql(sql, conn=conn).allowed


@pytest.mark.parametrize("sql", [
    "DROP TABLE city",
    "DELETE FROM city",
    "UPDATE city SET population = 0",
    "INSERT INTO city VALUES ('x', 1)",
    "ALTER TABLE city ADD COLUMN z INTEGER",
    "CREATE TABLE t (a INTEGER)",
    "PRAGMA table_info(city)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT * FROM city; DROP TABLE city",                       # multi-statement
    "/* trik */ DROP TABLE city",                                # komentar di depan
    "SELECT * FROM city WHERE name = 'a' OR 1=1; DELETE FROM city",
    "",
    "   ",
])
def test_dangerous_is_rejected(sql, conn):
    result = validate_sql(sql, conn=conn)
    assert not result.allowed
    assert result.reason


def test_bad_syntax_is_rejected(conn):
    assert not validate_sql("SELECT FROM WHERE", conn=conn).allowed
    assert not validate_sql("SELECT * FROM tabel_tidak_ada", conn=conn).allowed
    assert not validate_sql("SELECT (1 + 2", conn=conn).allowed


def test_unclosed_quote_is_rejected(conn):
    assert not validate_sql("SELECT * FROM city WHERE name = 'a", conn=conn).allowed


def test_batas_baris_pada_eksekusi(conn):
    cols, rows = safe_execute(conn, "SELECT * FROM city", max_rows=2)
    assert cols == ["name", "population"] and len(rows) == 2


def test_eksekusi_menolak_penulisan_walau_lolos_validator(conn):
    with pytest.raises(sqlite3.DatabaseError):
        safe_execute(conn, "DELETE FROM city")
    assert conn.execute("SELECT COUNT(*) FROM city").fetchone()[0] == 3


def test_max_rows_dikembalikan(conn):
    assert validate_sql("SELECT * FROM city", max_rows=7, conn=conn).max_rows == 7
