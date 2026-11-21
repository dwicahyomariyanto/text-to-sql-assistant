import os

import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from demo_db import create_demo_connection, get_schema, list_databases  # noqa: E402
from eval_utils import evaluate_records, write_results_table  # noqa: E402
from prompt_utils import build_messages, extract_sql  # noqa: E402
from sql_utils import (  # noqa: E402
    build_sqlite_db, complexity_category, exact_match, execution_match, extract_literals,
    normalize_sql, parse_schema, safe_execute,
)

SCHEMA = "CREATE TABLE head (name VARCHAR, age INTEGER, dept_id INTEGER); CREATE TABLE dept (dept_id INTEGER, title VARCHAR)"


def test_normalization_and_exact_match():
    assert normalize_sql('SELECT  COUNT( * )  FROM head ;') == "select count(*) from head"
    assert exact_match('SELECT name FROM head WHERE age > 5', 'select name from head where age>5;')
    assert not exact_match("SELECT name FROM head", "SELECT age FROM head")


def test_kategori_kompleksitas():
    assert complexity_category("SELECT name FROM head") == "simple"
    assert complexity_category("SELECT COUNT(*) FROM head") == "aggregation"
    assert complexity_category("SELECT dept_id, COUNT(*) FROM head GROUP BY dept_id") == "aggregation"
    assert complexity_category("SELECT a.name FROM head a JOIN dept b ON a.dept_id = b.dept_id") == "join"
    assert complexity_category("SELECT name FROM head WHERE age > (SELECT AVG(age) FROM head)") == "nested"
    assert complexity_category("SELECT name FROM head UNION SELECT title FROM dept") == "nested"
    # keywords inside literals are not counted
    assert complexity_category("SELECT name FROM head WHERE name = 'group by join'") == "simple"


def test_parse_schema():
    t = parse_schema(SCHEMA)
    assert list(t) == ["head", "dept"]
    assert t["head"] == [("name", "VARCHAR"), ("age", "INTEGER"), ("dept_id", "INTEGER")]
    quoted = parse_schema('CREATE TABLE "my table" ("Wins/Losses" VARCHAR(10), id INTEGER, PRIMARY KEY (id))')
    assert quoted["my table"] == [("Wins/Losses", "VARCHAR(10)"), ("id", "INTEGER")]


def test_synthetic_db_is_deterministic_and_contains_literals():
    a = build_sqlite_db(SCHEMA, seed=1, literals=["Bob", 99])
    b = build_sqlite_db(SCHEMA, seed=1, literals=["Bob", 99])
    q = "SELECT name, age, dept_id FROM head ORDER BY name, age"
    assert safe_execute(a, q)[1] == safe_execute(b, q)[1]
    assert safe_execute(a, "SELECT COUNT(*) FROM head WHERE name = 'Bob'")[1][0][0] >= 1
    assert safe_execute(a, "SELECT COUNT(*) FROM head WHERE age = 99")[1][0][0] >= 1


def test_ekstrak_literal():
    lits = extract_literals("SELECT * FROM head WHERE name = 'Bob' AND age > 30 AND x LIKE '%ab%' AND y = 1.5")
    assert "Bob" in lits and 30 in lits and "ab" in lits and 1.5 in lits


def test_execution_match():
    gold = "SELECT COUNT(*) FROM head WHERE age > 5"
    assert execution_match(SCHEMA, "select count(age) from head where age > 5", gold)["match"]
    assert not execution_match(SCHEMA, "SELECT COUNT(*) FROM head WHERE age > 10", gold)["match"]
    rusak = execution_match(SCHEMA, "SELECT nama_tidak_ada FROM head", gold)
    assert not rusak["match"] and rusak["error"]
    assert execution_match(SCHEMA, gold, gold)["gold_nonempty"]


def test_execution_match_urutan_hanya_jika_order_by():
    g = "SELECT name FROM head ORDER BY age ASC"
    assert not execution_match(SCHEMA, "SELECT name FROM head ORDER BY age DESC", g)["match"]
    # gold without ORDER BY -> prediction row order does not matter
    assert execution_match(SCHEMA, "SELECT name FROM head ORDER BY name", "SELECT name FROM head")["match"]


def test_extract_sql_and_messages():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT a FROM t; -- penjelasan tambahan") == "SELECT a FROM t"
    assert extract_sql("SELECT ';' FROM t") == "SELECT ';' FROM t"
    msgs = build_messages(SCHEMA, "How many heads?", answer="SELECT COUNT(*) FROM head")
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert [m["role"] for m in build_messages(SCHEMA, "q", history=[("q0", "SELECT 1")])] == ["system", "user", "assistant", "user"]


def test_evaluate_records_and_table():
    recs = [
        {"context": SCHEMA, "question": "q1", "answer": "SELECT COUNT(*) FROM head", "pred": "SELECT COUNT(*) FROM head"},
        {"context": SCHEMA, "question": "q2", "answer": "SELECT name FROM head WHERE age > 5", "pred": "SELECT name FROM head"},
    ]
    m = evaluate_records(recs)
    assert m["n"] == 2 and m["exact_match"] == 0.5
    assert set(m["by_category"]) == {"simple", "aggregation"}
    assert "Exact match" in write_results_table({"baseline": m, "fine-tuned": m}, os.devnull)


def test_database_demo():
    for name in list_databases():
        conn = create_demo_connection(name)
        for table in parse_schema(get_schema(name)):
            assert safe_execute(conn, f'SELECT COUNT(*) FROM "{table}"')[1][0][0] > 0


def test_duplicate_columns_and_odd_names_do_not_crash():
    # real dataset case: identically named columns (also differing in letter case)
    dup = "CREATE TABLE cars (Model VARCHAR, model VARCHAR, Year INTEGER, Model VARCHAR)"
    assert [c for c, _ in parse_schema(dup)["cars"]] == ["Model", "model_dup", "Year", "Model_dup_dup"]
    r = execution_match(dup, "SELECT Model FROM cars", "SELECT Model FROM cars")
    assert r["gold_ok"] and r["match"]
    weird = 'CREATE TABLE "t""x" ("a""b" INTEGER)'
    assert build_sqlite_db(weird) is not None
    # a broken schema must not raise an error during evaluation
    assert execution_match("CREATE TABLE (((", "SELECT 1", "SELECT 1")["gold_ok"] in (True, False)


def test_normalization_mismatch_and_comparison_yields_rows():
    assert exact_match("SELECT a FROM t WHERE b != 1", "select a from t where b!=1")
    schema = "CREATE TABLE city (Status VARCHAR, Population INTEGER)"
    for op in (">", "<"):
        gold = f"SELECT Status FROM city WHERE Population {op} 1000"
        assert execution_match(schema, gold, gold)["gold_nonempty"], op
    # ambang berbeda harus terdeteksi, bukan lolos karena sama-sama kosong
    assert not execution_match(schema, "SELECT Status FROM city WHERE Population > 5", "SELECT Status FROM city WHERE Population > 1000")["match"]


def test_peringatan_panjang_generasi_disaring_tapi_peringatan_lain_tetap():
    import logging

    from eval_utils import silence_generation_length_warning

    silence_generation_length_warning()
    silence_generation_length_warning()          # idempotent: filters do not pile up
    logger = logging.getLogger("transformers.generation.utils")
    assert sum(type(f).__name__ == "_Filter" for f in logger.filters) == 1
    seen = []

    class Catch(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = Catch()
    logger.addHandler(handler)
    try:
        logger.warning("Both `max_new_tokens` (=128) and `max_length`(=32768) seem to have been set. `max_new_tokens` will take precedence.")
        logger.warning("another important warning")
    finally:
        logger.removeHandler(handler)
    assert seen == ["another important warning"]


def test_get_hf_token_strips_whitespace_and_newlines(monkeypatch):
    from eval_utils import get_hf_token

    monkeypatch.setenv("HF_TOKEN", "  hf_abc123\n")
    assert get_hf_token() == "hf_abc123"
    monkeypatch.setenv("HF_TOKEN", "   ")
    monkeypatch.delenv("HF_TOKEN")
    assert get_hf_token() is None


# ---------------------------------------------------------------- helper Colab / token / adapter
import sys
import types


def _fake_hf(monkeypatch, files=None, whoami="pengguna", raise_whoami=False, raise_create=False):
    class FakeApi:
        created = []

        def __init__(self, token=None):
            self.token = token

        def whoami(self, token=None):
            if raise_whoami:
                raise RuntimeError("401")
            return {"name": whoami}

        def list_repo_files(self, repo):
            if files is None:
                raise RuntimeError("404")
            return files

        def create_repo(self, repo_id, exist_ok=False):
            if raise_create:
                raise RuntimeError("403")
            FakeApi.created.append(repo_id)

    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return FakeApi


def test_work_dir_env_and_scratch(monkeypatch, tmp_path):
    from eval_utils import get_scratch_dir, get_work_dir

    monkeypatch.setenv("T2S_WORK_DIR", str(tmp_path / "drive"))
    assert get_work_dir() == tmp_path / "drive" / "outputs" and get_work_dir().exists()
    assert get_scratch_dir() == get_work_dir()               # di luar Colab, scratch = work dir
    monkeypatch.setenv("T2S_SCRATCH_DIR", str(tmp_path / "scr"))
    assert get_scratch_dir() == tmp_path / "scr" and (tmp_path / "scr").exists()


def test_ensure_token_manual_env_and_prompt(monkeypatch):
    from eval_utils import ensure_hf_token

    _fake_hf(monkeypatch)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert ensure_hf_token(" hf_manual \n") == "hf_manual" and os.environ["HF_TOKEN"] == "hf_manual"
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "  hf_diketik ")
    assert ensure_hf_token() == "hf_diketik" and os.environ["HF_TOKEN"] == "hf_diketik"
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    monkeypatch.setattr("getpass.getpass", lambda prompt="": pytest.fail("must not ask for input"))
    assert ensure_hf_token() == "hf_env"


def test_ensure_token_rejected_and_empty(monkeypatch):
    from eval_utils import ensure_hf_token

    _fake_hf(monkeypatch, raise_whoami=True)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(ValueError, match="rejected"):
        ensure_hf_token("hf_wrong")
    assert "HF_TOKEN" not in os.environ                       # a rejected token is not stored
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "")
    with pytest.raises(ValueError, match="empty"):
        ensure_hf_token()
    assert ensure_hf_token(required=False) is None


def test_prepare_hub_repo(monkeypatch):
    from eval_utils import prepare_hub_repo

    api = _fake_hf(monkeypatch, whoami="dwi")
    cfg = {"hf_username": "", "hub_repo_name": "model-x"}
    assert prepare_hub_repo(cfg, "t") == "dwi/model-x" and cfg["hf_username"] == "dwi" and api.created == ["dwi/model-x"]
    _fake_hf(monkeypatch, raise_create=True)
    with pytest.raises(RuntimeError, match="Write"):
        prepare_hub_repo({"hf_username": "", "hub_repo_name": "m"}, "t")


def test_resolve_adapter(monkeypatch, tmp_path):
    from eval_utils import resolve_adapter

    monkeypatch.setenv("T2S_WORK_DIR", str(tmp_path))
    _fake_hf(monkeypatch, files=["adapter_config.json", "adapter_model.safetensors"], whoami="dwi")
    assert resolve_adapter({"adapter": "", "hub_repo_name": "m"}, "t") == "dwi/m"          # from HF
    assert resolve_adapter({"adapter": "orang/lain"}, "t") == "orang/lain"                  # explicit
    _fake_hf(monkeypatch, files=["config.json"], whoami="dwi")                              # repo exists but has no adapter yet
    with pytest.raises(FileNotFoundError, match="no final adapter"):
        resolve_adapter({"adapter": "", "hub_repo_name": "m"}, "t")
    local = tmp_path / "outputs" / "adapter"
    local.mkdir(parents=True)
    (local / "adapter_config.json").write_text("{}")
    assert resolve_adapter({"adapter": "", "hub_repo_name": "m"}, "t") == str(local)        # cadangan lokal
    _fake_hf(monkeypatch, files=None)
    assert resolve_adapter({"adapter": str(local)}, None) == str(local)


def test_export_results_membuat_zip(monkeypatch, tmp_path):
    from eval_utils import export_results

    monkeypatch.setenv("T2S_WORK_DIR", str(tmp_path))
    d = tmp_path / "res"
    d.mkdir()
    (d / "a.json").write_text("{}")
    z = export_results(d, name="uji")
    assert z.exists() and z.suffix == ".zip"
