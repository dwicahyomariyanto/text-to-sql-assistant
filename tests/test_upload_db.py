import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "Interface_submission"):
    sys.path.insert(0, str(ROOT / sub))

from app import UPLOAD, MockGenerator, answer_question, describe_result  # noqa: E402
from upload_db import UploadError, import_files, sanitize_name  # noqa: E402


def test_sanitize():
    assert sanitize_name("Harga Jual (Rp)") == "Harga_Jual_Rp"
    assert sanitize_name("2024 data") == "t_2024_data"
    assert sanitize_name("!!!") == "table"


def test_csv_import_and_query(tmp_path):
    f = tmp_path / "Penjualan Toko.csv"
    f.write_text("Nama Barang,Harga,Jumlah\nPensil,2500,10\nBuku,12000.5,4\nPena,4000,7\n", encoding="utf-8")
    res = import_files([str(f)], workdir=str(tmp_path / "out"))
    assert "CREATE TABLE Penjualan_Toko" in res.schema and "Nama_Barang VARCHAR" in res.schema
    assert "Harga REAL" in res.schema and "Jumlah INTEGER" in res.schema
    t = answer_question(MockGenerator(), UPLOAD, res.schema, "How many rows?", db_path=res.db_path)
    assert t.guard.allowed and t.rows == [(3,)]
    assert "3" in describe_result(t)


def test_upload_is_readonly(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x,y\n1,2\n", encoding="utf-8")
    res = import_files([str(f)], workdir=str(tmp_path / "o"))
    t = answer_question(MockGenerator(), UPLOAD, res.schema, "please delete everything", db_path=res.db_path)
    assert not t.guard.allowed


def test_multiple_csv_and_duplicate_columns(tmp_path):
    (tmp_path / "a.csv").write_text("id,id\n1,2\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("id;nama\n1;x\n", encoding="utf-8")   # pemisah titik koma terdeteksi otomatis
    res = import_files([str(tmp_path / "a.csv"), str(tmp_path / "b.csv")], workdir=str(tmp_path / "o"))
    assert set(res.tables) == {"a", "b"} and "id_1" in res.schema and "nama" in res.schema


def test_sqlite_upload(tmp_path):
    import sqlite3
    p = tmp_path / "x.db"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE kota (nama TEXT, penduduk INT)")
    c.execute("INSERT INTO kota VALUES ('A', 5)")
    c.commit(); c.close()
    res = import_files([str(p)], workdir=str(tmp_path / "o"))
    assert res.tables == {"kota": 1} and "penduduk INTEGER" in res.schema


@pytest.mark.parametrize("name,content,msg", [
    ("a.xlsx", b"x", "Unsupported format"),
    ("a.db", b"bukan sqlite", "not a valid SQLite file"),
])
def test_rejects(tmp_path, name, content, msg):
    f = tmp_path / name
    f.write_bytes(content)
    with pytest.raises(UploadError, match=msg):
        import_files([str(f)], workdir=str(tmp_path / "o"))


def test_limits(tmp_path):
    f = tmp_path / "w.csv"
    f.write_text(",".join(f"c{i}" for i in range(30)) + "\n" + ",".join("1" for _ in range(30)) + "\n", encoding="utf-8")
    with pytest.raises(UploadError, match="columns"):
        import_files([str(f)], workdir=str(tmp_path / "o"))
    with pytest.raises(UploadError):
        import_files([])


def test_upload_without_file():
    t = answer_question(MockGenerator(), UPLOAD, "", "How many?")
    assert not t.guard.allowed
