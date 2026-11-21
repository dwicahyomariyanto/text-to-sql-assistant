import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "Interface_submission"):
    sys.path.insert(0, str(ROOT / sub))

from answer_utils import accept_answer, build_explain_messages, clean_answer, numbers_grounded  # noqa: E402
from app import GuardrailResult, MockGenerator, Turn, compose_answer  # noqa: E402

COLS, ROWS = ["doctor_id", "total"], [(1, 9), (2, 7), (3, 1234)]
Q = "How many appointments does each doctor have?"


def test_grounded_numbers():
    assert numbers_grounded("Doctor 1 has 9 appointments, doctor 3 has 1,234.", Q, COLS, ROWS)
    assert numbers_grounded("There are 3 doctors; the total is 1234 for doctor 3.", Q, COLS, ROWS)
    assert not numbers_grounded("Doctor 1 has 10 appointments.", Q, COLS, ROWS)       # angka karangan
    assert not numbers_grounded("The total is 17 appointments.", Q, COLS, ROWS)              # recomputed figure is rejected


def test_decimals_and_question_numbers():
    assert numbers_grounded("The average GPA is 3.28.", "q", ["a"], [(3.28,)])
    assert numbers_grounded("4 people have a GPA above 3.5.", "GPA above 3.5", ["n"], [(4,)])


def test_accept_rejects_sql_and_short():
    assert accept_answer("SELECT * FROM x WHERE a = 1 LIMIT 2", Q, COLS, ROWS) is None
    assert accept_answer("ok", Q, COLS, ROWS) is None
    assert accept_answer("Answer: **Doctor 2 has 7 appointments.**", Q, COLS, ROWS) == "Doctor 2 has 7 appointments."


def test_clean_truncates():
    assert len(clean_answer("kata " * 400)) <= 705


def test_prompt_contains_data_and_varies():
    m1 = build_explain_messages(Q, "SELECT 1", COLS, ROWS, random.Random(1))
    assert "doctor_id=1 | total=9" in m1[1]["content"] and "Number of result rows: 3" in m1[1]["content"]
    styles = {build_explain_messages(Q, "s", COLS, ROWS, random.Random(i))[1]["content"].split("Style:")[1] for i in range(12)}
    assert len(styles) > 1


class _Gen(MockGenerator):
    def __init__(self, text):
        self.text = text

    def explain(self, *a, **k):
        if isinstance(self.text, Exception):
            raise self.text
        return self.text


def _turn():
    return Turn(Q, "SELECT 1", GuardrailResult(True, "ok"), COLS, ROWS, "")


def test_compose_uses_llm_when_grounded():
    ans, src = compose_answer(_Gen("Doctor 1 handles 9 appointments and doctor 2 handles 7."), _turn())
    assert src == "llm" and "9" in ans


def test_compose_falls_back():
    for g in (_Gen("Doctor 1 handles 99 appointments."), _Gen(RuntimeError("x")), _Gen(None), MockGenerator()):
        ans, src = compose_answer(g, _turn())
        assert src == "plain" and "Found 3 rows" in ans
    assert compose_answer(_Gen("Doctor 1 handles 9 appointments."), _turn(), use_llm=False)[1] == "plain"
