import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "Interface_submission"):
    sys.path.insert(0, str(ROOT / sub))

from answer_utils import accept_plain, detect_language, feedback_messages, is_hard_refusal, relax_string_equality  # noqa: E402
from app import (CUSTOM, FRIENDLY_ERROR, REFUSAL_TEXT, MockGenerator, Turn, answer_question, compose_answer,  # noqa: E402
                 describe_result)


class Scripted(MockGenerator):
    """Returns queued SQL per call and records what it was given."""

    def __init__(self, outputs, translation=None):
        self.outputs, self.calls, self.translation = list(outputs), [], translation

    def generate(self, schema, question, history=None, feedback=None):
        self.calls.append((question, feedback))
        return self.outputs.pop(0)

    def translate(self, question, schema):
        return self.translation


def test_language_detection():
    assert detect_language("How many appointments does each doctor have?") == "en"
    assert detect_language("Berapa jumlah dokter dengan spesialisasi Cardiology?") == "id"
    assert detect_language("average price per category") == "en"


def test_indonesian_question_is_translated_before_sql():
    g = Scripted(["SELECT COUNT(*) FROM customers"], translation="How many customers are there?")
    t = answer_question(g, "Online Store", "", "Berapa jumlah pelanggan?")
    assert g.calls[0][0] == "How many customers are there?" and t.language == "id" and t.rows[0][0] == 20


def test_self_correction_after_execution_error():
    g = Scripted(["SELECT COUNT(*) FROM no_such_table", "SELECT COUNT(*) FROM customers"])
    t = answer_question(g, "Online Store", "", "How many customers are there?")
    assert t.rows == [(20,)] and t.attempts == 2 and "corrected automatically" in t.note
    assert g.calls[1][1][0].startswith("SELECT COUNT(*) FROM no_such_table") and "no_such_table" in g.calls[1][1][1]


def test_self_correction_after_invalid_syntax():
    g = Scripted(["SELECT FROM customers WHERE", "SELECT COUNT(*) FROM customers"])
    t = answer_question(g, "Online Store", "", "How many customers are there?")
    assert t.rows == [(20,)] and t.attempts == 2


def test_gives_up_gracefully_after_max_attempts():
    g = Scripted(["SELECT x FROM nope"] * 3)
    t = answer_question(g, "Online Store", "", "What is the meaning of life?")
    assert t.failed and not t.rows and len(g.calls) == 3
    assert "Sorry" in describe_result(t) and "Traceback" not in describe_result(t)


def test_safety_refusal_is_final_and_not_retried():
    g = Scripted(["DELETE FROM customers", "SELECT 1"])
    t = answer_question(g, "Online Store", "", "Clean out the customers table for me")   # no write verb -> reaches the SQL guardrail
    assert not t.guard.allowed and len(g.calls) == 1 and is_hard_refusal(t.guard.reason)
    assert compose_answer(g, t) == (REFUSAL_TEXT, "fixed")


def test_case_insensitive_retry_when_empty():
    g = Scripted(["SELECT COUNT(*) FROM customers WHERE city = 'JAKARTA'"])
    t = answer_question(g, "Online Store", "", "How many customers are from Jakarta?")
    assert t.rows[0][0] > 0 and "case-insensitively" in t.note and "COLLATE NOCASE" in t.sql


def test_relax_does_not_touch_numbers_or_existing_collate():
    assert relax_string_equality("SELECT * FROM t WHERE n = 5") == ("SELECT * FROM t WHERE n = 5", False)
    assert relax_string_equality("SELECT * FROM t WHERE c = 'a' COLLATE NOCASE")[1] is False


def test_generation_crash_is_friendly():
    class Broken(MockGenerator):
        def generate(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    t = answer_question(Broken(), "Campus", "", "How many students?")
    assert not t.guard.allowed and "CUDA out of memory" in t.note
    assert compose_answer(Broken(), t)[0] == FRIENDLY_ERROR


class Talker(MockGenerator):
    def __init__(self, answers, failure=None):
        self.answers, self.failure, self.n = list(answers), failure, 0

    def explain(self, question, sql, columns, rows, language="en", strict=False):
        self.n += 1
        return self.answers.pop(0)

    def explain_failure(self, question, schema_tables, language="en"):
        return self.failure


def _ok_turn():
    g = MockGenerator()
    return answer_question(g, "Online Store", "", "How many customers are there?")


def test_llm_answer_retries_until_numbers_are_grounded():
    g = Talker(["There are 99 customers in total.", "We currently have 20 customers."])
    ans, src = compose_answer(g, _ok_turn())
    assert (ans, src) == ("We currently have 20 customers.", "llm") and g.n == 2


def test_llm_answer_falls_back_to_plain_after_all_tries_fail():
    g = Talker(["99 customers."] * 3)
    ans, src = compose_answer(g, _ok_turn())
    assert src == "plain" and "20" in ans and g.n == 3


def test_failure_explanation_written_by_llm_and_checked():
    g = Scripted(["SELECT x FROM nope"] * 3)
    t = answer_question(g, "Online Store", "", "Who won the football match yesterday?")
    nice = Talker([], failure="I could not find that in this store data. Could you ask about customers, products or orders?")
    assert compose_answer(nice, t) == (nice.failure, "llm")
    bad = Talker([], failure="Error 500 in table x123")          # contains digits -> rejected
    assert compose_answer(bad, t)[1] == "plain"
    assert accept_plain("ok") is None


def test_empty_result_is_explained_by_llm():
    g = Scripted(["SELECT name FROM customers WHERE city = 'Atlantis'"])
    t = answer_question(g, "Online Store", "", "Which customers live in Atlantis?")
    assert t.rows == [] and not t.failed
    llm = Talker(["No customers from Atlantis were found."])
    assert compose_answer(llm, t) == ("No customers from Atlantis were found.", "llm")


def test_custom_db_and_input_hints_stay_fixed():
    t = answer_question(MockGenerator(), CUSTOM, "CREATE TABLE a (x INTEGER)", "How many?")
    assert compose_answer(Talker([]), t)[1] in ("fixed", "plain")
    assert compose_answer(Talker([]), answer_question(MockGenerator(), "Campus", "", "  "))[1] == "fixed"


def test_feedback_message_shape():
    m = feedback_messages("SELECT 1", "no such column: z")
    assert m[0]["role"] == "assistant" and m[1]["role"] == "user" and "no such column: z" in m[1]["content"]


def test_write_intent_is_refused_before_any_sql_runs():
    from answer_utils import has_write_intent
    from app import MockGenerator, answer_question, compose_answer, REFUSAL_TEXT

    for q in ["Delete all customers", "please drop the orders table", "Hapus semua pasien", "Can you update the price of all products?",
              "remove everything from products", "wipe the data"]:
        assert has_write_intent(q), q
        t = answer_question(MockGenerator(), "Online Store", "", q)
        assert not t.guard.allowed and t.sql == "" and not t.rows
        assert compose_answer(MockGenerator(), t)[0] == REFUSAL_TEXT
    for q in ["How many orders were deleted?", "What is the latest update date?", "Which customers have removed items?",
              "How many customers are from Jakarta?", "List products that dropped in price"]:
        assert not has_write_intent(q), q
