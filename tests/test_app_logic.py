"""UI logic tests with a MOCK model. Not real model output; they only prove the application flow."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Interface_submission"))

from app import CUSTOM, MockGenerator, answer_question, render_badge, render_chat  # noqa: E402


class BrokenGenerator:
    def generate(self, schema, question, history=None):
        raise RuntimeError("GPU habis")


def test_safe_question_runs_on_demo_db():
    t = answer_question(MockGenerator(), "Online Store", "", "How many customers are there?")
    assert t.guard.allowed and t.sql.lower().startswith("select count")
    assert t.columns and t.rows[0][0] == 20


def test_dangerous_command_is_rejected_and_not_executed():
    t = answer_question(MockGenerator(), "Campus", "", "Please delete everything")   # refused from the question itself
    assert t.sql == "" and not t.guard.allowed and t.guard.reason
    t = answer_question(MockGenerator(), "Campus", "", "Get rid of everything, drop it")   # sneaks past the intent check -> SQL guardrail
    assert t.sql.upper().startswith("DROP") and not t.guard.allowed and t.guard.reason
    assert t.rows == [] and "not executed" in t.note


def test_custom_database_only_validates_syntax():
    t = answer_question(MockGenerator(), CUSTOM, "CREATE TABLE city (name VARCHAR, population INTEGER)", "How many cities?")
    assert t.guard.allowed and t.rows == [] and "syntax-checked" in t.note


def test_empty_input_and_model_errors_are_handled_gracefully():
    assert answer_question(MockGenerator(), "Campus", "", "   ").note
    assert not answer_question(MockGenerator(), CUSTOM, "", "How many?").guard.allowed       # skema kosong
    t = answer_question(BrokenGenerator(), "Campus", "", "How many students?")
    assert not t.guard.allowed and "GPU habis" in t.note


def test_render_escapes_html():
    out = render_chat([{"question": "<script>alert(1)</script>", "sql": "SELECT '<b>'", "allowed": True}])
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "badge-ok" in render_badge(answer_question(MockGenerator(), "Campus", "", "How many students?").guard)
    assert "No conversation yet" in render_chat([])
