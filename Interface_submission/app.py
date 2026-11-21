"""Text-to-SQL question-answering interface (Gradio).

Run:  python app.py --model <hf-username>/qwen2.5-coder-3b-text2sql-qlora [--share]
Test the UI without a GPU/model:  python app.py --mock
The core logic (answer_question, compose_answer, render_*) does not depend on Gradio, so it can be tested on its own.

Robustness for free-form questions:
  * Indonesian questions are translated to English first (the SQL model was trained on English).
  * A query that is invalid or fails is sent back to the model together with the error (up to MAX_ATTEMPTS tries).
  * A query that returns no rows is retried once with case-insensitive text comparison.
  * Answers are written by the LLM from the query result; every number is verified against the data.
  * Safety refusals (anything that is not a read-only SELECT) are final and are never "fixed" into something runnable.
"""
from __future__ import annotations

import argparse
import contextlib
import html
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from answer_utils import (  # noqa: E402
    accept_answer, accept_plain, build_explain_messages, build_failure_messages, build_translate_messages,
    clean_translation, detect_language, feedback_messages, has_write_intent, is_hard_refusal, relax_string_equality,
)
from demo_db import create_demo_connection, get_examples, get_schema, list_databases  # noqa: E402
from guardrail import GuardrailResult, validate_sql  # noqa: E402
from prompt_utils import build_messages, extract_sql  # noqa: E402
from sql_utils import build_sqlite_db, parse_schema, safe_execute  # noqa: E402
from upload_db import UploadError, import_files, open_readonly  # noqa: E402

CUSTOM = "Custom (paste your own schema)"
UPLOAD = "My data (upload CSV / SQLite)"
MAX_ROWS = 100
HISTORY_TURNS = 2
MAX_ATTEMPTS = 3          # SQL generation tries per question (first try + self-corrections)
EXPLAIN_TRIES = 3         # LLM answer tries before the plain fallback is used

REFUSAL_TEXT = (
    "Sorry, I did not run that request because it is not safe. "
    "I can only read data (SELECT), not change or delete it. Try asking something about what the data contains."
)
FRIENDLY_ERROR = "Something went wrong on my side while preparing the answer. Please try again."


# ------------------------------------------------------------------ generators
class MockGenerator:
    """Rule-based stand-in model: ONLY for testing the UI/guardrail, not real model output."""

    name = "mock"

    def generate(self, schema: str, question: str, history=None, feedback=None) -> str:
        tables = parse_schema(schema)
        if not tables:
            return "SELECT 1"
        table, cols = next(iter(tables.items()))
        q = question.lower()
        if any(w in q for w in ("drop", "delete", "hapus")):
            return f"DROP TABLE {table}"
        if "how many" in q or "berapa" in q:
            return f"SELECT COUNT(*) FROM {table}"
        if "average" in q or "rata" in q:
            num = next((c for c, t in cols if any(k in t.upper() for k in ("INT", "REAL"))), cols[0][0])
            return f"SELECT AVG({num}) FROM {table}"
        return f"SELECT * FROM {table}"

    # The mock writes no sentences: the UI falls back to the plain summary.
    def explain(self, question, sql, columns, rows, language="en", strict=False):
        return None

    def explain_failure(self, question, schema_tables, language="en"):
        return None

    def translate(self, question, schema):
        return None


class HFGenerator:
    """Fine-tuned model (HF repo or local adapter folder), loaded through Unsloth. Needs a GPU.

    The fine-tuned adapter writes SQL. For everything else (translation, answers) the adapter is switched off
    temporarily so the base instruct model is used as a normal chat model.
    """

    def __init__(self, model_id: str, max_seq_length: int = 1024, max_new_tokens: int = 160):
        from eval_utils import get_hf_token, load_for_inference, silence_generation_length_warning

        silence_generation_length_warning()
        self.name, self.max_new_tokens = model_id, max_new_tokens
        self.model, self.tokenizer = load_for_inference(model_id, max_seq_length, token=get_hf_token())

    def _run(self, msgs, max_new_tokens, sample=False, base=False) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        off = getattr(self.model, "disable_adapter", None) if base else None
        if base and off is None:
            raise RuntimeError("adapter switch unavailable")
        kwargs = dict(max_new_tokens=max_new_tokens, use_cache=True, pad_token_id=self.tokenizer.pad_token_id)
        kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9) if sample else dict(do_sample=False))
        with torch.no_grad(), (off() if off else contextlib.nullcontext()):
            out = self.model.generate(**enc, **kwargs)
        return self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def generate(self, schema: str, question: str, history=None, feedback=None) -> str:
        msgs = build_messages(schema, question, history=(history or [])[-HISTORY_TURNS:])
        if feedback:
            msgs += feedback_messages(*feedback)
        return extract_sql(self._run(msgs, self.max_new_tokens))

    def translate(self, question: str, schema: str):
        return clean_translation(self._run(build_translate_messages(question, schema), 80, base=True), question)

    def explain(self, question, sql, columns, rows, language="en", strict=False):
        msgs = build_explain_messages(question, sql, columns, rows, language=language, strict=strict)
        return self._run(msgs, 200, sample=not strict, base=True)

    def explain_failure(self, question, schema_tables, language="en"):
        return self._run(build_failure_messages(question, schema_tables, language), 120, sample=True, base=True)


# ------------------------------------------------------------------ core logic
@dataclass
class Turn:
    question: str
    sql: str
    guard: GuardrailResult
    columns: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    note: str = ""
    language: str = "en"
    attempts: int = 1
    failed: bool = False          # no usable query after all attempts
    schema: str = ""
    q_en: str = ""


def _open_connection(db_name, schema, db_path):
    if db_name == UPLOAD:
        return open_readonly(db_path)
    if db_name == CUSTOM:
        return build_sqlite_db(schema)
    return create_demo_connection(db_name)


def answer_question(generator, db_name: str, schema: str, question: str, history=None, use_history: bool = False, db_path: str = "") -> Turn:
    """Question -> (translate) -> SQL -> guardrail -> read-only execution, with automatic self-correction."""
    question = (question or "").strip()
    if not question:
        return Turn("", "", GuardrailResult(False, "Empty question."), note="Please type a question first.")
    uploaded = db_name == UPLOAD
    demo = db_name not in (CUSTOM, UPLOAD)
    has_data = demo or uploaded
    schema = (get_schema(db_name) if demo else schema or "").strip()
    if not schema:
        return Turn(question, "", GuardrailResult(False, "Empty schema."), note="Please enter a CREATE TABLE schema first.")
    if uploaded and not db_path:
        return Turn(question, "", GuardrailResult(False, "No data uploaded."), note="Please upload a CSV or SQLite file first.")

    language = detect_language(question)
    if has_write_intent(question):   # the request itself asks to change data: refuse up front, never run anything
        return Turn(question, "", GuardrailResult(False, "Only SELECT queries are allowed: this request asks to change or delete data."),
                    note="The query was not executed.", language=language, schema=schema, q_en=question)
    q_en = question
    if language == "id" and hasattr(generator, "translate"):
        try:
            q_en = generator.translate(question, schema) or question
        except Exception:  # noqa: BLE001 - translation is best effort; the original text is still tried
            q_en = question

    hist = history if use_history else None
    conn = _open_connection(db_name, schema, db_path)
    feedback, last_sql, last_guard, last_error = None, "", None, ""
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                sql = generator.generate(schema, q_en, hist) if feedback is None else generator.generate(schema, q_en, hist, feedback)
            except Exception as exc:  # noqa: BLE001 - infrastructure problem (e.g. out of GPU memory)
                return Turn(question, last_sql, GuardrailResult(False, "Failed to generate SQL."), note=f"The model ran into a problem: {exc}",
                            language=language, attempts=attempt, schema=schema, q_en=q_en)
            last_sql = sql
            guard = validate_sql(sql, MAX_ROWS, conn)
            last_guard = guard
            if not guard.allowed:
                if is_hard_refusal(guard.reason):
                    return Turn(question, sql, guard, note="The query was not executed.", language=language, attempts=attempt, schema=schema, q_en=q_en)
                feedback, last_error = (sql, guard.reason), guard.reason
                continue
            if not has_data:
                return Turn(question, guard.sanitized_sql, guard, note="Custom database: the SQL is only syntax-checked (no data).",
                            language=language, attempts=attempt, schema=schema, q_en=q_en)
            try:
                cols, rows = safe_execute(conn, guard.sanitized_sql, MAX_ROWS)
            except Exception as exc:  # noqa: BLE001
                feedback, last_error = (guard.sanitized_sql, str(exc)), str(exc)
                continue
            final_sql, note = guard.sanitized_sql, f"{len(rows)} row(s) (max {MAX_ROWS})."
            if _is_empty(rows):   # maybe only the letter case of a text value differs
                relaxed, changed = relax_string_equality(final_sql)
                if changed:
                    g2 = validate_sql(relaxed, MAX_ROWS, conn)
                    if g2.allowed:
                        try:
                            cols2, rows2 = safe_execute(conn, g2.sanitized_sql, MAX_ROWS)
                        except Exception:  # noqa: BLE001
                            cols2, rows2 = cols, []
                        if not _is_empty(rows2):
                            cols, rows, final_sql = cols2, rows2, g2.sanitized_sql
                            note = f"{len(rows)} row(s) (max {MAX_ROWS}). Text values were matched case-insensitively."
            if attempt > 1:
                note += f" The query was corrected automatically (attempt {attempt})."
            return Turn(question, final_sql, guard, cols, rows, note, language, attempt, schema=schema, q_en=q_en)
        reason = last_error or "No valid query could be produced."
        return Turn(question, last_sql, last_guard or GuardrailResult(False, reason), note=f"No usable query after {MAX_ATTEMPTS} attempts: {reason}",
                    language=language, attempts=MAX_ATTEMPTS, failed=True, schema=schema, q_en=q_en)
    finally:
        conn.close()


def _is_empty(rows) -> bool:
    """No rows, or a single aggregate row that is 0/NULL (e.g. COUNT(*) = 0)."""
    return not rows or (len(rows) == 1 and all(v in (0, None) for v in rows[0]))


def _fmt(v) -> str:
    """Format numbers (1,234.5); None -> 'empty'."""
    if v is None:
        return "empty"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)


def describe_result(turn: "Turn") -> str:
    """Plain, rule-based summary. Last resort only: used when no LLM is available or its answer fails verification."""
    if not turn.question:
        return turn.note
    if not turn.guard.allowed and is_hard_refusal(turn.guard.reason):
        return REFUSAL_TEXT
    if turn.note.startswith("The model ran into"):
        return FRIENDLY_ERROR
    if turn.failed:
        return "Sorry, I could not answer that from this data. Try rephrasing the question with the table and column names from the schema."
    if turn.note.startswith("Custom database"):
        return "The SQL was generated and its syntax is valid. A custom database has no data, so there is no result to show yet."
    if not turn.sql:
        return turn.note
    if not turn.rows:
        return "I searched, but no data matches your question."
    cols, rows, n = turn.columns, turn.rows, len(turn.rows)
    if n == 1 and len(cols) == 1:
        return f"The answer is {_fmt(rows[0][0])}."
    if n == 1:
        return "Result: " + ", ".join(f"{c} = {_fmt(v)}" for c, v in zip(cols, rows[0])) + "."
    if len(cols) == 1:
        shown = ", ".join(_fmt(r[0]) for r in rows[:8])
        more = f", and {n - 8} more" if n > 8 else ""
        return f"There are {n} results: {shown}{more}."
    ex = "; ".join("(" + ", ".join(f"{c} = {_fmt(v)}" for c, v in zip(cols, r)) + ")" for r in rows[:3])
    return f"Found {n} rows. Top examples: {ex}{' ...' if n > 3 else ''}. See the Results tab for the full table."


def compose_answer(generator, turn: "Turn", use_llm: bool = True) -> tuple[str, str]:
    """Final answer text and its source: 'llm', 'fixed' (safety refusal / input hints) or 'plain' (fallback)."""
    if not turn.question or (not turn.sql and not turn.failed):
        return describe_result(turn), "fixed"
    if not turn.guard.allowed and is_hard_refusal(turn.guard.reason):
        return REFUSAL_TEXT, "fixed"          # deliberately not LLM-written: a safety message must be predictable
    if not use_llm:
        return describe_result(turn), "plain"
    try:
        if turn.failed:
            fn = getattr(generator, "explain_failure", None)
            raw = fn(turn.question, parse_schema(turn.schema), turn.language) if fn else None
            good = accept_plain(raw) if raw else None
            return (good, "llm") if good else (describe_result(turn), "plain")
        fn = getattr(generator, "explain", None)
        if fn is None or not turn.guard.allowed or turn.note.startswith(("Custom database", "The model ran into")):
            return describe_result(turn), "plain"
        for i in range(EXPLAIN_TRIES):
            raw = fn(turn.question, turn.sql, turn.columns, turn.rows, turn.language, i == EXPLAIN_TRIES - 1)
            good = accept_answer(raw, turn.question, turn.columns, turn.rows) if raw else None
            if good:
                return good, "llm"
    except Exception:  # noqa: BLE001 - an LLM problem must never break the UI
        pass
    return describe_result(turn), "plain"


def render_badge(guard: GuardrailResult | None) -> str:
    if guard is None:
        return ""
    cls, icon = ("badge-ok", "&#10003;") if guard.allowed else ("badge-bad", "&#10007;")
    return f'<div class="t2s-badge {cls}">{icon} {html.escape(guard.reason)}</div>'


def render_chat(history: list[dict], pending: str = "") -> str:
    """Conversation history as HTML (all text is escaped)."""
    if not history and not pending:
        return '<div class="t2s-chat t2s-empty">No conversation yet. Pick a database, then ask a question.</div>'
    parts = ['<div class="t2s-chat">']
    for h in history:
        mark = "&#10003;" if h["allowed"] else "&#10007; rejected"
        if h["allowed"] and h.get("attempts", 1) > 1:
            mark += " auto-corrected"
        answer = html.escape(h.get("answer", "")).replace("**", "")
        sql_box = f'<details><summary>Show SQL</summary><code>{html.escape(h["sql"])}</code></details>' if h["sql"] else ""
        parts.append(f'<div class="bubble bubble-user">{html.escape(h["question"])}</div>')
        parts.append(f'<div class="bubble bubble-bot">{answer}{sql_box}<span class="mini">{mark}</span></div>')
    if pending:
        parts.append(f'<div class="bubble bubble-user">{html.escape(pending)}</div>')
        parts.append('<div class="bubble bubble-bot loading">Thinking&hellip;</div>')
    parts.append("</div>")
    return "".join(parts)


# ------------------------------------------------------------------ UI Gradio
def build_ui(generator):
    import gradio as gr
    import pandas as pd

    css = (HERE / "assets" / "style.css").read_text(encoding="utf-8")
    theme = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate")
    # Gradio >= 6 moved theme/css from Blocks() to launch(); Gradio 5 is the opposite.
    modern = _gradio_major(gr) >= 6
    blocks = gr.Blocks(title="Text-to-SQL Assistant") if modern else gr.Blocks(theme=theme, title="Text-to-SQL Assistant")

    first = list_databases()[0]
    with blocks as demo:
        gr.HTML(f"<style>{css}</style>")   # CSS juga disuntik langsung agar berlaku di semua versi Gradio
        gr.HTML(
            '<div class="t2s-header"><h1>Text-to-SQL Assistant</h1>'
            "<p>Ask a database a question in natural language; the model writes the SQL, "
            "a guardrail checks it, then it runs read-only.</p>"
            "<p class='hint'>The model was trained on English questions, so English questions work best.</p></div>"
        )
        state = gr.State([])
        db_path = gr.State("")
        with gr.Row():
            with gr.Column(scale=4):
                db = gr.Dropdown(list_databases() + [UPLOAD, CUSTOM], value=first, label="Database")
                upload = gr.File(label="Upload your data (several .csv files at once, or one .db/.sqlite file)", file_count="multiple",
                                 file_types=[".csv", ".tsv", ".db", ".sqlite", ".sqlite3"], visible=False)
                upload_info = gr.Markdown(visible=False)
                schema = gr.Textbox(value=get_schema(first), label="Schema (CREATE TABLE)", lines=9, interactive=False)
                use_hist = gr.Checkbox(False, label="Use chat history as context (experimental)")
                use_llm = gr.Checkbox(True, label="Natural-language answer by LLM (numbers verified against the data)")
                gr.Markdown("**Example questions** (click to fill in)")
                ex_btns = [gr.Button(q, size="sm") for q in get_examples(first)]
            with gr.Column(scale=6):
                chat = gr.HTML(render_chat([]))
                with gr.Row():
                    question = gr.Textbox(placeholder="Example: How many customers are from Jakarta?", show_label=False, scale=8, lines=1)
                    send = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    clear = gr.Button("Clear history", size="sm")
                    mode = gr.Button("Dark/light mode", size="sm")
                badge = gr.HTML()
                with gr.Tabs():
                    with gr.Tab("SQL"):
                        sql_out = gr.Code(language="sql", label="Generated SQL", interactive=False)
                    with gr.Tab("Results"):
                        table = gr.Dataframe(label="Execution result (read-only)", interactive=False)
                        note = gr.Markdown()

        def on_db_change(name):
            hide = [gr.update(visible=False)] * 3
            if name == UPLOAD:
                return [gr.update(value="", interactive=False, placeholder="The schema is filled in automatically after you upload data.")] + [
                    gr.update(visible=True), gr.update(visible=True, value="**Your data is processed in this session.** Do not upload confidential data if you use a public share link."), ] + hide
            if name == CUSTOM:
                return [gr.update(value="", interactive=True, placeholder="CREATE TABLE ...; CREATE TABLE ...")] + [gr.update(visible=False)] * 2 + hide
            ex = get_examples(name)
            return [gr.update(value=get_schema(name), interactive=False)] + [gr.update(visible=False)] * 2 + [gr.update(value=q, visible=True) for q in ex]

        def on_upload(files):
            paths = [getattr(f, "name", None) or (f.get("path") if isinstance(f, dict) else f) for f in (files or [])]
            if not paths:
                return gr.update(value=""), "", gr.update(value="")
            try:
                res = import_files(paths)
            except UploadError as exc:
                return gr.update(value=""), "", gr.update(value=f"**Failed:** {exc}")
            except Exception as exc:  # noqa: BLE001
                return gr.update(value=""), "", gr.update(value=f"**Could not read the data:** {exc}")
            return gr.update(value=res.schema), res.db_path, gr.update(value=res.message)

        def on_send(db_name, schema_text, q, hist, use_h, path, llm_on):
            yield render_chat(hist, pending=(q or "").strip()), '<div class="t2s-badge badge-wait">&#9203; Thinking&hellip;</div>', gr.update(), gr.update(), gr.update(), hist, gr.update()
            past = [(h.get("q_en") or h["question"], h["sql"]) for h in hist if h["allowed"]]
            try:
                turn = answer_question(generator, db_name, schema_text, q, past, use_h, path)
                answer = compose_answer(generator, turn, llm_on)[0]
            except Exception:  # noqa: BLE001 - never show a stack trace to the user
                yield render_chat(hist), "", "", pd.DataFrame(), FRIENDLY_ERROR, hist, q
                return
            if turn.question and (turn.sql or answer):
                hist = hist + [{"question": turn.question, "sql": turn.sql, "allowed": turn.guard.allowed, "answer": answer,
                                "attempts": turn.attempts, "q_en": turn.q_en or turn.question}]
            df = pd.DataFrame(turn.rows, columns=turn.columns) if turn.columns else pd.DataFrame()
            yield render_chat(hist), render_badge(turn.guard), turn.sql, df, turn.note, hist, ""

        def on_clear():
            return render_chat([]), "", "", pd.DataFrame(), "", [], ""

        io_in = [db, schema, question, state, use_hist, db_path, use_llm]
        io_out = [chat, badge, sql_out, table, note, state, question]
        send.click(on_send, io_in, io_out)
        question.submit(on_send, io_in, io_out)
        db.change(on_db_change, db, [schema, upload, upload_info] + ex_btns)
        upload.change(on_upload, upload, [schema, db_path, upload_info])
        clear.click(on_clear, None, io_out)
        for b in ex_btns:
            b.click(lambda text: text, b, question)
        mode.click(None, None, None, js="() => { document.body.classList.toggle('dark'); }")
    demo._t2s_launch_kwargs = {"theme": theme, "css": css} if modern else {}
    return demo


def _gradio_major(gr) -> int:
    try:
        return int(str(gr.__version__).split(".")[0])
    except (AttributeError, ValueError):
        return 5


def launch_ui(demo, **kwargs):
    """Luncurkan UI dengan argumen theme/css yang sesuai versi Gradio terpasang."""
    for key, value in getattr(demo, "_t2s_launch_kwargs", {}).items():
        kwargs.setdefault(key, value)
    return demo.queue().launch(**kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("T2S_MODEL", ""), help="HF repo or local folder of the fine-tuned adapter")
    ap.add_argument("--mock", action="store_true", help="use the mock model (only to test the UI)")
    ap.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    if not args.mock and not args.model:
        sys.exit("Set --model (HF repo/adapter) or use --mock to test the UI.")
    generator = MockGenerator() if args.mock else HFGenerator(args.model)
    demo = build_ui(generator)
    launch_ui(demo, share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
