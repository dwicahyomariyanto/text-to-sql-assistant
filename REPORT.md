# Project Report: Text-to-SQL Assistant with LLM Fine-tuning (QLoRA)

Author: Dwi Cahyo Mariyanto

> The numbers in the Evaluation section are filled in automatically from `Evaluation_submission/results/` after the notebooks have been run
> (`python Evaluation_submission/evaluate.py --fill-docs`). No result number is typed by hand.

## 1. Domain

Accessing data in relational databases usually requires SQL, so non-technical people depend on analysts even for simple questions such as
"how many customers are from Jakarta?". *Text-to-SQL* translates a natural-language question into a SQL query so that data is easier to reach.
The main challenges: general large models often get table/column names wrong (schema hallucination) and may produce dangerous commands if run directly.

## 2. Business Understanding

**Problem statements**
1. How much can a small model (3B parameters, 4-bit) improve on Text-to-SQL through QLoRA fine-tuning compared with its base model?
2. How can the model's SQL be run safely?
3. How can it be presented in an interface that is comfortable for non-technical users?

**Goals**
1. Measure the improvement before vs after fine-tuning with exact match and execution match.
2. Provide a layered guardrail (validator + read-only connection + row/time limits).
3. Build a question-answering UI (Gradio) with example databases.

**Solution:** QLoRA fine-tuning (Unsloth) of a 3B instruct model, measured evaluation, a SQL guardrail, and a Gradio UI.
All work is designed to fit on Google Colab (T4 GPU, limited sessions) with economical settings.

## 3. Data Understanding

- **Dataset:** `b-mc2/sql-create-context` (Hugging Face), built from WikiSQL and Spider; **78,577** examples; license CC-BY-4.0.
- **Columns:** `question`, `context` (a `CREATE TABLE` statement used as the schema), `answer` (reference SQL).
- **Strength:** the context is only a schema (no table contents), which suits training a model not to invent table/column names.
- **Limitations:** English only; no table contents, so *execution match* is computed on synthetic data (see Evaluation).
- Actual statistics (count per split, complexity distribution, text lengths) are produced by notebook 01: `outputs/data/data_stats.json`
  and `results/complexity_distribution.png`.

## 4. Data Preparation

1. **Cleaning:** drop empty and overly long examples (question > 300, schema > 1200, SQL > 500 characters) and duplicates
   (normalized question + schema + SQL).
2. **Split by schema:** schemas are grouped (90/5/5 for train/val/test) and then subsampled: 5,000 train, 200 validation, 300 test (economical setting).
   This way the same schema never appears in both train and test (preventing leakage). Question+SQL pairs identical to training are also dropped from val/test.
3. **Complexity categories** (from the reference SQL): `simple`, `aggregation`, `join`, `nested` (subquery/UNION/INTERSECT/EXCEPT).
4. **A single prompt template** (`src/prompt_utils.py`): system + schema + question -> SQL, used identically in training, evaluation, and the UI.

## 5. Modeling

- **Base model:** `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit`. Chosen because (a) it is code-oriented and already familiar with SQL,
  (b) 3B parameters in 4-bit fit on a Colab T4 GPU, and (c) it is available ready-made in Unsloth.
- **QLoRA:** the base weights are frozen in 4-bit; only the LoRA adapter is trained.

| Parameter | Value | Reason |
|---|---|---|
| `max_seq_length` | 512 | dataset examples are short; saves memory and time (longer examples are dropped, not truncated) |
| LoRA `r` / `alpha` | 16 / 16 | enough capacity for a narrow task; adapter scale 1.0 |
| Target modules | q,k,v,o,gate,up,down | all attention and MLP projections (common QLoRA practice) |
| Effective batch | 4 x 4 = 16 | fits a T4 with gradient accumulation |
| Learning rate / scheduler | 2e-4 / cosine, warmup 5% | common QLoRA values, smooth decay |
| Epochs | 1 | economical setting; 5,000 examples |
| Loss | on the SQL answer only | the model learns to produce SQL, not to copy the prompt |

The loss curve and training log are produced by notebook 02 (`results/loss_curve.png`, `train_log.json`).
Ablation experiments (rank, learning rate, other models) were **deliberately not done** in this economical version.

## 6. Evaluation

**Metrics**
- **Exact match:** the normalized SQL (lowercase, whitespace, quotes, semicolon) is identical to the reference. Strict: a correct query written differently counts as wrong.
- **Execution match:** the predicted and the reference SQL are run on a SQLite database filled with **synthetic data** built from the schema
  (values are shared across tables so JOINs return rows, and literals from the query are inserted so `WHERE` is not automatically empty).
  Results are compared as a multiset of rows (order only counts when the reference uses `ORDER BY`).
  A stricter version that only counts cases with a non-empty reference result is also reported.

<!-- RESULTS_START -->
**Evaluation results (filled automatically from `results/metrics.json`):**

| Metric | Baseline (before tuning) | Fine-tuned (QLoRA) |
|---|---|---|
| Number of test examples | 300 | 300 |
| Exact match (normalized) | 49.7% | 75.3% |
| Execution match (synthetic data) | 46.6% | 86.2% |
| Execution match (non-empty gold result) | 45.6% | 86.7% |
| Prediction execution error rate | 4.0% | 0.0% |

**By complexity category (execution match):**

| Category | Baseline (before tuning) (n) | Fine-tuned (QLoRA) (n) |
|---|---|---|
| simple | 44.2% (217) | 86.1% (217) |
| aggregation | 50.6% (79) | 87.0% (79) |
| join | 100.0% (4) | 75.0% (4) |
| nested | - | - |

Final loss: train = 0.030521079897880554, validation = 0.04360850900411606 (313 steps).

Model on Hugging Face: https://huggingface.co/dwicahyom16/qwen2.5-coder-3b-text2sql-qlora

<!-- RESULTS_END -->

**Error analysis:** `Evaluation_submission/results/error_analysis.md` lists real failure examples per complexity category
with a likely cause (automatic heuristic; review manually before concluding).

**Guardrail:** the validator only lets a single `SELECT`/`WITH` statement through, rejects DDL/DML (DROP, DELETE, UPDATE, INSERT, ALTER, CREATE, PRAGMA, ATTACH, etc.),
checks the syntax against the schema, and limits the number of rows. Second layer: the SQLite connection has a read-only *authorizer* and an execution time limit.
Tested by `tests/test_guardrail.py` and `tests/test_sql_utils.py`.

## 7. Interface

Gradio app (`Interface_submission/app.py`) with an English interface: light/dark theme, SQL with syntax highlighting and a copy button, chat history,
guardrail badge, read-only result table on three synthetic example databases (Online Store, Campus, Hospital), a custom-schema mode
(syntax check only), and a progress indicator.

Features for free-form questions (details in `README.md`):
- **Own data:** several CSV files or one SQLite file can be uploaded (`src/upload_db.py`); the schema is built automatically and queries run on a read-only connection.
  Limits: 25 MB per file, at most 6 tables and 25 columns per table (small model context).
- **Robust answering** (`src/answer_utils.py`, `app.py`): Indonesian questions are translated to English; an invalid or failing query is sent back to the model with the
  error (up to 3 attempts); an empty result is retried with case-insensitive text comparison; the answer is written by the LLM (base model, adapter switched off) in the
  language of the question and every number in it is verified against the query result; refusals for non-read-only requests are fixed and final.

The core logic is tested with a mock model (`tests/`). The UI was run with the fine-tuned model on a T4 GPU (see the outputs of `04_interface_demo.ipynb`);
screenshots are stored in `Interface_submission/screenshots/`.

## 8. Conclusion and Limitations

On 300 test examples (schemas never seen during training), QLoRA fine-tuning raised *execution match* from 46.6% to 86.2% and
*exact match* from 49.7% to 75.3%. The prediction execution error rate fell from 4.0% to 0%. The final train loss is 0.031 and the validation loss 0.044 after 313 steps
(1 epoch, 5,000 examples), with no obvious sign of overfitting. This suggests that small, cheap fine-tuning (one T4 GPU) is already effective for this narrow task.

**How to read these numbers carefully:**
- Part of the gap between baseline and fine-tuned comes from **dataset conventions**, not only SQL ability. The dataset writes string values in lowercase and with double quotes
  (`WHERE team = "@ Dallas"` / `region = "australia"`), while the base model tends to write `'Australia'`. Because the comparison is *case-sensitive* on synthetic data,
  a different way of writing a value counts as wrong even when the SQL is sensible. Fine-tuning teaches that convention. This was seen in the baseline failure examples that were inspected; the exact proportion has not been measured.
- The test set is dominated by the `simple` category (217 of 300) and `aggregation` (79). The `join` category has only 4 examples and `nested` has 0,
  so the numbers for those categories (e.g. join 100% baseline vs 75% fine-tuned) are **not statistically meaningful** and must not be used to draw conclusions.
- The remaining failures (see `error_analysis.md`) are mostly wrong values or wrong aggregation, e.g. `count` vs a column value, `sum` vs `max`,
  or a value spelled differently from the question; they are no longer invalid queries.
- One run, one seed, no confidence intervals; a difference this large is probably robust, but the exact numbers may shift on another run.

**Known limitations**
- The model was trained on **English** questions. The UI translates Indonesian questions with the base model, which adds another possible source of error.
- Execution match uses synthetic data, not a real database; two different queries may coincidentally give the same result on small data.
- Small subset (5,000 examples, 1 epoch) and a single run without ablation; no variance across seeds.
- The history-as-context mode is experimental (the model was trained on single turns).
- The target dialect is SQLite.
- The model can write a suboptimal or wrong query for harder questions (example in the UI: an unnecessary JOIN made a doctor's name appear twice),
  and the LLM-written answer can repeat that mistake; verification only checks the numbers, not the correctness of the query. Self-correction can only fix queries that fail or return nothing.

**Further work:** an additional evaluation with case-insensitive value comparison to separate SQL ability from dataset conventions,
ablation (rank/learning rate/model), evaluation on Spider with real databases, Indonesian training data,
and packaging the model as an API (FastAPI + Docker).

## Attribution
Dataset `b-mc2/sql-create-context` (CC-BY-4.0), built from WikiSQL and Spider. Base model Qwen2.5-Coder (Alibaba Cloud/Qwen) through Unsloth.
