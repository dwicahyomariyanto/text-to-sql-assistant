---
base_model: unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit
datasets:
  - b-mc2/sql-create-context
library_name: peft
language:
  - en
license: other
tags:
  - text-to-sql
  - sql
  - qlora
  - unsloth
---

# Qwen2.5-Coder-3B Text-to-SQL (QLoRA adapter)

LoRA adapter fine-tuned from `Qwen2.5-Coder-3B-Instruct` (4-bit) that turns a natural-language question and a
`CREATE TABLE` schema into a single SQL query (SQLite dialect).

## Usage

```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("dwicahyom16/qwen2.5-coder-3b-text2sql-qlora", max_seq_length=512, load_in_4bit=True)
FastLanguageModel.for_inference(model)
```

Use the same prompt template as in training (`src/prompt_utils.py` in the project repository): a system message, a
`### Schema` section, then `### Question`. The model output contains only SQL.

## Training data
`b-mc2/sql-create-context` (CC-BY-4.0; built from WikiSQL and Spider; 78,577 examples). A subset of **5,000** training examples,
200 validation, and 300 test examples; split by schema so that no schema appears in more than one split.

## Training procedure
QLoRA (Unsloth): r=16, alpha=16, targeting all attention and MLP projections, learning rate 2e-4 (cosine, warmup 5%), effective batch 16,
1 epoch, `max_seq_length` 512; loss on the SQL answer only.

## Evaluation

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

Metrics: exact match (normalized SQL) and execution match on SQLite with **synthetic** data (not a real database).
Part of the baseline-vs-fine-tuned gap comes from dataset conventions (lowercase string values, double quotes); see the project report.

## Limitations and out-of-scope use
- Trained on **English** questions; other languages may be less accurate.
- It can produce wrong or inefficient SQL; always review it before running on important data.
- Do not run the model's SQL directly on a production database without a guardrail (SELECT only, read-only, row/time limits).
- Small subset and a single training run without ablation.

## License and attribution
The adapter follows the license terms of the base model Qwen2.5-Coder-3B-Instruct (check the base model page before commercial use)
and the CC-BY-4.0 dataset (`b-mc2/sql-create-context`).
