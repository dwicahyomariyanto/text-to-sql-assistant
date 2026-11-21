"""Stage 3: baseline vs fine-tuned evaluation (needs a GPU) and automatic filling of the documents from the results.

Run evaluation :  python evaluate.py
Fill documents  :  python evaluate.py --fill-docs [--results <results folder>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "Fine-tuning_submission"))

from eval_utils import (  # noqa: E402
    evaluate_records, generate_sql_batch, get_hf_token, get_work_dir, load_for_inference, load_jsonl,
    save_jsonl, write_error_analysis, write_metrics, write_results_table,
)

CONFIG = {
    "base_model": "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
    "adapter": "",            # HF repo "username/repo" OR a local path; empty -> <work_dir>/adapter
    "max_seq_length": 512,
    "batch_size": 16,
    "max_new_tokens": 128,
}
START, END = "<!-- RESULTS_START -->", "<!-- RESULTS_END -->"


def _free_gpu() -> None:
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def get_predictions(name: str, model_name: str, test: list[dict], cfg: dict, res_dir: Path) -> list[dict]:
    """Use saved predictions (e.g. the baseline from notebook 01) if they match the test data, otherwise generate them."""
    cached = res_dir / f"{name}_predictions.jsonl"
    if cached.exists():
        recs = load_jsonl(cached)
        if [r["question"] for r in recs] == [r["question"] for r in test]:
            print(f"[{name}] using saved predictions: {cached.name}")
            return [{k: r[k] for k in ("question", "context", "answer", "category", "pred") if k in r} for r in recs]
    print(f"[{name}] generating predictions with {model_name} ...")
    model, tokenizer = load_for_inference(model_name, cfg["max_seq_length"], token=get_hf_token())
    preds = generate_sql_batch(model, tokenizer, test, cfg["batch_size"], cfg["max_new_tokens"])
    del model, tokenizer
    _free_gpu()
    return [dict(r, pred=p) for r, p in zip(test, preds)]


def run_evaluation(cfg: dict = CONFIG) -> dict:
    from data_prep import CONFIG as DATA_CONFIG, prepare_data

    work = get_work_dir()
    res_dir = work / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    if not (work / "data" / "test.jsonl").exists():
        prepare_data(DATA_CONFIG, work / "data")        # same seed -> identical split as notebooks 01/02
    test = load_jsonl(work / "data" / "test.jsonl")
    adapter = cfg["adapter"] or str(work / "adapter")

    all_metrics, records = {}, {}
    for name, model_name in (("baseline", cfg["base_model"]), ("finetuned", adapter)):
        recs = get_predictions(name, model_name, test, cfg, res_dir)
        all_metrics[name] = evaluate_records(recs)
        records[name] = recs
        write_metrics(all_metrics[name], res_dir / f"metrics_{name}.json")
        save_jsonl(recs, res_dir / f"{name}_predictions.jsonl")

    write_metrics(all_metrics, res_dir / "metrics.json")
    write_results_table({"Baseline (before tuning)": all_metrics["baseline"], "Fine-tuned (QLoRA)": all_metrics["finetuned"]}, res_dir / "results_table.md")
    write_error_analysis(records["finetuned"], res_dir / "error_analysis.md", title="Error Analysis - fine-tuned model")
    return all_metrics


def fill_docs(results_dir: Path, doc_paths: list[Path]) -> list[str]:
    """Fill the block between the RESULTS_START/RESULTS_END markers in the documents with REAL results from the results folder."""
    if not (results_dir / "results_table.md").exists():
        raise SystemExit(f"results_table.md not found in {results_dir}. Copy the contents of the results zip (outputs/results/) from Colab into that folder first.")
    table = (results_dir / "results_table.md").read_text(encoding="utf-8")
    block = ["**Evaluation results (filled automatically from `results/metrics.json`):**", "", table]
    summary_path = results_dir / "train_summary.json"
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        block += [f"Final loss: train = {s.get('final_train_loss')}, validation = {s.get('final_eval_loss')} ({s.get('steps')} steps).", ""]
        if s.get("hub_model_id"):
            block += [f"Model on Hugging Face: https://huggingface.co/{s['hub_model_id']}", ""]
    new_block = f"{START}\n" + "\n".join(block) + f"\n{END}"
    done = []
    for p in doc_paths:
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        if START in text and END in text:
            p.write_text(re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _m: new_block, text, flags=re.S), encoding="utf-8")
            done.append(p.name)
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-docs", action="store_true", help="fill REPORT.md, README.md, MODEL_CARD.md from the results folder")
    ap.add_argument("--results", default=str(ROOT / "Evaluation_submission" / "results"))
    args = ap.parse_args()
    if args.fill_docs:
        print("Documents filled:", fill_docs(Path(args.results), [ROOT / n for n in ("REPORT.md", "README.md", "MODEL_CARD.md")]))
    else:
        run_evaluation()
