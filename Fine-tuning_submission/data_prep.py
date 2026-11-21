"""Stage 1: Text-to-SQL data preparation (cleaning, a split without schema leakage, complexity statistics).

Run:  python data_prep.py            (output: <work_dir>/data/{train,val,test}.jsonl + data_stats.json)
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_utils import get_work_dir, save_jsonl  # noqa: E402
from sql_utils import CATEGORIES, complexity_category, normalize_sql  # noqa: E402

CONFIG = {
    "dataset_name": "b-mc2/sql-create-context",   # columns: question, context (CREATE TABLE), answer (SQL)
    "seed": 42,
    "n_train": 5000,          # economical setting
    "n_val": 200,
    "n_test": 300,
    "group_fracs": (0.90, 0.05, 0.05),   # fraction of SCHEMAS (not rows) for train/val/test
    "max_chars_question": 300,
    "max_chars_context": 1200,
    "max_chars_answer": 500,
}


def load_raw(cfg: dict) -> list[dict]:
    from datasets import load_dataset  # import malas

    ds = load_dataset(cfg["dataset_name"], split="train")
    return [{"question": r["question"], "context": r["context"], "answer": r["answer"]} for r in ds]


def clean(records: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """Drop duplicates and examples that are empty or too long. Returns (clean data, statistics)."""
    stats = {"raw": len(records)}
    seen, kept = set(), []
    n_empty = n_long = n_dup = 0
    for r in records:
        q, c, a = (r["question"] or "").strip(), (r["context"] or "").strip(), (r["answer"] or "").strip()
        if not q or not c or not a:
            n_empty += 1
            continue
        if len(q) > cfg["max_chars_question"] or len(c) > cfg["max_chars_context"] or len(a) > cfg["max_chars_answer"]:
            n_long += 1
            continue
        key = (q.lower(), normalize_sql(c), normalize_sql(a))
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        kept.append({"question": q, "context": c, "answer": a, "category": complexity_category(a)})
    stats.update(removed_empty=n_empty, removed_too_long=n_long, removed_duplicates=n_dup, clean=len(kept))
    return kept, stats


def group_split(records: list[dict], cfg: dict) -> dict[str, list[dict]]:
    """Split per SCHEMA (context) so the same schema never appears in both train and test (prevents leakage)."""
    rng = random.Random(cfg["seed"])
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[normalize_sql(r["context"])].append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    f_train, f_val, _ = cfg["group_fracs"]
    a, b = int(len(keys) * f_train), int(len(keys) * (f_train + f_val))
    parts = {"train": keys[:a], "val": keys[a:b], "test": keys[b:]}
    out = {name: [r for k in ks for r in groups[k]] for name, ks in parts.items()}

    # extra leakage check: identical question+SQL across splits -> drop from val/test
    train_keys = {(r["question"].lower(), normalize_sql(r["answer"])) for r in out["train"]}
    for name in ("val", "test"):
        out[name] = [r for r in out[name] if (r["question"].lower(), normalize_sql(r["answer"])) not in train_keys]

    sizes = {"train": cfg["n_train"], "val": cfg["n_val"], "test": cfg["n_test"]}
    for name, n in sizes.items():
        rng.shuffle(out[name])
        out[name] = out[name][:n]
    return out


def split_stats(splits: dict[str, list[dict]]) -> dict:
    stats = {}
    for name, rs in splits.items():
        cats = Counter(r["category"] for r in rs)
        stats[name] = {
            "n": len(rs),
            "category_counts": {c: cats.get(c, 0) for c in CATEGORIES},
            "avg_chars_question": round(sum(len(r["question"]) for r in rs) / max(len(rs), 1), 1),
            "avg_chars_context": round(sum(len(r["context"]) for r in rs) / max(len(rs), 1), 1),
            "avg_chars_answer": round(sum(len(r["answer"]) for r in rs) / max(len(rs), 1), 1),
        }
    ctx = {name: {normalize_sql(r["context"]) for r in rs} for name, rs in splits.items()}
    stats["schema_overlap"] = {
        "train_val": len(ctx["train"] & ctx["val"]),
        "train_test": len(ctx["train"] & ctx["test"]),
        "val_test": len(ctx["val"] & ctx["test"]),
    }
    return stats


def prepare_data(cfg: dict = CONFIG, out_dir: Path | None = None) -> tuple[dict[str, list[dict]], dict]:
    out_dir = Path(out_dir) if out_dir else get_work_dir() / "data"
    cleaned, clean_stats = clean(load_raw(cfg), cfg)
    splits = group_split(cleaned, cfg)
    stats = {"cleaning": clean_stats, "splits": split_stats(splits), "config": {k: v for k, v in cfg.items()}}
    for name, rs in splits.items():
        save_jsonl(rs, out_dir / f"{name}.jsonl")
    (out_dir / "data_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return splits, stats


if __name__ == "__main__":
    _, st = prepare_data()
    print(json.dumps(st["splits"], indent=2, ensure_ascii=False))
