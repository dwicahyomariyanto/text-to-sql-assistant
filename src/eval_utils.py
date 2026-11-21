"""Evaluation (exact match, execution match), result writing, and GPU/HF helpers (lazy imports)."""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from prompt_utils import build_messages, extract_sql
from sql_utils import CATEGORIES, complexity_category, exact_match, execution_match, mask_literals, normalize_sql


# ---------------------------------------------------------------- lingkungan & I/O
def is_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def is_kaggle() -> bool:
    return Path("/kaggle/working").exists()


def get_work_dir() -> Path:
    """Folder keluaran (data, results, adapter).

    Priority: env T2S_WORK_DIR (e.g. a Google Drive folder so results survive a dropped Colab session)
    -> /kaggle/working on Kaggle -> /content on Colab -> the current folder. Always inside an 'outputs' subfolder.
    """
    if os.environ.get("T2S_WORK_DIR"):
        base = Path(os.environ["T2S_WORK_DIR"])
    elif is_kaggle():
        base = Path("/kaggle/working")
    elif Path("/content").exists() and is_colab():
        base = Path("/content")
    else:
        base = Path.cwd()
    out = base / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def get_scratch_dir() -> Path:
    """Fast local scratch folder (large training checkpoints). Do not point it at Google Drive."""
    if os.environ.get("T2S_SCRATCH_DIR"):
        out = Path(os.environ["T2S_SCRATCH_DIR"])
    elif Path("/content").exists() and is_colab():
        out = Path("/content/scratch")
    else:
        return get_work_dir()
    out.mkdir(parents=True, exist_ok=True)
    return out


_WARNED_TOKEN = False


def get_hf_token() -> Optional[str]:
    """Get HF_TOKEN from an env var, Kaggle Secrets, or Colab userdata (never hardcoded).

    Leading/trailing whitespace is stripped (often pasted along with the token). If Kaggle Secrets exist but
    cannot be read, the reason is printed once (the token value is never printed).
    """
    global _WARNED_TOKEN
    token = (os.environ.get("HF_TOKEN") or "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
    except Exception:
        UserSecretsClient = None  # not on Kaggle
    if UserSecretsClient is not None:
        try:
            value = (UserSecretsClient().get_secret("HF_TOKEN") or "").strip()
            if value:
                return value
            reason = "the HF_TOKEN secret is empty"
        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
        if not _WARNED_TOKEN:
            _WARNED_TOKEN = True
            print(f"[warning] HF_TOKEN could not be read from Kaggle Secrets ({reason}). Make sure the name is exactly HF_TOKEN and "
                  "'Attached' is ticked under Add-ons -> Secrets for this notebook, then Restart session.")
    try:
        from google.colab import userdata  # type: ignore

        return (userdata.get("HF_TOKEN") or "").strip() or None
    except Exception:
        return None


def ensure_hf_token(manual: str = "", required: bool = True, validate: bool = True) -> Optional[str]:
    """Make sure HF_TOKEN is available. Order: manual value -> env/Secrets -> hidden input box.

    The token is validated against Hugging Face (whoami) and stored in os.environ["HF_TOKEN"] for later cells.
    The token value is never printed.
    """
    token = (manual or "").strip() or get_hf_token()
    if not token and required:
        from getpass import getpass

        token = getpass("Tempel HF_TOKEN (izin Write) lalu Enter: ").strip()
    if not token:
        if required:
            raise ValueError("HF_TOKEN is empty. Run this cell again and paste a token (create one at huggingface.co/settings/tokens).")
        return None
    if validate:
        try:
            from huggingface_hub import HfApi

            print("Logged in to Hugging Face as:", HfApi().whoami(token=token)["name"])
        except Exception as e:
            raise ValueError(
                f"HF_TOKEN was rejected or Hugging Face could not be reached ({type(e).__name__}). Check the token (Write permission) "
                "and your Internet connection, then run this cell again."
            ) from None
    os.environ["HF_TOKEN"] = token
    return token


def prepare_hub_repo(cfg: dict, token: str) -> str:
    """Fill cfg['hf_username'] if empty (from the token), then create the target repo. Fails HERE if the token has no Write permission."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = cfg.get("hf_username") or api.whoami()["name"]
    cfg["hf_username"] = user
    repo_id = f"{user}/{cfg['hub_repo_name']}"
    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create/access {repo_id} ({type(e).__name__}). Make sure the token has Write permission.") from None
    return repo_id


def _adapter_ok(candidate: str, token: Optional[str]) -> tuple[bool, str]:
    p = Path(candidate)
    if p.is_dir():
        ok = (p / "adapter_config.json").exists()
        return ok, "local folder contains adapter_config.json" if ok else "local folder does not contain adapter_config.json"
    if "/" not in candidate or candidate.startswith(("/", ".")):
        return False, "neither a local folder nor an HF repo"
    try:
        from huggingface_hub import HfApi

        files = HfApi(token=token).list_repo_files(candidate)
    except Exception as e:
        return False, f"HF repo not found or not accessible ({type(e).__name__})"
    ok = "adapter_config.json" in files and any(f.endswith(".safetensors") or f.endswith(".bin") for f in files)
    return ok, "HF repo contains the adapter" if ok else f"HF repo exists but has no final adapter yet (files: {files[:6]})"


def resolve_adapter(cfg: dict, token: Optional[str] = None) -> str:
    """Decide which adapter to use and make SURE it exists.

    Order: cfg['adapter'] (if set) -> repo <token-user>/<hub_repo_name> on HF -> <work_dir>/adapter.
    """
    candidates: list[str] = []
    if cfg.get("adapter"):
        candidates.append(cfg["adapter"])
    else:
        if token and cfg.get("hub_repo_name"):
            try:
                from huggingface_hub import HfApi

                candidates.append(f"{HfApi().whoami(token=token)['name']}/{cfg['hub_repo_name']}")
            except Exception:
                pass
        candidates.append(str(get_work_dir() / "adapter"))
    problems = []
    for c in candidates:
        ok, why = _adapter_ok(c, token)
        if ok:
            return c
        problems.append(f"- {c}: {why}")
    raise FileNotFoundError(
        "Fine-tuned adapter not found:\n" + "\n".join(problems) +
        "\nRun notebook 02 until the push cell succeeds (it prints 'Adapter pushed to: ...'), or set CONFIG['adapter'] "
        "to the correct HF repo/folder."
    )


def export_results(folder, name: str = "text2sql_results") -> Path:
    """Pack the results folder into a zip; on Colab it is downloaded to your computer, on Kaggle it appears in the Output tab."""
    import shutil

    archive = Path(shutil.make_archive(str(get_work_dir() / name), "zip", str(folder)))
    print("Results archive:", archive)
    if is_colab():
        from google.colab import files  # type: ignore

        files.download(str(archive))
    return archive


def load_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(records: list[dict], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- generasi (butuh GPU)
def load_for_inference(model_name: str, max_seq_length: int = 512, token: Optional[str] = None):
    """Load the base model OR a fine-tuned adapter (HF repo/local folder) for inference."""
    from unsloth import FastLanguageModel  # lazy import: only available on Colab/Kaggle (GPU)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name, max_seq_length=max_seq_length, dtype=None, load_in_4bit=True, token=token
    )
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    return model, tokenizer


def silence_generation_length_warning() -> None:
    """Hide the harmless HF warning 'Both max_new_tokens and max_length seem to have been set'.

    max_new_tokens still applies; other warnings are left alone.
    """
    import logging

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            return not ("max_new_tokens" in msg and "max_length" in msg and "seem to have been set" in msg)

    logger = logging.getLogger("transformers.generation.utils")
    if not any(type(f).__name__ == "_Filter" for f in logger.filters):
        logger.addFilter(_Filter())


def generate_sql_batch(model, tokenizer, items: list[dict], batch_size: int = 16, max_new_tokens: int = 128) -> list[str]:
    """Generate SQL (greedy) for a list of {context, question}. Returns the list of extracted SQL strings."""
    import torch

    silence_generation_length_warning()
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_messages(it["context"], it["question"]), tokenize=False, add_generation_prompt=True
            )
            for it in chunk
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(extract_sql(t) for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


# ---------------------------------------------------------------- metrik
def evaluate_records(records: list[dict]) -> dict:
    """records: dicts with context, question, answer (gold), pred. Filled in place with em/exec/category."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        r["category"] = r.get("category") or complexity_category(r["answer"])
        r["em"] = exact_match(r["pred"], r["answer"])
        res = execution_match(r["context"], r["pred"], r["answer"])
        r["exec_match"], r["gold_ok"], r["gold_nonempty"], r["exec_error"] = (
            res["match"], res["gold_ok"], res["gold_nonempty"], res["error"],
        )
        by_cat[r["category"]].append(r)

    def summarize(rs: list[dict]) -> dict:
        n = len(rs)
        ok = [r for r in rs if r["gold_ok"]]
        nonempty = [r for r in ok if r["gold_nonempty"]]
        return {
            "n": n,
            "exact_match": round(sum(r["em"] for r in rs) / n, 4) if n else None,
            "execution_match": round(sum(r["exec_match"] for r in ok) / len(ok), 4) if ok else None,
            "execution_match_nonempty": round(sum(r["exec_match"] for r in nonempty) / len(nonempty), 4) if nonempty else None,
            "n_gold_executable": len(ok),
            "n_gold_nonempty": len(nonempty),
            "pred_error_rate": round(sum(1 for r in ok if r["exec_error"]) / len(ok), 4) if ok else None,
        }

    metrics = summarize(records)
    metrics["by_category"] = {c: summarize(by_cat[c]) for c in CATEGORIES if by_cat.get(c)}
    return metrics


def write_metrics(metrics: dict, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.1f}%"


def write_results_table(named_metrics: dict[str, dict], path) -> str:
    """Markdown table comparing several models (e.g. baseline vs fine-tuned) from metrics.json."""
    names = list(named_metrics)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for label, key in [
        ("Number of test examples", "n"),
        ("Exact match (normalized)", "exact_match"),
        ("Execution match (synthetic data)", "execution_match"),
        ("Execution match (non-empty gold result)", "execution_match_nonempty"),
        ("Prediction execution error rate", "pred_error_rate"),
    ]:
        vals = [named_metrics[n].get(key) for n in names]
        lines.append(f"| {label} | " + " | ".join(str(v) if key == "n" else _pct(v) for v in vals) + " |")
    lines += ["", "**By complexity category (execution match):**", "",
              "| Category | " + " | ".join(f"{n} (n)" for n in names) + " |", "|---|" + "---|" * len(names)]
    for cat in CATEGORIES:
        cells = []
        for n in names:
            c = named_metrics[n].get("by_category", {}).get(cat)
            cells.append("-" if not c else f"{_pct(c['execution_match'])} ({c['n']})")
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------- error analysis
def diagnose(gold: str, pred: str, error: Optional[str]) -> str:
    """Automatic heuristic (not a manual judgment) for the likely cause of a failure."""
    if error:
        return f"The predicted query failed to run on the schema ({error})."
    g, p = mask_literals(gold).lower(), mask_literals(pred).lower()
    checks = [
        ("join", "The model did not use a required JOIN."),
        ("group by", "The model did not use GROUP BY."),
        ("order by", "The model did not sort the result (ORDER BY)."),
        ("having", "The model missed a HAVING condition."),
        ("where", "The model missed a WHERE condition."),
    ]
    for kw, msg in checks:
        if kw in g and kw not in p:
            return msg
    if len(re.findall(r"\bselect\b", g)) != len(re.findall(r"\bselect\b", p)):
        return "The subquery structure differs from the reference."
    return "Similar structure, but the columns/values/aggregation differ from the reference."


def write_error_analysis(records: list[dict], path, per_category: int = 3, title: str = "Error Analysis") -> None:
    """Write REAL failure examples (wrong execution match) per category to a Markdown file."""
    lines = [f"# {title}", "", "The examples below are taken automatically from real predictions; the 'Likely cause' line is a heuristic, not a manual judgment.", ""]
    for cat in CATEGORIES:
        fails = [r for r in records if r.get("category") == cat and r.get("gold_ok") and not r.get("exec_match")]
        total = sum(1 for r in records if r.get("category") == cat)
        lines += [f"## {cat} ({len(fails)} failed out of {total})", ""]
        if not fails:
            lines += ["_No failures in this category._", ""]
        for r in fails[:per_category]:
            lines += [
                f"**Question:** {r['question']}",
                f"- Schema: `{r['context']}`",
                f"- Reference: `{normalize_sql(r['answer'])}`",
                f"- Prediction: `{normalize_sql(r['pred'])}`",
                f"- Likely cause: {diagnose(r['answer'], r['pred'], r.get('exec_error'))}",
                "",
            ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
