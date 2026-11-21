"""Stage 2: QLoRA fine-tuning with Unsloth. Needs a GPU (Colab/Kaggle).

Run:  python train_qlora.py     (HF_TOKEN from an env var / Colab Secrets; set hf_username in CONFIG)
Note: the TRL/Unsloth APIs change often between versions; the functions here filter arguments to match the installed version.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_utils import get_hf_token, get_work_dir, load_jsonl  # noqa: E402
from prompt_utils import build_messages  # noqa: E402

CONFIG = {
    # --- model & LoRA
    "model_name": "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",   # verified to exist on the HF Hub
    "max_seq_length": 512,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # --- optimization (effective batch = batch_size * grad_accum = 16)
    "batch_size": 4,
    "grad_accum": 4,
    "epochs": 1,
    "learning_rate": 2e-4,
    "lr_scheduler": "cosine",
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "seed": 42,
    "save_steps": 100,        # checkpoint to the HF Hub every 100 steps
    # --- publishing
    "hf_username": "",        # <-- set your Hugging Face username (or leave empty to take it from the token)
    "hub_repo_name": "qwen2.5-coder-3b-text2sql-qlora",
}


def hub_model_id(cfg: dict) -> str:
    return f"{cfg['hf_username']}/{cfg['hub_repo_name']}" if cfg.get("hf_username") else ""


def build_model(cfg: dict):
    from unsloth import FastLanguageModel  # must be imported before transformers/trl

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"], max_seq_length=cfg["max_seq_length"], dtype=None, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=cfg["target_modules"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg["seed"],
    )
    return model, tokenizer


def to_text_dataset(records: list[dict], tokenizer, max_seq_length: int):
    """Turn records into a 'text' column (chat template). Examples longer than max_seq_length are dropped (not truncated)."""
    from datasets import Dataset

    texts, dropped = [], 0
    for r in records:
        text = tokenizer.apply_chat_template(
            build_messages(r["context"], r["question"], r["answer"]), tokenize=False, add_generation_prompt=False
        )
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > max_seq_length:
            dropped += 1
            continue
        texts.append(text)
    print(f"{len(texts)} examples used, {dropped} dropped because they exceed {max_seq_length} tokens")
    return Dataset.from_dict({"text": texts})


def _accepted_params(cls):
    """Argument names accepted by the class; None if they cannot be determined (e.g. __init__ takes only **kwargs)."""
    import dataclasses

    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls) if f.init}
    params = inspect.signature(cls.__init__).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return None
    return set(params)


def _filter_kwargs(cls, kwargs: dict) -> dict:
    """Adapt argument names to the installed library version (e.g. max_seq_length -> max_length)."""
    accepted = _accepted_params(cls)
    if accepted is None:          # cannot be inspected -> pass through unchanged (do not drop important arguments)
        return dict(kwargs)
    aliases = {"max_seq_length": "max_length", "eval_strategy": "evaluation_strategy"}
    reverse = {v: k for k, v in aliases.items()}
    out = {}
    for k, v in kwargs.items():
        if k in accepted:
            out[k] = v
        elif aliases.get(k) in accepted:
            out[aliases[k]] = v
        elif reverse.get(k) in accepted:
            out[reverse[k]] = v
        else:
            print(f"[info] argument '{k}' is not recognized by this version, skipped")
    return out


def build_trainer(model, tokenizer, train_ds, val_ds, cfg: dict, output_dir: str):
    from trl import SFTConfig, SFTTrainer
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

    hub_id, token = hub_model_id(cfg), get_hf_token()
    push = bool(hub_id and token)
    if not push:
        print("[warning] hf_username/HF_TOKEN is empty -> checkpoints are NOT pushed to the HF Hub (local only).")
    args = SFTConfig(**_filter_kwargs(SFTConfig, dict(
        output_dir=output_dir,
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=cfg["save_steps"],
        save_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        report_to="none",
        seed=cfg["seed"],
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_length"],
        packing=False,
        dataset_num_proc=2,
        push_to_hub=push,
        hub_model_id=hub_id if push else None,
        hub_strategy="every_save",
        hub_token=token if push else None,
    )))
    trainer_kwargs = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=args)
    accepted = _accepted_params(SFTTrainer)
    key = "processing_class" if (accepted is not None and "processing_class" in accepted) else "tokenizer"
    trainer = SFTTrainer(**trainer_kwargs, **{key: tokenizer})
    try:  # the loss is computed on the SQL answer only (not the prompt)
        trainer = train_on_responses_only(
            trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n"
        )
    except Exception as exc:
        print(f"[warning] train_on_responses_only failed ({exc}); the loss is computed on all tokens.")
    return trainer


def save_log_and_plot(trainer, out_dir: Path) -> dict:
    """Save the training log (JSON) and the loss curve (PNG) from trainer.state.log_history."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    log = trainer.state.log_history
    (out_dir / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    tr = [(x["step"], x["loss"]) for x in log if "loss" in x]
    ev = [(x["step"], x["eval_loss"]) for x in log if "eval_loss" in x]
    fig, ax = plt.subplots(figsize=(7, 4))
    if tr:
        ax.plot(*zip(*tr), label="train loss")
    if ev:
        ax.plot(*zip(*ev), marker="o", label="validation loss")
    ax.set(xlabel="langkah (step)", ylabel="loss", title="Kurva loss QLoRA Text-to-SQL")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close(fig)
    return {"final_train_loss": tr[-1][1] if tr else None, "final_eval_loss": ev[-1][1] if ev else None, "steps": log[-1].get("step") if log else None}


def dump_environment(path: Path) -> None:
    """Record the installed library versions (basis for a pinned requirements.txt)."""
    from importlib import metadata

    lines = []
    for pkg in ["unsloth", "transformers", "trl", "peft", "torch", "datasets", "bitsandbytes", "accelerate", "gradio", "matplotlib"]:
        try:
            lines.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(f"# {pkg} is not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: dict = CONFIG) -> None:
    work = get_work_dir()
    data_dir, res_dir = work / "data", work / "results"
    train, val = load_jsonl(data_dir / "train.jsonl"), load_jsonl(data_dir / "val.jsonl")
    model, tokenizer = build_model(cfg)
    train_ds = to_text_dataset(train, tokenizer, cfg["max_seq_length"])
    val_ds = to_text_dataset(val, tokenizer, cfg["max_seq_length"])
    trainer = build_trainer(model, tokenizer, train_ds, val_ds, cfg, str(work / "checkpoints"))
    trainer.train()
    summary = save_log_and_plot(trainer, res_dir)
    adapter_dir = work / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    hub_id, token = hub_model_id(cfg), get_hf_token()
    if hub_id and token:
        model.push_to_hub(hub_id, token=token)
        tokenizer.push_to_hub(hub_id, token=token)
    dump_environment(res_dir / "environment.txt")
    (res_dir / "train_summary.json").write_text(json.dumps({**summary, "hub_model_id": hub_id, "config": cfg}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
