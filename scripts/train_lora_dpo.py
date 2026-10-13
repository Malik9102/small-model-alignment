"""Task 5 ("Idiot Sandwich"): TRL DPOTrainer + PEFT/LoRA persona tuning.

Model:   HuggingFaceTB/SmolLM2-360M-Instruct
Data:    data/persona_lora/ramsay_style_preferences.json (prompt / chosen / rejected)

Rows are converted to TRL's *conversational* preference format:
    {"prompt":   [{"role": "user", "content": ...}],
     "chosen":   [{"role": "assistant", "content": ...}],
     "rejected": [{"role": "assistant", "content": ...}]}
so DPOTrainer applies the model's own chat template.

With ``peft_config`` set and ``ref_model=None``, TRL uses the *same* base model
with the LoRA adapter disabled as the frozen reference, so no second model copy
is loaded.

Example:
    python scripts/train_lora_dpo.py --config configs/lora_persona.yaml
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import sys
from pathlib import Path

# DPOTrainer would wrap the model in nn.DataParallel if it saw both T4s. One GPU
# is plenty for a 360M model + LoRA, and avoids DataParallel quirks.
if "--all_gpus" not in sys.argv:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.utils import (  # noqa: E402
    add_common_args,
    count_lines,
    load_config,
    read_json,
    set_seed,
    write_json,
    write_jsonl,
)

DEFAULTS = {
    "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "data_file": "data/persona_lora/ramsay_style_preferences.json",
    "output_dir": "outputs/lora_persona",
    "seed": 42,
    "val_size": 20,
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
    },
    "dpo": {
        "beta": 0.1,
        "learning_rate": 5.0e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 3,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "max_length": 512,
        "logging_steps": 5,
        "eval_strategy": "steps",
        "eval_steps": 10,
        "save_strategy": "no",
        "report_to": "none",
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
    },
}


def to_conversational(row: dict) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["prompt"].strip()}],
        "chosen": [{"role": "assistant", "content": row["chosen"].strip()}],
        "rejected": [{"role": "assistant", "content": row["rejected"].strip()}],
    }


def split_rows(rows, val_size: int, seed: int):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_idx = set(indices[:val_size])
    train = [rows[i] for i in indices if i not in val_idx]
    val = [rows[i] for i in sorted(val_idx)]
    return train, val


def build_dpo_config(dpo_kwargs: dict, output_dir: str, seed: int, use_fp16: bool):
    from trl import DPOConfig

    accepted = set(inspect.signature(DPOConfig.__init__).parameters)
    # T4 GPUs have no bf16 support, and recent TRL versions default to bf16=True.
    kwargs = {"output_dir": output_dir, "seed": seed, "fp16": use_fp16, "bf16": False, **dpo_kwargs}
    if "eval_strategy" not in accepted and "eval_strategy" in kwargs:  # older transformers
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    dropped = {key: value for key, value in kwargs.items() if key not in accepted}
    if dropped:
        print(f"[lora] DPOConfig in this TRL version ignores: {dropped}")
    return DPOConfig(**{key: value for key, value in kwargs.items() if key in accepted})


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--all_gpus", action="store_true", help="Do not pin training to GPU 0")
    args = parser.parse_args()
    user = load_config(args.config, args.overrides)
    config = {**DEFAULTS, **user}
    config["lora"] = {**DEFAULTS["lora"], **user.get("lora", {})}
    config["dpo"] = {**DEFAULTS["dpo"], **user.get("dpo", {})}

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOTrainer

    from alignment.train_common import dtype_kwarg

    seed = int(config["seed"])
    set_seed(seed)
    output_dir = Path(config["output_dir"])
    if (output_dir / "summary.json").exists() and not config.get("overwrite", False):
        print(f"[lora] {output_dir / 'summary.json'} exists - run already complete, skipping (set overwrite=true to retrain)")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_used.json", config)

    rows = read_json(config["data_file"])
    train_rows, val_rows = split_rows(rows, int(config["val_size"]), seed)
    write_jsonl(output_dir / "train_split.jsonl", train_rows)
    write_jsonl(output_dir / "val_split.jsonl", val_rows)
    train_ds = Dataset.from_list([to_conversational(row) for row in train_rows])
    val_ds = Dataset.from_list([to_conversational(row) for row in val_rows])
    print(f"[lora] {len(train_rows)} train / {len(val_rows)} val preference pairs")

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model_name"], **{dtype_kwarg(): torch.float32})
    use_fp16 = torch.cuda.is_available()  # T4: fp16 autocast, LoRA weights stay fp32

    lora_config = LoraConfig(task_type="CAUSAL_LM", **config["lora"])
    dpo_args = build_dpo_config(config["dpo"], str(output_dir / "trainer"), seed, use_fp16)

    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
    )
    from transformers import TrainerCallback

    class LogCollector(TrainerCallback):
        """Keeps every log dict (trainer.train() resets log_history, losing the pre-training eval)."""

        def __init__(self):
            self.rows = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                split = "val" if any(key.startswith("eval_") for key in logs) else "train"
                self.rows.append({"split": split, "step": state.global_step, **logs})

    class BestAdapterSaver(TrainerCallback):
        """Keep the adapter with the lowest held-out DPO loss (manual 5.5: best + final)."""

        def __init__(self, directory):
            self.directory = directory
            self.best = float("inf")
            self.best_step = None

        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            loss = (metrics or {}).get("eval_loss")
            if loss is None or model is None or state.global_step == 0 or loss >= self.best:
                return
            self.best, self.best_step = loss, state.global_step
            model.save_pretrained(self.directory)
            tokenizer.save_pretrained(self.directory)

    collector = LogCollector()
    best_saver = BestAdapterSaver(str(output_dir / "adapter_best"))
    trainer_kwargs["callbacks"] = [collector, best_saver]
    trainer_params = inspect.signature(DPOTrainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"[lora] trainable params {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    initial_eval = trainer.evaluate()
    print(f"[lora] eval before training: {initial_eval}")
    trainer.train()
    final_eval = trainer.evaluate()
    print(f"[lora] eval after training: {final_eval}")

    adapter_dir = output_dir / "adapter"  # final adapter (used for generation)
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # TRL attaches reward metrics to the logged dicts (evaluate() may only return eval_loss).
    metrics_rows = collector.rows
    write_jsonl(output_dir / "metrics.jsonl", metrics_rows)
    eval_rows = [row for row in metrics_rows if row["split"] == "val"]
    if eval_rows:
        initial_eval = {**initial_eval, **{k: v for k, v in eval_rows[0].items() if k.startswith("eval_")}}
        final_eval = {**final_eval, **{k: v for k, v in eval_rows[-1].items() if k.startswith("eval_")}}
    summary = {
        "run": "lora_persona_dpo",
        "model": config["model_name"],
        "trainable_params": trainable,
        "total_params": total,
        "train_pairs": len(train_rows),
        "val_pairs": len(val_rows),
        "optimizer_steps": trainer.state.global_step,
        "initial_eval": initial_eval,
        "final_eval": final_eval,
        "lora": config["lora"],
        "dpo": config["dpo"],
        "fp16": use_fp16,
        "best_adapter": {"path": "adapter_best", "step": best_saver.best_step, "eval_loss": best_saver.best},
        "final_adapter": "adapter",
        "metrics_rows": count_lines(output_dir / "metrics.jsonl"),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[lora] adapter saved -> {adapter_dir}")


if __name__ == "__main__":
    main()
