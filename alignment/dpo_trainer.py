"""From-scratch DPO training loop (no TRL, no HF Trainer).

* policy  = SFT best checkpoint, trainable, fp32 master weights (+ fp16 autocast)
* reference = a second copy of the SFT best checkpoint, frozen (eval mode,
  ``requires_grad=False``, forward passes under ``torch.no_grad``), placed on
  ``cuda:1`` when two GPUs are visible.

By default the reference is also kept in fp32 and run under the same fp16
autocast as the policy, so at step 0 policy and reference produce *identical*
log-probs (rewards exactly 0, reward accuracy exactly 0, loss exactly ln 2).
Loading the reference in fp16 (``reference_dtype: float16``) is allowed by the
manual and saves memory, but introduces tiny numerical differences that make the
step-0 accuracy noisy instead of exactly 0.

Validation reference log-probs never change, so they are computed once and cached.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from alignment.batching import get_pad_id
from alignment.dpo_core import preference_logps
from alignment.losses import dpo_loss_from_logps, dpo_statistics
from alignment.train_common import (
    PreferenceDataset,
    autocast_context,
    build_optimizer,
    build_scheduler,
    count_trainable,
    load_causal_lm,
    load_tokenizer,
    make_grad_scaler,
    make_preference_collate,
    resolve_device,
)
from alignment.utils import (
    MetricsLogger,
    cuda_memory_gib,
    describe_cuda,
    format_seconds,
    read_jsonl,
    save_checkpoint,
    set_seed,
    write_json,
)

DPO_DEFAULTS = {
    "run_name": "dpo",
    "sft_checkpoint": "outputs/sft/best",
    "train_file": None,
    "val_file": None,
    "output_dir": "outputs/dpo",
    "seed": 42,
    "beta": 0.1,
    "max_length": 512,
    "max_prompt_length": 256,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "eval_batch_size": 4,
    "learning_rate": 1.0e-6,
    "lr_schedule": "constant",
    "warmup_steps": 0,
    "optimizer": "rmsprop",
    "weight_decay": 0.0,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1.0e-8,
    "max_grad_norm": 1.0,
    "num_epochs": 1,
    "max_steps": None,
    "limit_train": None,
    "limit_val": None,
    "fp16": True,
    "gradient_checkpointing": True,
    "fast_logprobs": True,
    "reference_dtype": "float32",
    "policy_device": "auto",
    "reference_device": "auto",
    "log_every": 5,
    "eval_every": 50,
    "eval_at_start": True,
    "save_final": True,
}


def _accumulation_windows(loader, accum: int, num_epochs: int):
    window = []
    for epoch in range(num_epochs):
        for batch in loader:
            window.append(batch)
            if len(window) == accum:
                yield epoch, window
                window = []
    if window:
        yield num_epochs - 1, window


def _reference_logps(reference, batch, policy_device, pad_id, fp16, fast):
    ref_device = next(reference.parameters()).device
    with torch.no_grad():
        with autocast_context(ref_device, fp16):
            chosen, rejected = preference_logps(reference, batch, pad_id, fast)
    return chosen.float().to(policy_device), rejected.float().to(policy_device)


def _checkpoint_state(path) -> dict:
    """Which SFT step/val loss the policy init and frozen reference came from."""

    state_path = Path(path) / "training_state.json"
    if not state_path.exists():
        return {}
    import json

    state = json.loads(state_path.read_text())
    return {key: state[key] for key in ("step", "val_loss") if key in state}


def _mean(values):
    return float(torch.cat(values).mean()) if values else float("nan")


class DPOEvaluator:
    """Validation loss / reward accuracy / reward margin with cached reference log-probs."""

    def __init__(self, loader, reference, policy_device, pad_id, beta, fp16, fast):
        self.loader = loader
        self.policy_device = policy_device
        self.pad_id = pad_id
        self.beta = beta
        self.fp16 = fp16
        self.fast = fast
        self.reference_cache = [
            _reference_logps(reference, batch, policy_device, pad_id, fp16, fast) for batch in loader
        ]

    @torch.no_grad()
    def __call__(self, policy) -> dict:
        was_training = policy.training
        policy.eval()
        collected = {key: [] for key in ("losses", "chosen_rewards", "rejected_rewards", "margins", "accuracies")}
        for batch, (ref_c, ref_r) in zip(self.loader, self.reference_cache):
            with autocast_context(self.policy_device, self.fp16):
                pol_c, pol_r = preference_logps(policy, batch, self.pad_id, self.fast)
            stats = dpo_statistics(pol_c, pol_r, ref_c, ref_r, self.beta)
            for key in collected:
                collected[key].append(stats[key])
        if was_training:
            policy.train()
        return {
            "loss": _mean(collected["losses"]),
            "reward_accuracy": _mean(collected["accuracies"]),
            "reward_margin": _mean(collected["margins"]),
            "chosen_reward": _mean(collected["chosen_rewards"]),
            "rejected_reward": _mean(collected["rejected_rewards"]),
        }


def train_dpo(user_config: dict) -> dict:
    config = {**DPO_DEFAULTS, **(user_config or {})}
    run = config["run_name"]
    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_used.json", config)

    policy_device = resolve_device(config["policy_device"], fallback_index=0)
    reference_device = resolve_device(config["reference_device"], fallback_index=1)
    fp16 = bool(config["fp16"]) and policy_device.type == "cuda"
    fast = bool(config["fast_logprobs"])
    beta = float(config["beta"])
    print(f"[{run}] policy={policy_device} reference={reference_device} ({describe_cuda()}) fp16={fp16} beta={beta}")

    tokenizer = load_tokenizer(config["sft_checkpoint"])
    pad_id = get_pad_id(tokenizer)
    train_rows = read_jsonl(config["train_file"])
    val_rows = read_jsonl(config["val_file"])
    if config["limit_train"]:
        train_rows = train_rows[: int(config["limit_train"])]
    if config["limit_val"]:
        val_rows = val_rows[: int(config["limit_val"])]
    max_length, max_prompt_length = int(config["max_length"]), int(config["max_prompt_length"])
    train_set = PreferenceDataset(train_rows, tokenizer, max_length, max_prompt_length)
    val_set = PreferenceDataset(val_rows, tokenizer, max_length, max_prompt_length)
    collate = make_preference_collate(tokenizer, max_length)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=int(config["per_device_batch_size"]), shuffle=True, generator=generator, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=int(config["eval_batch_size"]), shuffle=False, collate_fn=collate)

    policy = load_causal_lm(config["sft_checkpoint"], "float32", policy_device, bool(config["gradient_checkpointing"]))
    policy.train()
    reference = load_causal_lm(config["sft_checkpoint"], config["reference_dtype"], reference_device, False)
    reference.eval()
    reference.requires_grad_(False)
    trainable, total = count_trainable(policy)
    print(f"[{run}] policy trainable={trainable:,}/{total:,} | train pairs={len(train_set)} val pairs={len(val_set)}")

    accum = int(config["gradient_accumulation_steps"])
    total_steps = math.ceil(len(train_loader) / accum) * int(config["num_epochs"])
    if config["max_steps"]:
        total_steps = min(total_steps, int(config["max_steps"]))
    optimizer = build_optimizer(
        policy,
        config["optimizer"],
        float(config["learning_rate"]),
        float(config["weight_decay"]),
        betas=config["adam_betas"],
        eps=float(config["adam_eps"]),
    )
    scheduler = build_scheduler(optimizer, config["lr_schedule"], total_steps, int(config["warmup_steps"]))
    scaler = make_grad_scaler(policy_device, fp16)
    logger = MetricsLogger(output_dir / "metrics.jsonl")
    print(f"[{run}] optimizer steps={total_steps} effective batch={accum * int(config['per_device_batch_size'])} pairs")

    print(f"[{run}] caching reference log-probs for {len(val_set)} validation pairs ...")
    evaluator = DPOEvaluator(val_loader, reference, policy_device, pad_id, beta, fp16, fast)

    best = {"reward_accuracy": -1.0, "reward_margin": -float("inf"), "step": None}
    last_eval = {"step": None, "metrics": None}

    def run_eval(step: int) -> dict:
        metrics = evaluator(policy)
        logger.log("val", step, **metrics)
        print(
            f"[{run}] step {step:4d} | val loss {metrics['loss']:.4f} | acc {metrics['reward_accuracy']:.3f} "
            f"| margin {metrics['reward_margin']:.4f} | r_w {metrics['chosen_reward']:.4f} r_l {metrics['rejected_reward']:.4f}"
        )
        better = (metrics["reward_accuracy"], metrics["reward_margin"]) > (best["reward_accuracy"], best["reward_margin"])
        if step > 0 and better:
            best.update(reward_accuracy=metrics["reward_accuracy"], reward_margin=metrics["reward_margin"], step=step)
            policy.config.use_cache = True
            save_checkpoint(policy, tokenizer, output_dir / "best", {"step": step, **metrics, "config": config})
            policy.config.use_cache = False
            print(f"[{run}]   saved new best checkpoint (step {step})")
        last_eval.update(step=step, metrics=metrics)
        return metrics

    if config["eval_at_start"]:
        run_eval(0)

    step, skipped_steps = 0, 0
    train_start = time.time()
    params = [p for p in policy.parameters() if p.requires_grad]
    log_buffer = {key: [] for key in ("losses", "chosen_rewards", "rejected_rewards", "margins", "accuracies", "policy_chosen_logps", "policy_rejected_logps")}
    for epoch, window in _accumulation_windows(train_loader, accum, int(config["num_epochs"])):
        if step >= total_steps:
            break
        for batch in window:
            ref_c, ref_r = _reference_logps(reference, batch, policy_device, pad_id, fp16, fast)
            with autocast_context(policy_device, fp16):
                pol_c, pol_r = preference_logps(policy, batch, pad_id, fast)
            loss = dpo_loss_from_logps(pol_c, pol_r, ref_c, ref_r, beta) / len(window)
            scaler.scale(loss).backward()
            stats = dpo_statistics(pol_c, pol_r, ref_c, ref_r, beta)
            for key in log_buffer:
                log_buffer[key].append(stats[key])

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(config["max_grad_norm"]))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if fp16 and scaler.get_scale() < scale_before:
            skipped_steps += 1
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % int(config["log_every"]) == 0 or step == total_steps:
            elapsed = time.time() - train_start
            row = {
                "epoch": epoch,
                "loss": _mean(log_buffer["losses"]),
                "reward_accuracy": _mean(log_buffer["accuracies"]),
                "reward_margin": _mean(log_buffer["margins"]),
                "chosen_reward": _mean(log_buffer["chosen_rewards"]),
                "rejected_reward": _mean(log_buffer["rejected_rewards"]),
                "policy_chosen_logp": _mean(log_buffer["policy_chosen_logps"]),
                "policy_rejected_logp": _mean(log_buffer["policy_rejected_logps"]),
                "lr": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "skipped_steps": skipped_steps,
                "max_mem_gib": round(cuda_memory_gib(policy_device), 2),
            }
            logger.log("train", step, **row)
            print(
                f"[{run}] step {step:4d}/{total_steps} | loss {row['loss']:.4f} | acc {row['reward_accuracy']:.3f} "
                f"| margin {row['reward_margin']:.4f} | gnorm {row['grad_norm']:.2f} | eta {format_seconds(elapsed / step * (total_steps - step))}"
            )
            log_buffer = {key: [] for key in log_buffer}
        if step % int(config["eval_every"]) == 0 or step == total_steps:
            run_eval(step)

    final_metrics = last_eval["metrics"] if last_eval["step"] == step else run_eval(step)
    if config["save_final"]:
        policy.config.use_cache = True
        save_checkpoint(policy, tokenizer, output_dir / "final", {"step": step, **final_metrics, "config": config})
    summary = {
        "run": run,
        "optimizer_steps": step,
        "best_step": best["step"],
        "best_val_reward_accuracy": best["reward_accuracy"],
        "best_val_reward_margin": best["reward_margin"],
        "final_val": final_metrics,
        "skipped_fp16_steps": skipped_steps,
        "train_time": format_seconds(time.time() - train_start),
        "train_pairs": len(train_set),
        "val_pairs": len(val_set),
        "reference_checkpoint": str(config["sft_checkpoint"]),
        "reference_checkpoint_state": _checkpoint_state(config["sft_checkpoint"]),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[{run}] done: {summary}")
    return summary
