"""From-scratch supervised fine-tuning loop (no HF Trainer).

Implements: prompt formatting, response-only masking, padding/attention masks,
gradient accumulation, fp16 autocast with fp32 master weights + GradScaler,
gradient clipping, warmup + cosine schedule, periodic validation loss, and
best/final checkpointing. Metrics are appended to ``metrics.jsonl`` after every
log/eval step so partial progress survives a Kaggle disconnect.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from alignment.batching import IGNORE_INDEX
from alignment.logprobs import move_batch, response_logps
from alignment.train_common import (
    SFTDataset,
    autocast_context,
    build_optimizer,
    build_scheduler,
    count_trainable,
    load_causal_lm,
    load_tokenizer,
    make_grad_scaler,
    make_sft_collate,
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

SFT_DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-0.5B",
    "train_file": "data/processed/sft_train.jsonl",
    "val_file": "data/processed/sft_val.jsonl",
    "output_dir": "outputs/sft",
    "seed": 42,
    "max_length": 512,
    "per_device_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "eval_batch_size": 8,
    "learning_rate": 2.0e-5,
    "warmup_ratio": 0.03,
    "lr_schedule": "cosine",
    "min_lr_ratio": 0.0,
    "weight_decay": 0.1,
    "adam_betas": [0.9, 0.999],
    "adam_eps": 1.0e-8,
    "optimizer": "adamw",
    "max_grad_norm": 1.0,
    "num_epochs": 1,
    "max_steps": None,
    "limit_train": None,
    "limit_val": None,
    "fp16": True,
    "gradient_checkpointing": True,
    "fast_logprobs": True,
    "log_every": 10,
    "eval_every": 100,
    "eval_at_start": True,
    "device": "auto",
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


@torch.no_grad()
def evaluate_sft(model, loader, device, fp16: bool, fast: bool) -> dict:
    was_training = model.training
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        with autocast_context(device, fp16):
            summed, counts = response_logps(model, batch, fast=fast)
        total_nll += float(-summed.sum())
        total_tokens += int(counts.sum())
    if was_training:
        model.train()
    loss = total_nll / max(total_tokens, 1)
    return {"loss": loss, "ppl": math.exp(min(loss, 50)), "tokens": total_tokens}


def train_sft(user_config: dict) -> dict:
    config = {**SFT_DEFAULTS, **(user_config or {})}
    set_seed(int(config["seed"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config_used.json", config)
    device = resolve_device(config["device"])
    fp16 = bool(config["fp16"]) and device.type == "cuda"
    fast = bool(config["fast_logprobs"])
    print(f"[sft] device={device} ({describe_cuda()}) fp16={fp16}")

    tokenizer = load_tokenizer(config["model_name"])
    train_rows = read_jsonl(config["train_file"])
    val_rows = read_jsonl(config["val_file"])
    if config["limit_train"]:
        train_rows = train_rows[: int(config["limit_train"])]
    if config["limit_val"]:
        val_rows = val_rows[: int(config["limit_val"])]
    max_length = int(config["max_length"])
    train_set = SFTDataset(train_rows, tokenizer, max_length)
    val_set = SFTDataset(val_rows, tokenizer, max_length)
    collate = make_sft_collate(tokenizer, max_length)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=int(config["per_device_batch_size"]), shuffle=True, generator=generator, collate_fn=collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=int(config["eval_batch_size"]), shuffle=False, collate_fn=collate
    )

    model = load_causal_lm(config["model_name"], "float32", device, bool(config["gradient_checkpointing"]))
    model.train()
    trainable, total = count_trainable(model)
    print(f"[sft] params trainable={trainable:,} total={total:,} train={len(train_set)} val={len(val_set)}")

    accum = int(config["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = steps_per_epoch * int(config["num_epochs"])
    if config["max_steps"]:
        total_steps = min(total_steps, int(config["max_steps"]))
    warmup_steps = math.ceil(float(config["warmup_ratio"]) * total_steps)
    optimizer = build_optimizer(
        model,
        config["optimizer"],
        float(config["learning_rate"]),
        float(config["weight_decay"]),
        betas=config["adam_betas"],
        eps=float(config["adam_eps"]),
    )
    scheduler = build_scheduler(optimizer, config["lr_schedule"], total_steps, warmup_steps, float(config["min_lr_ratio"]))
    scaler = make_grad_scaler(device, fp16)
    logger = MetricsLogger(output_dir / "metrics.jsonl")
    print(f"[sft] optimizer steps={total_steps} warmup={warmup_steps} effective batch={accum * int(config['per_device_batch_size'])}")

    best = {"val_loss": float("inf"), "step": None}
    last_eval = {"step": None, "metrics": None}

    def run_eval(step: int):
        metrics = evaluate_sft(model, val_loader, device, fp16, fast)
        logger.log("val", step, loss=metrics["loss"], ppl=metrics["ppl"], tokens=metrics["tokens"])
        print(f"[sft] step {step:5d} | val_loss {metrics['loss']:.4f} | ppl {metrics['ppl']:.2f}")
        if metrics["loss"] < best["val_loss"] and step > 0:
            best.update(val_loss=metrics["loss"], step=step)
            model.config.use_cache = True
            save_checkpoint(model, tokenizer, output_dir / "best", {"step": step, "val_loss": metrics["loss"], "config": config})
            model.config.use_cache = False
            print(f"[sft]   saved new best checkpoint (step {step})")
        last_eval.update(step=step, metrics=metrics)
        return metrics

    if config["eval_at_start"]:
        run_eval(0)

    step = 0
    skipped_steps = 0
    window_loss_sum, window_tokens = 0.0, 0
    log_tokens, log_start = 0, time.time()
    train_start = time.time()
    params = [p for p in model.parameters() if p.requires_grad]
    for epoch, window in _accumulation_windows(train_loader, accum, int(config["num_epochs"])):
        if step >= total_steps:
            break
        total_window_tokens = sum(int(b["labels"][:, 1:].ne(IGNORE_INDEX).sum()) for b in window)
        normalizer = max(total_window_tokens, 1)
        step_nll = 0.0
        for batch in window:
            batch = move_batch(batch, device)
            with autocast_context(device, fp16):
                summed, _ = response_logps(model, batch, fast=fast)
                loss = -summed.sum() / normalizer
            scaler.scale(loss).backward()
            step_nll += float(-summed.detach().sum())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(config["max_grad_norm"]))
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if fp16 and scaler.get_scale() < scale_before:
            skipped_steps += 1  # inf/nan gradients -> GradScaler skipped this update
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        window_loss_sum += step_nll
        window_tokens += total_window_tokens
        log_tokens += sum(int(b["attention_mask"].sum()) for b in window)
        if step % int(config["log_every"]) == 0 or step == total_steps:
            elapsed = time.time() - train_start
            eta = elapsed / step * (total_steps - step)
            train_loss = window_loss_sum / max(window_tokens, 1)
            tok_per_s = log_tokens / max(time.time() - log_start, 1e-6)
            logger.log(
                "train",
                step,
                epoch=epoch,
                loss=train_loss,
                lr=scheduler.get_last_lr()[0],
                grad_norm=float(grad_norm),
                tokens_per_s=round(tok_per_s, 1),
                skipped_steps=skipped_steps,
                max_mem_gib=round(cuda_memory_gib(device), 2),
            )
            print(
                f"[sft] step {step:5d}/{total_steps} | loss {train_loss:.4f} | lr {scheduler.get_last_lr()[0]:.2e} "
                f"| gnorm {float(grad_norm):.2f} | {tok_per_s:.0f} tok/s | eta {format_seconds(eta)}"
            )
            window_loss_sum, window_tokens, log_tokens, log_start = 0.0, 0, 0, time.time()
        if step % int(config["eval_every"]) == 0 or step == total_steps:
            run_eval(step)

    final_metrics = last_eval["metrics"] if last_eval["step"] == step else run_eval(step)
    if config["save_final"]:
        model.config.use_cache = True
        save_checkpoint(model, tokenizer, output_dir / "final", {"step": step, "val_loss": final_metrics["loss"], "config": config})
    summary = {
        "run": "sft",
        "optimizer_steps": step,
        "best_step": best["step"],
        "best_val_loss": best["val_loss"],
        "final_val_loss": final_metrics["loss"],
        "skipped_fp16_steps": skipped_steps,
        "train_time": format_seconds(time.time() - train_start),
        "train_examples": len(train_set),
        "val_examples": len(val_set),
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[sft] done: {summary}")
    return summary
