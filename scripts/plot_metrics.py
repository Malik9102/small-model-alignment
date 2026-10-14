"""Training plots from metrics.jsonl files (SFT, from-scratch DPO, LoRA-DPO).

Example:
    python scripts/plot_metrics.py --sft outputs/sft --dpo outputs/dpo_hh outputs/dpo_uf \
        --lora outputs/lora_persona --output_dir outputs/plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alignment.utils import read_json, read_jsonl  # noqa: E402


def ema(values, alpha: float = 0.2):
    smoothed, current = [], None
    for value in values:
        current = value if current is None else alpha * value + (1 - alpha) * current
        smoothed.append(current)
    return smoothed


def series(rows, split, key):
    points = [(r["step"], r[key]) for r in rows if r.get("split") == split and isinstance(r.get(key), (int, float))]
    return [p[0] for p in points], [p[1] for p in points]


def run_info(run_dir) -> dict:
    run_dir = Path(run_dir)
    info = {}
    for name in ("config_used.json", "summary.json"):
        if (run_dir / name).exists():
            info.update(read_json(run_dir / name))
    return info


def best_step(run_dir):
    summary = Path(run_dir) / "summary.json"
    return read_json(summary).get("best_step") if summary.exists() else None


def plot_sft(run_dir, output_dir):
    rows = read_jsonl(Path(run_dir) / "metrics.jsonl")
    fig, ax = plt.subplots(figsize=(7, 4))
    steps, loss = series(rows, "train", "loss")
    if steps:
        ax.plot(steps, loss, alpha=0.3, color="tab:blue", label="train loss (raw, per log window)")
        ax.plot(steps, ema(loss), color="tab:blue", label="train loss (EMA, α=0.2)")
    steps, loss = series(rows, "val", "loss")
    if steps:
        n_val = run_info(run_dir).get("val_examples", "?")
        ax.plot(steps, loss, "o-", color="tab:orange", label=f"validation loss ({n_val} examples)")
    best = best_step(run_dir)
    if best:
        ax.axvline(best, ls="--", color="gray", label=f"best checkpoint (step {best})")
    ax.set_xlabel("optimizer step (effective batch 32)")
    ax.set_ylabel("response-token NLL (nats/token)")
    info = run_info(run_dir)
    ax.set_title(f"SFT: {info.get('model_name', '')} on {info.get('train_examples', '?')} examples (lr {info.get('learning_rate', '?')})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(output_dir) / "sft_loss.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_dpo(run_dir, output_dir):
    run_dir = Path(run_dir)
    rows = read_jsonl(run_dir / "metrics.jsonl")
    name = run_dir.name
    best = best_step(run_dir)
    info = run_info(run_dir)
    n_val = info.get("val_pairs", "?")
    batch = int(info.get("per_device_batch_size", 1)) * int(info.get("gradient_accumulation_steps", 32))
    panels = [
        ("loss", "DPO loss", 0.6931),
        ("reward_accuracy", "reward accuracy (r_w > r_l)", None),
        ("reward_margin", "reward margin (r_w − r_l)", 0.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, (key, label, reference) in zip(axes.flat, panels):
        steps, values = series(rows, "train", key)
        if steps:
            ax.plot(steps, values, alpha=0.3, color="tab:blue", label="train (raw)")
            ax.plot(steps, ema(values), color="tab:blue", label="train (EMA, α=0.2)")
        steps, values = series(rows, "val", key)
        if steps:
            ax.plot(steps, values, "o-", color="tab:orange", label=f"validation ({n_val} pairs)")
        if reference is not None:
            ax.axhline(reference, ls=":", color="black", lw=0.8)
        if best:
            ax.axvline(best, ls="--", color="gray", lw=0.8, label=f"best (step {best})")
        ax.set_xlabel(f"optimizer step ({batch} pairs)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    ax = axes.flat[3]
    for key, color, label in (("chosen_reward", "tab:green", "chosen r_w"), ("rejected_reward", "tab:red", "rejected r_l")):
        steps, values = series(rows, "val", key)
        if steps:
            ax.plot(steps, values, "o-", color=color, label=f"validation {label}")
    ax.axhline(0, ls=":", color="black", lw=0.8)
    ax.set_xlabel(f"optimizer step ({batch} pairs)")
    ax.set_ylabel("implicit reward β·log(π/π_ref)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.suptitle(
        f"From-scratch DPO: {info.get('run_name', name)} (β={info.get('beta', '?')}, lr {info.get('learning_rate', '?')}, "
        f"{info.get('optimizer', '?')}, {info.get('train_pairs', '?')} train pairs)"
    )
    fig.tight_layout()
    path = Path(output_dir) / f"{name}_training.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_lora(run_dir, output_dir):
    rows = read_jsonl(Path(run_dir) / "metrics.jsonl")
    info = run_info(run_dir)
    dpo_cfg = info.get("dpo", {})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    specs = [
        ("loss", "eval_loss", "DPO loss"),
        ("rewards/accuracies", "eval_rewards/accuracies", "reward accuracy"),
        ("rewards/margins", "eval_rewards/margins", "reward margin"),
    ]
    for ax, (train_key, eval_key, label) in zip(axes, specs):
        steps, values = series(rows, "train", train_key)
        if steps:
            ax.plot(steps, values, "-", color="tab:blue", label="train")
        steps, values = series(rows, "val", eval_key)
        if steps:
            ax.plot(steps, values, "o-", color="tab:orange", label="held-out 20 pairs")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"LoRA-DPO persona tuning: {Path(run_dir).name} (SmolLM2-360M-Instruct, β={dpo_cfg.get('beta', '?')}, "
        f"lr {dpo_cfg.get('learning_rate', '?')}, {dpo_cfg.get('num_train_epochs', '?')} epochs)"
    )
    fig.tight_layout()
    path = Path(output_dir) / f"{Path(run_dir).name}_training.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_overlay(run_dirs, output_dir, kind: str):
    """Validation curves of several runs on shared axes (guide config vs ablations)."""

    if kind == "dpo":
        specs = [("loss", "validation DPO loss"), ("reward_accuracy", "validation reward accuracy"), ("reward_margin", "validation reward margin")]
        step_label = "optimizer step (32 pairs)"
    else:
        specs = [("eval_loss", "held-out DPO loss"), ("eval_rewards/accuracies", "held-out reward accuracy"), ("eval_rewards/margins", "held-out reward margin")]
        step_label = "optimizer step"
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for run_dir in run_dirs:
        rows = read_jsonl(Path(run_dir) / "metrics.jsonl")
        info = run_info(run_dir)
        if kind == "dpo":
            label = f"{info.get('run_name', Path(run_dir).name)} (lr {info.get('learning_rate', '?')})"
        else:
            cfg = info.get("dpo", {})
            label = f"{Path(run_dir).name} (β={cfg.get('beta', '?')}, {cfg.get('num_train_epochs', '?')} ep)"
        for ax, (key, _) in zip(axes, specs):
            steps, values = series(rows, "val", key)
            if steps:
                ax.plot(steps, values, "o-", ms=3, label=label)
    for ax, (_, title) in zip(axes, specs):
        ax.set_xlabel(step_label)
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Guide configuration vs ablation" + (" (from-scratch DPO, 1,000 val pairs)" if kind == "dpo" else " (LoRA-DPO, 20 held-out pairs)"))
    fig.tight_layout()
    path = Path(output_dir) / f"{kind}_ablation_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft", default=None)
    parser.add_argument("--dpo", nargs="*", default=[])
    parser.add_argument("--lora", nargs="*", default=[])
    parser.add_argument("--overlay_dpo", nargs="*", default=[], help="DPO run dirs to overlay (e.g. guide + ablation)")
    parser.add_argument("--overlay_lora", nargs="*", default=[], help="LoRA run dirs to overlay")
    parser.add_argument("--output_dir", default="outputs/plots")
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    made = []
    if args.sft:
        made.append(plot_sft(args.sft, args.output_dir))
    for run in args.dpo:
        made.append(plot_dpo(run, args.output_dir))
    for run in args.lora:
        made.append(plot_lora(run, args.output_dir))
    if len(args.overlay_dpo) > 1:
        made.append(plot_overlay(args.overlay_dpo, args.output_dir, "dpo"))
    if len(args.overlay_lora) > 1:
        made.append(plot_overlay(args.overlay_lora, args.output_dir, "lora"))
    for path in made:
        print(f"[plot] {path}")


if __name__ == "__main__":
    main()
