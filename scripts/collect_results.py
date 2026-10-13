"""Collect every number the report needs into one Markdown file.

Reads the run directories (config_used.json / summary.json), judged files,
generation metadata and generation statistics, and writes tables you can paste
straight into the report (Sections 14.2, 14.4, 14.5).

Example:
    python scripts/collect_results.py --runs_root outputs --output outputs/report_tables.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judging import summarize_judgments, summary_markdown_row  # noqa: E402
from alignment.utils import fix_markdown_tables, read_json, read_jsonl  # noqa: E402


def load(path):
    path = Path(path)
    return read_json(path) if path.exists() else {}


def config_table(root: Path) -> list[str]:
    lines = [
        "| Run | Params trained | Optimizer | Peak LR | Schedule / warmup | Weight decay | Clip | Per-device batch | Grad accum | Effective batch | Precision | Max len | Epochs / steps | Seed | β | Reference |",
        "|---|",
    ]
    sft = load(root / "sft" / "config_used.json")
    sft_sum = load(root / "sft" / "summary.json")
    if sft:
        eff = sft["per_device_batch_size"] * sft["gradient_accumulation_steps"]
        lines.append(
            f"| SFT | full | {sft['optimizer']} (betas {sft.get('adam_betas')}) | {sft['learning_rate']} | {sft['lr_schedule']}, {sft['warmup_ratio']:.0%} warmup "
            f"| {sft['weight_decay']} | {sft['max_grad_norm']} | {sft['per_device_batch_size']} | {sft['gradient_accumulation_steps']} | {eff} "
            f"| fp16 autocast, fp32 weights | {sft['max_length']} | {sft['num_epochs']} / {sft_sum.get('optimizer_steps', '?')} | {sft['seed']} | – | – |"
        )
    for run in ("dpo_hh", "dpo_uf"):
        cfg = load(root / run / "config_used.json")
        summ = load(root / run / "summary.json")
        if not cfg:
            continue
        eff = cfg["per_device_batch_size"] * cfg["gradient_accumulation_steps"]
        lines.append(
            f"| {run} | full | {cfg['optimizer']} | {cfg['learning_rate']} | {cfg['lr_schedule']}, {cfg['warmup_steps']} warmup steps "
            f"| {cfg['weight_decay']} | {cfg['max_grad_norm']} | {cfg['per_device_batch_size']} pair | {cfg['gradient_accumulation_steps']} | {eff} "
            f"| fp16 autocast, fp32 policy; reference {cfg['reference_dtype']} | {cfg['max_length']} (prompt ≤ {cfg['max_prompt_length']}) "
            f"| {cfg['num_epochs']} / {summ.get('optimizer_steps', '?')} | {cfg['seed']} | {cfg['beta']} "
            f"| SFT best, step {summ.get('reference_checkpoint_state', {}).get('step', '?')} (frozen) |"
        )
    lora = load(root / "lora_persona" / "config_used.json")
    lora_sum = load(root / "lora_persona" / "summary.json")
    if lora:
        l, d = lora["lora"], lora["dpo"]
        eff = d["per_device_train_batch_size"] * d["gradient_accumulation_steps"]
        lines.append(
            f"| LoRA-DPO | LoRA r={l['r']}, α={l['lora_alpha']}, dropout={l['lora_dropout']}, {','.join(l['target_modules'])} "
            f"({lora_sum.get('trainable_params', '?'):,} params) | {d.get('optim')} | {d['learning_rate']} | {d['lr_scheduler_type']}, {d['warmup_ratio']:.0%} warmup "
            f"| {d['weight_decay']} | {d['max_grad_norm']} | {d['per_device_train_batch_size']} | {d['gradient_accumulation_steps']} | {eff} "
            f"| fp16 AMP, fp32 weights | {d['max_length']} | {d['num_train_epochs']} / {lora_sum.get('optimizer_steps', '?')} | {lora['seed']} | {d['beta']} "
            f"| base model with adapter disabled |"
        )
    return lines


def training_table(root: Path) -> list[str]:
    lines = ["| Run | Best step | Selection metric (best) | Final validation | Train time | Skipped fp16 steps |", "|---|"]
    sft = load(root / "sft" / "summary.json")
    if sft:
        lines.append(
            f"| SFT | {sft['best_step']} | val loss {sft['best_val_loss']:.4f} | val loss {sft['final_val_loss']:.4f} | {sft['train_time']} | {sft['skipped_fp16_steps']} |"
        )
    for run in ("dpo_hh", "dpo_uf"):
        s = load(root / run / "summary.json")
        if s:
            f = s["final_val"]
            lines.append(
                f"| {run} | {s['best_step']} | reward acc {s['best_val_reward_accuracy']:.3f}, margin {s['best_val_reward_margin']:.4f} "
                f"| loss {f['loss']:.4f}, acc {f['reward_accuracy']:.3f}, margin {f['reward_margin']:.4f} | {s['train_time']} | {s['skipped_fp16_steps']} |"
            )
    lora = load(root / "lora_persona" / "summary.json")
    if lora:
        fe = lora.get("final_eval", {})
        lines.append(
            f"| LoRA-DPO | {lora['optimizer_steps']} (final) | – | loss {fe.get('eval_loss', float('nan')):.4f}, acc {fe.get('eval_rewards/accuracies', 'n/a')}, "
            f"margin {fe.get('eval_rewards/margins', 'n/a')} | – | – |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_root", default="outputs")
    parser.add_argument("--eval_dir", default=None, help="Directory with *_judged.jsonl (default <runs_root>/eval)")
    parser.add_argument("--generations_dir", default=None)
    parser.add_argument("--output", default="outputs/report_tables.md")
    args = parser.parse_args()
    root = Path(args.runs_root)
    eval_dir = Path(args.eval_dir) if args.eval_dir else root / "eval"
    gen_dir = Path(args.generations_dir) if args.generations_dir else root / "generations"

    out = ["# Report tables (auto-generated)\n", "## Experimental configuration (Section 14.2)\n"]
    out += config_table(root)
    out += ["", "## Training outcomes (Section 14.4)\n"]
    out += training_table(root)
    out += [
        "",
        "## Pairwise judgments (Section 14.5)\n",
        "| Comparison | n | Wins | Ties | Losses | Win rate (excl. ties) | Win rate (ties = ½) | Unknown | Failed | Pairs retried | Judge |",
        "|---|",
    ]
    per_category = []
    for path in sorted(eval_dir.glob("*_judged.jsonl")):
        records = read_jsonl(path)
        if not records:
            continue
        summary = summarize_judgments(records)
        summary["judge_model"] = records[0].get("judge_model", "")
        out.append(summary_markdown_row(f"{path.stem.replace('_judged', '')} [{records[0].get('rubric', '')}]", summary))
        if len(summary["per_category"]) > 1:
            per_category.append(f"\n### Per category: {path.stem}\n")
            per_category.append("| Category | Wins | Ties | Losses | Win rate (excl. ties) |")
            per_category.append("|---|")
            for category, r in summary["per_category"].items():
                rate = "n/a" if r["win_rate_excluding_ties"] is None else f"{100 * r['win_rate_excluding_ties']:.1f}%"
                per_category.append(f"| {category} | {r['wins']} | {r['ties']} | {r['losses']} | {rate} |")
    out += per_category
    out += ["", "## Generation settings\n", "| File | Model | Prompt format | Decoding | Temperature / top-p | max_new_tokens | repetition_penalty | length_penalty | no_repeat_ngram | Stop |", "|---|"]
    for meta_path in sorted(gen_dir.glob("*.meta.json")):
        meta = read_json(meta_path)
        dec = meta["decoding"]
        model = (meta["model_path"] + (f" + {meta['adapter_path']}" if meta.get("adapter_path") else "")).replace("|", "\\|")
        out.append(
            f"| {meta_path.name.replace('.meta.json', '')} | {model} | {meta['prompt_format']} | {dec['strategy']} ({dec['dtype']}) "
            f"| n/a (greedy) | {dec['max_new_tokens']} | {dec['repetition_penalty']} | {dec.get('length_penalty') or 'none'} "
            f"| {dec['no_repeat_ngram_size']} | EOS{' + ' + ', '.join(repr(x) for x in dec['stop_strings']) if dec.get('stop_strings') else ' / <\\|im_end\\|> (chat template)'} |"
        )
    text = fix_markdown_tables("\n".join(out) + "\n")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
