"""Build every fixed data file used by the assignment (run once, on Kaggle).

Outputs (default ``data/processed``):
    sft_train.jsonl (30k), sft_val.jsonl (1k)
    hh_train.jsonl (15k),  hh_val.jsonl (1k)
    uf_train.jsonl (15k),  uf_val.jsonl (1k)
    manifest.json          counts, filter statistics, seeds, sources
and ``eval_sets/alpaca_eval_300_seed42.json``.

Example:
    python scripts/prepare_data.py --output_dir data/processed
    python scripts/prepare_data.py --only alpaca      # just the eval subset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.data import (  # noqa: E402
    load_alpaca_eval_rows,
    make_alpaca_eval_subset,
    normalize_text,
    parse_hh_pair,
    parse_ultrafeedback_pair,
    sample_preference_examples,
    sample_sft_examples,
)
from alignment.utils import read_json, write_json, write_jsonl  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "data" / "processed"))
    parser.add_argument("--eval_dir", default=str(REPO_ROOT / "eval_sets"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    parser.add_argument("--sft_train", type=int, default=30000)
    parser.add_argument("--sft_val", type=int, default=1000)
    parser.add_argument("--dpo_train", type=int, default=15000)
    parser.add_argument("--dpo_val", type=int, default=1000)
    parser.add_argument("--alpaca_n", type=int, default=300)
    parser.add_argument("--alpaca_seed", type=int, default=42)
    parser.add_argument(
        "--only", nargs="*", default=["alpaca", "sft", "hh", "uf"], choices=["alpaca", "sft", "hh", "uf"]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from datasets import load_dataset
    from transformers import AutoTokenizer

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = Path(args.eval_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    manifest.update(
        {
            "seed": args.seed,
            "tokenizer": args.tokenizer,
            "max_length": args.max_length,
            "max_prompt_tokens": args.max_prompt_tokens,
            "prompt_template": "### Instruction:\\n{instruction}\\n\\n### Response:\\n",
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # ------------------------------------------------------------------ eval prompts (contamination guard)
    blocked: set[str] = set()
    hh_eval_path = eval_dir / "hh_rlhf_eval_300_seed42.json"
    if hh_eval_path.exists():
        blocked |= {normalize_text(row["instruction"]) for row in read_json(hh_eval_path)}
    alpaca_path = eval_dir / f"alpaca_eval_{args.alpaca_n}_seed{args.alpaca_seed}.json"

    if "alpaca" in args.only:
        rows = load_alpaca_eval_rows()
        subset = make_alpaca_eval_subset(rows, n=args.alpaca_n, seed=args.alpaca_seed)
        write_json(alpaca_path, subset)
        manifest["alpaca_eval"] = {"path": str(alpaca_path.name), "pool_size": len(rows), "n": len(subset), "seed": args.alpaca_seed}
        print(f"[alpaca] wrote {len(subset)} prompts -> {alpaca_path}")
    if alpaca_path.exists():
        blocked |= {normalize_text(row["instruction"]) for row in read_json(alpaca_path)}
    persona_eval_path = eval_dir / "persona_style_eval.json"
    if persona_eval_path.exists():
        blocked |= {normalize_text(row["instruction"]) for row in read_json(persona_eval_path)}

    common = dict(
        tokenizer=tokenizer,
        seed=args.seed,
        max_length=args.max_length,
        max_prompt_tokens=args.max_prompt_tokens,
        blocked_instructions=blocked,
    )

    # ------------------------------------------------------------------ SFT
    if "sft" in args.only:
        start = time.time()
        train_rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
        val_rows = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")
        train, train_stats = sample_sft_examples(train_rows, n=args.sft_train, id_prefix="sft-train", source="ultrachat_200k/train_sft", **common)
        val, val_stats = sample_sft_examples(val_rows, n=args.sft_val, id_prefix="sft-val", source="ultrachat_200k/test_sft", **common)
        write_jsonl(output_dir / "sft_train.jsonl", train)
        write_jsonl(output_dir / "sft_val.jsonl", val)
        manifest["sft"] = {"train": train_stats, "val": val_stats, "dataset": "HuggingFaceH4/ultrachat_200k"}
        print(f"[sft] train={len(train)} val={len(val)} ({time.time() - start:.0f}s) {train_stats}")

    # ------------------------------------------------------------------ HH-RLHF
    if "hh" in args.only:
        start = time.time()
        hh_parser = lambda row: parse_hh_pair(row["chosen"], row["rejected"])  # noqa: E731
        train_rows = load_dataset("Anthropic/hh-rlhf", split="train")
        val_rows = load_dataset("Anthropic/hh-rlhf", split="test")
        train, train_stats = sample_preference_examples(train_rows, hh_parser, n=args.dpo_train, id_prefix="hh-train", source="Anthropic/hh-rlhf/train", **common)
        val, val_stats = sample_preference_examples(val_rows, hh_parser, n=args.dpo_val, id_prefix="hh-val", source="Anthropic/hh-rlhf/test", **common)
        write_jsonl(output_dir / "hh_train.jsonl", train)
        write_jsonl(output_dir / "hh_val.jsonl", val)
        manifest["hh"] = {"train": train_stats, "val": val_stats, "dataset": "Anthropic/hh-rlhf"}
        print(f"[hh] train={len(train)} val={len(val)} ({time.time() - start:.0f}s) {train_stats}")

    # ------------------------------------------------------------------ UltraFeedback
    if "uf" in args.only:
        start = time.time()
        train_rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
        val_rows = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
        train, train_stats = sample_preference_examples(train_rows, parse_ultrafeedback_pair, n=args.dpo_train, id_prefix="uf-train", source="ultrafeedback_binarized/train_prefs", **common)
        val, val_stats = sample_preference_examples(val_rows, parse_ultrafeedback_pair, n=args.dpo_val, id_prefix="uf-val", source="ultrafeedback_binarized/test_prefs", **common)
        write_jsonl(output_dir / "uf_train.jsonl", train)
        write_jsonl(output_dir / "uf_val.jsonl", val)
        manifest["uf"] = {"train": train_stats, "val": val_stats, "dataset": "HuggingFaceH4/ultrafeedback_binarized"}
        print(f"[uf] train={len(train)} val={len(val)} ({time.time() - start:.0f}s) {train_stats}")

    write_json(manifest_path, manifest)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
