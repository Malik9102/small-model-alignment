"""End-to-end CPU smoke test of the real scripts on a tiny random model.

prepare (synthetic) -> SFT -> DPO -> generate (SFT, DPO) -> pairs -> mock judge
-> plots -> report tables. Takes well under a minute; catches path/config/API
breakage before you spend Kaggle GPU hours.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def run(*args):
    result = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, f"{' '.join(args)}\nSTDOUT:\n{result.stdout[-3000:]}\nSTDERR:\n{result.stderr[-3000:]}"
    return result.stdout


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_full_pipeline_on_tiny_model(tmp_path):
    pytest.importorskip("transformers")
    sys.path.insert(0, str(REPO))
    from tests.tiny_hf import build_tiny_model

    model = build_tiny_model(tmp_path / "tiny")
    rng = random.Random(0)
    words = "the quick brown fox sky blue oven pan sear rain".split()
    sent = lambda n: " ".join(rng.choice(words) for _ in range(n))  # noqa: E731
    data = tmp_path / "data"
    write_rows(data / "sft_train.jsonl", [{"id": f"s{i}", "instruction": sent(4), "response": sent(6)} for i in range(32)])
    write_rows(data / "sft_val.jsonl", [{"id": f"v{i}", "instruction": sent(4), "response": sent(6)} for i in range(8)])
    for split, n in (("train", 32), ("val", 8)):
        write_rows(data / f"hh_{split}.jsonl", [{"id": f"p{i}", "instruction": sent(4), "chosen": "blue sky " + sent(3), "rejected": sent(5)} for i in range(n)])
    out = tmp_path / "outputs"
    common = ["--set", "gradient_accumulation_steps=4", "--set", "eval_every=4", "--set", "log_every=2"]
    run("scripts/train_sft.py", "--config", "configs/sft.yaml", "--set", f"model_name={model}",
        "--set", f"train_file={data / 'sft_train.jsonl'}", "--set", f"val_file={data / 'sft_val.jsonl'}",
        "--set", f"output_dir={out / 'sft'}", *common)
    assert (out / "sft" / "best" / "model.safetensors").exists() and (out / "sft" / "final").exists()
    run("scripts/train_dpo.py", "--config", "configs/dpo_hh.yaml", "--set", f"sft_checkpoint={out / 'sft' / 'best'}",
        "--set", f"train_file={data / 'hh_train.jsonl'}", "--set", f"val_file={data / 'hh_val.jsonl'}",
        "--set", f"output_dir={out / 'dpo_hh'}", *common)
    first_val = next(json.loads(l) for l in (out / "dpo_hh" / "metrics.jsonl").read_text().splitlines() if '"val"' in l)
    assert first_val["step"] == 0 and first_val["reward_accuracy"] == 0.0
    assert abs(first_val["loss"] - 0.693147) < 1e-4
    gen = out / "generations"
    for name, ckpt in (("tiny-sft", out / "sft" / "best"), ("tiny-dpo-hh", out / "dpo_hh" / "best")):
        run("scripts/generate.py", "--model_path", str(ckpt), "--eval_set", "eval_sets/hh_rlhf_eval_300_seed42.json",
            "--generator", name, "--output", str(gen / f"{name}_hh.json"), "--limit", "6", "--max_new_tokens", "8", "--batch_size", "3")
    ev = out / "eval"
    run("scripts/build_pairs.py", "--candidate", str(gen / "tiny-dpo-hh_hh.json"), "--baseline", str(gen / "tiny-sft_hh.json"),
        "--output", str(ev / "dpo_hh_vs_sft_pairs.jsonl"))
    run("scripts/judge_pairwise_kaggle.py", "--pairs_path", str(ev / "dpo_hh_vs_sft_pairs.jsonl"),
        "--output_path", str(ev / "dpo_hh_vs_sft_judged.jsonl"), "--backend", "mock", "--sleep_seconds", "0",
        "--reuse_judgments", "--rubric", "safety")
    summary = json.loads((ev / "dpo_hh_vs_sft_judged_summary.json").read_text())
    assert summary["overall"]["n"] == 6 and summary["overall"]["failed"] == 0
    run("scripts/plot_metrics.py", "--sft", str(out / "sft"), "--dpo", str(out / "dpo_hh"), "--output_dir", str(out / "plots"))
    assert (out / "plots" / "dpo_hh_training.png").exists()
    tables = run("scripts/collect_results.py", "--runs_root", str(out), "--output", str(out / "report_tables.md"))
    assert "dpo_hh_vs_sft" in tables
