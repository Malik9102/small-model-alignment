"""Judge blinded pairwise files with a Kaggle model-proxy LLM (kaggle_benchmarks).

Run this inside a Kaggle notebook configured as a *benchmark* notebook so the
model proxy credentials exist. Judging is resumable: rerunning with
--reuse_judgments skips pairs that already have a valid verdict.

Example (manual Section 13.2):
    python scripts/judge_pairwise_kaggle.py \
        --pairs_path outputs/eval/dpo_vs_sft_pairs.jsonl \
        --output_path outputs/eval/dpo_vs_sft_judged.jsonl \
        --judge_model google/gemini-3.6-flash \
        --batch_size 5 --sleep_seconds 5 --max_retries 3 --reuse_judgments

Offline dry run (no API calls):  add  --backend mock --sleep_seconds 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judging import RUBRICS, run_judging  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.6-flash")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--sleep_seconds", type=float, default=5.0, help="Minimum gap between requests (>=5 for kbench)")
    parser.add_argument("--max_retries", type=int, default=3, help="Attempts per batch")
    parser.add_argument("--reuse_judgments", action="store_true")
    parser.add_argument("--rubric", choices=sorted(RUBRICS), default="general")
    parser.add_argument("--failures_path", default=None)
    parser.add_argument("--summary_path", default=None)
    parser.add_argument("--backend", choices=["kbench", "mock"], default="kbench")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--keep_unknown", action="store_true", help="Do not re-judge 'unknown' verdicts on resume")
    args = parser.parse_args()
    run_judging(
        pairs_path=args.pairs_path,
        output_path=args.output_path,
        judge_model=args.judge_model,
        batch_size=args.batch_size,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
        reuse_judgments=args.reuse_judgments,
        rubric=args.rubric,
        failures_path=args.failures_path,
        summary_path=args.summary_path,
        backend=args.backend,
        limit=args.limit,
        retry_unknown=not args.keep_unknown,
    )


if __name__ == "__main__":
    main()
