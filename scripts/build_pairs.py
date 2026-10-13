"""Build a blinded pairwise comparison file (candidate vs baseline).

Response order is randomized per pair with a fixed seed; the position -> model
mapping is stored only under each pair's ``hidden`` field.

Examples:
    python scripts/build_pairs.py --candidate outputs/generations/dpo_uf_alpaca.json \
        --baseline outputs/generations/sft_alpaca.json --output outputs/eval/dpo_uf_vs_sft_pairs.jsonl

    # A/B-swapped copy of the first 100 pairs, to measure judge position bias
    python scripts/build_pairs.py --mirror_of outputs/eval/dpo_uf_vs_sft_pairs.jsonl --limit 100 \
        --output outputs/eval/dpo_uf_vs_sft_pairs_mirrored.jsonl
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judging import build_pairs, mirror_pairs  # noqa: E402
from alignment.utils import read_json, read_jsonl, write_jsonl  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", help="Generation JSON of the model being evaluated (e.g. DPO)")
    parser.add_argument("--baseline", help="Generation JSON of the reference model (e.g. SFT)")
    parser.add_argument("--candidate_name", default=None)
    parser.add_argument("--baseline_name", default=None)
    parser.add_argument("--mirror_of", default=None, help="Existing pairs JSONL to A/B-swap instead")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mirror_of:
        pairs = mirror_pairs(read_jsonl(args.mirror_of), args.limit)
    else:
        candidate_rows = read_json(args.candidate)
        baseline_rows = read_json(args.baseline)
        candidate_name = args.candidate_name or candidate_rows[0]["generator"]
        baseline_name = args.baseline_name or baseline_rows[0]["generator"]
        if candidate_name == baseline_name:
            raise SystemExit("candidate and baseline must have different names")
        pairs, missing = build_pairs(candidate_rows, baseline_rows, candidate_name, baseline_name, args.seed)
        if missing:
            print(f"[pairs] WARNING: {len(missing)} candidate ids missing from baseline: {missing[:5]}")
        if args.limit:
            pairs = pairs[: args.limit]
    write_jsonl(args.output, pairs)
    positions = Counter(pair["hidden"]["candidate_position"] for pair in pairs)
    print(f"[pairs] wrote {len(pairs)} pairs -> {args.output} | candidate position counts {dict(positions)}")


if __name__ == "__main__":
    main()
