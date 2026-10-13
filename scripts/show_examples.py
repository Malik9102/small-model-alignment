"""Print de-blinded examples (wins / losses / ties for the candidate) for the report.

Example:
    python scripts/show_examples.py --pairs outputs/eval/dpo_uf_vs_sft_pairs.jsonl \
        --judged outputs/eval/dpo_uf_vs_sft_judged.jsonl --k 3 --output outputs/eval/dpo_uf_examples.md
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.utils import read_jsonl  # noqa: E402


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + " [...]"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--judged", required=True)
    parser.add_argument("--k", type=int, default=3, help="Examples per outcome")
    parser.add_argument("--category", default=None)
    parser.add_argument("--max_chars", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    pairs = {pair["pair_id"]: pair for pair in read_jsonl(args.pairs)}
    judged = [r for r in read_jsonl(args.judged) if r["status"] == "ok"]
    if args.category:
        judged = [r for r in judged if r.get("category") == args.category]
    rng = random.Random(args.seed)
    lines = []
    for outcome in ("win", "loss", "tie"):
        chosen = [r for r in judged if r["candidate_outcome"] == outcome]
        rng.shuffle(chosen)
        label = {"win": "WINS", "loss": "LOSSES", "tie": "TIES"}[outcome]
        lines.append(f"\n## Candidate {label} ({len(chosen)} total, showing {min(args.k, len(chosen))})\n")
        for record in chosen[: args.k]:
            pair = pairs[record["pair_id"]]
            hidden = pair["hidden"]
            cand = pair["response_a"] if hidden["candidate_position"] == "A" else pair["response_b"]
            base = pair["response_b"] if hidden["candidate_position"] == "A" else pair["response_a"]
            lines += [
                f"### {record['pair_id']} ({record.get('category') or pair.get('dataset')})",
                f"**Prompt:** {clip(pair['instruction'], 400)}",
                f"**{hidden['candidate']}:** {clip(cand, args.max_chars)}",
                f"**{hidden['baseline']}:** {clip(base, args.max_chars)}",
                f"**Judge ({record['verdict']}):** {record['reason']}",
                "",
            ]
    text = "\n".join(lines)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
