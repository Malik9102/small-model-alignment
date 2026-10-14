"""Does the judge simply prefer longer answers? (Dubois et al. 2024, Length-Controlled AlpacaEval.)

For each judged comparison:
* mean response length (words) of candidate vs baseline;
* how often the judge picked the LONGER response (among decided, non-tie pairs);
* candidate win rate split by whether the candidate's answer was longer or shorter;
* candidate win rate on near-equal-length pairs (|length difference| <= 20%), a crude
  length-controlled estimate.

Example:
    python scripts/analyze_length_bias.py \
        --pairs outputs/eval/dpo_uf_vs_sft_pairs.jsonl --judged outputs/eval/dpo_uf_vs_sft_judged.jsonl \
        --name "UF-DPO vs SFT" --output outputs/eval/length_bias.json
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judging import wilson_interval  # noqa: E402
from alignment.utils import read_json, read_jsonl, write_json  # noqa: E402


def words(text: str) -> int:
    return len((text or "").split())


def analyze(pairs_path, judged_path) -> dict:
    pairs = {pair["pair_id"]: pair for pair in read_jsonl(pairs_path)}
    judged = [r for r in read_jsonl(judged_path) if r["status"] == "ok" and r["pair_id"] in pairs]
    cand_len, base_len = [], []
    picked_longer = decided_unequal = 0
    buckets = {"candidate_longer": [0, 0], "candidate_shorter": [0, 0], "near_equal": [0, 0]}  # [wins, decided]
    for record in judged:
        pair = pairs[record["pair_id"]]
        a, b = words(pair["response_a"]), words(pair["response_b"])
        cand, base = (a, b) if pair["hidden"]["candidate_position"] == "A" else (b, a)
        cand_len.append(cand)
        base_len.append(base)
        if record["verdict"] == "tie":
            continue
        if a != b:
            decided_unequal += 1
            picked_longer += int((record["verdict"] == "A") == (a > b))
        win = record["candidate_outcome"] == "win"
        if abs(cand - base) <= 0.2 * max(cand, base, 1):
            key = "near_equal"
        else:
            key = "candidate_longer" if cand > base else "candidate_shorter"
        buckets[key][0] += int(win)
        buckets[key][1] += 1
    result = {
        "pairs_judged": len(judged),
        "mean_words_candidate": round(statistics.mean(cand_len), 1) if cand_len else None,
        "mean_words_baseline": round(statistics.mean(base_len), 1) if base_len else None,
        "candidate_longer_fraction": (sum(c > b for c, b in zip(cand_len, base_len)) / len(cand_len)) if cand_len else None,
        "judge_picked_longer_rate": picked_longer / decided_unequal if decided_unequal else None,
        "judge_picked_longer_ci95": wilson_interval(picked_longer, decided_unequal),
        "decided_pairs_with_unequal_length": decided_unequal,
    }
    for key, (wins, decided) in buckets.items():
        result[f"win_rate_{key}"] = wins / decided if decided else None
        result[f"n_{key}"] = decided
    return result


def pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--judged", nargs="+", required=True)
    parser.add_argument("--name", nargs="+", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not (len(args.pairs) == len(args.judged) == len(args.name)):
        raise SystemExit("--pairs, --judged and --name need the same number of entries")
    results = {}
    header = (
        "| Comparison | Mean words (cand / base) | Cand. longer | Judge picked longer [95% CI] "
        "| Win rate: cand. longer | Win rate: cand. shorter | Win rate: near-equal length (n) |"
    )
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for pairs_path, judged_path, name in zip(args.pairs, args.judged, args.name):
        if not Path(judged_path).exists():
            print(f"[length] skipping {name}: {judged_path} not found")
            continue
        r = analyze(pairs_path, judged_path)
        results[name] = r
        ci = r["judge_picked_longer_ci95"]
        ci_text = f" [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else ""
        lines.append(
            f"| {name} | {r['mean_words_candidate']} / {r['mean_words_baseline']} | {pct(r['candidate_longer_fraction'])} "
            f"| {pct(r['judge_picked_longer_rate'])}{ci_text} | {pct(r['win_rate_candidate_longer'])} "
            f"| {pct(r['win_rate_candidate_shorter'])} | {pct(r['win_rate_near_equal'])} ({r['n_near_equal']}) |"
        )
    print("\n".join(lines))
    if args.output:
        previous = read_json(args.output) if Path(args.output).exists() else {}
        write_json(args.output, {**previous, **results})


if __name__ == "__main__":
    main()
