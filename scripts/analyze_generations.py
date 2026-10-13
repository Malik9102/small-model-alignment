"""Descriptive statistics of generation files (for the error-analysis section).

Per generator: length, truncation at max_new_tokens, empty outputs, degenerate
repetition, refusal-phrase rate (overall and per category), and persona-style
markers. These are heuristics for *describing* behaviour (e.g. verbosity or
over-refusal / alignment tax); they are not a substitute for the judge.

Example:
    python scripts/analyze_generations.py outputs/generations/*.json --output outputs/eval/generation_stats.json
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.utils import read_json, write_json  # noqa: E402

REFUSAL_PATTERNS = re.compile(
    r"\b(i can(?:no|')t|i cannot|i'm not able to|i am not able to|i won't|i will not|i'm sorry|i am sorry|"
    r"as an ai|i'm unable|i am unable|not appropriate|i must decline|cannot assist|can't assist|can't help with)\b",
    re.IGNORECASE,
)
PERSONA_PATTERNS = re.compile(
    r"\b(donkey|idiot sandwich|bloody|disaster|raw|stop|wake up|for god's sake|absolute|disgrace|pathetic|"
    r"what are you|are you serious|unbelievable|shut it down|come on)\b",
    re.IGNORECASE,
)


def is_repetitive(text: str, n: int = 4, threshold: float = 0.5) -> bool:
    tokens = text.split()
    if len(tokens) < 40:
        return False
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) < threshold


def analyze(rows):
    words = [len(row["output"].split()) for row in rows]
    refusal = [bool(REFUSAL_PATTERNS.search(row["output"])) for row in rows]
    by_category = defaultdict(list)
    for row, refused in zip(rows, refusal):
        by_category[row.get("category") or "all"].append(refused)
    exclamations = [row["output"].count("!") / max(len(row["output"].split()), 1) * 100 for row in rows]
    return {
        "generator": rows[0].get("generator", "?") if rows else "?",
        "dataset": rows[0].get("dataset", "?") if rows else "?",
        "n": len(rows),
        "mean_words": round(statistics.mean(words), 1) if words else 0,
        "median_words": statistics.median(words) if words else 0,
        "hit_length_limit_rate": sum(r.get("finish_reason") == "length" for r in rows) / max(len(rows), 1),
        "empty_rate": sum(not r["output"].strip() for r in rows) / max(len(rows), 1),
        "repetitive_rate": sum(is_repetitive(r["output"]) for r in rows) / max(len(rows), 1),
        "refusal_phrase_rate": sum(refusal) / max(len(rows), 1),
        "refusal_phrase_rate_by_category": {c: round(sum(v) / len(v), 3) for c, v in sorted(by_category.items())},
        "persona_marker_rate": sum(bool(PERSONA_PATTERNS.search(r["output"])) for r in rows) / max(len(rows), 1),
        "exclamations_per_100_words": round(statistics.mean(exclamations), 2) if exclamations else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    stats = []
    for path in args.files:
        if path.endswith(".meta.json"):
            continue
        rows = read_json(path)
        if not rows or "output" not in rows[0]:
            continue
        entry = analyze(rows)
        entry["file"] = path
        stats.append(entry)
    header = "| Generator | Eval set | n | Mean words | Hit max tokens | Repetitive | Refusal phrases | Persona markers | ! per 100 words |"
    print(header)
    print("|" + "---|" * (header.count("|") - 1))
    for s in stats:
        print(
            f"| {s['generator']} | {s['dataset']} | {s['n']} | {s['mean_words']} | {100 * s['hit_length_limit_rate']:.1f}% "
            f"| {100 * s['repetitive_rate']:.1f}% | {100 * s['refusal_phrase_rate']:.1f}% | {100 * s['persona_marker_rate']:.1f}% "
            f"| {s['exclamations_per_100_words']} |"
        )
    if args.output:
        write_json(args.output, stats)


if __name__ == "__main__":
    main()
