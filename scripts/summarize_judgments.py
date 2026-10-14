"""Summaries from judged JSONL files: win/loss/tie table, per-category, position bias.

Examples:
    python scripts/summarize_judgments.py --judged outputs/eval/dpo_uf_vs_sft_judged.jsonl --name "UF-DPO vs SFT"
    python scripts/summarize_judgments.py --judged outputs/eval/dpo_uf_vs_sft_judged.jsonl \
        --mirror_judged outputs/eval/dpo_uf_vs_sft_mirrored_judged.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.judging import format_summary, position_consistency, summarize_judgments, summary_markdown_row  # noqa: E402
from alignment.utils import fix_markdown_tables, read_jsonl, write_json  # noqa: E402

HEADER = (
    "| Comparison | n | Wins | Ties | Losses | Win rate excl. ties [95% CI] | Win rate ties = ½ [95% CI] | Unknown | Failed | Pairs retried | Judge |\n"
    "|---|"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", nargs="+", required=True)
    parser.add_argument("--name", nargs="*", default=None, help="Display names (same order as --judged)")
    parser.add_argument("--mirror_judged", default=None, help="A/B-swapped judgments of the FIRST --judged file")
    parser.add_argument("--output", default=None, help="Optional JSON with all summaries")
    args = parser.parse_args()

    all_summaries = {}
    rows = []
    for index, path in enumerate(args.judged):
        records = read_jsonl(path)
        summary = summarize_judgments(records)
        summary["judge_model"] = records[0].get("judge_model", "") if records else ""
        summary["rubric"] = records[0].get("rubric", "") if records else ""
        name = args.name[index] if args.name and index < len(args.name) else Path(path).stem
        all_summaries[name] = summary
        print(format_summary(summary))
        print()
        rows.append(summary_markdown_row(name, summary))
    print(fix_markdown_tables(HEADER + "\n" + "\n".join(rows)))
    if args.mirror_judged:
        consistency = position_consistency(read_jsonl(args.judged[0]), read_jsonl(args.mirror_judged))
        all_summaries["position_consistency"] = consistency
        print(f"\nPosition-swap consistency: {consistency}")
    if args.output:
        write_json(args.output, all_summaries)


if __name__ == "__main__":
    main()
