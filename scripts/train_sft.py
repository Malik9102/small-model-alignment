"""Train the SFT model from a YAML config.

Example:
    python scripts/train_sft.py --config configs/sft.yaml --set max_steps=20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alignment.utils import add_common_args, load_config  # noqa: E402
from alignment.sft_trainer import train_sft  # noqa: E402


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    summary = Path(config.get("output_dir", "")) / "summary.json"
    if summary.exists() and not config.get("overwrite", False):
        # Restartable notebooks (manual 5.5): a finished run is not repeated.
        print(f"[train_sft] {summary} exists - run already complete, skipping (set overwrite=true to retrain)")
        return
    train_sft(config)


if __name__ == "__main__":
    main()
