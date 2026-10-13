"""Small shared utilities: configs, seeding, JSON/JSONL IO, metrics logging."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _parse_value(raw: str):
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def load_config(path: str | os.PathLike, overrides: Iterable[str] = ()) -> dict[str, Any]:
    """Load a YAML config and apply ``key=value`` overrides (dotted keys allowed)."""

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for override in overrides or ():
        if "=" not in override:
            raise ValueError(f"Override must look like key=value, got {override!r}")
        key, raw = override.split("=", 1)
        target = config
        parts = key.strip().split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = _parse_value(raw)
    return config


def add_common_args(parser):
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override: key=value")
    return parser


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload, indent: int = 2) -> None:
    """Atomic JSON write (tmp file + rename) so a disconnect never leaves half a file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=False)
    os.replace(tmp, path)


def read_jsonl(path) -> list[dict]:
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def append_jsonl(path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


class MetricsLogger:
    """Append-only JSONL metrics file; every line is flushed immediately."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.start = time.time()

    def log(self, split: str, step: int, **values) -> dict:
        row = {"split": split, "step": int(step), "elapsed_s": round(time.time() - self.start, 1)}
        for key, value in values.items():
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                value = str(value)
            row[key] = value
        append_jsonl(self.path, row)
        return row


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(model, tokenizer, directory, state: dict | None = None) -> None:
    """Save model + tokenizer (+ JSON state) to ``directory`` via a temp dir swap."""

    directory = Path(directory)
    tmp = directory.with_name(directory.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(tmp, safe_serialization=True)
    tokenizer.save_pretrained(tmp)
    if state is not None:
        write_json(tmp / "training_state.json", state)
    if directory.exists():
        shutil.rmtree(directory)
    os.replace(tmp, directory)


def format_seconds(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    return f"{seconds // 3600:d}h{(seconds % 3600) // 60:02d}m{seconds % 60:02d}s"


def describe_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        return "CUDA not available"
    lines = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        lines.append(f"cuda:{index} {props.name} {props.total_memory / 2**30:.1f} GiB")
    return "; ".join(lines)


def cuda_memory_gib(device=None) -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 2**30


def count_lines(path) -> int:
    path = Path(path)
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def fix_markdown_tables(text: str) -> str:
    """Make every Markdown separator row (|---|---|) match the column count of the header above it."""

    import re

    lines = text.split("\n")
    for index in range(1, len(lines)):
        if re.fullmatch(r"\|(\s*-+\s*\|)+", lines[index].strip()) and lines[index - 1].strip().startswith("|"):
            columns = lines[index - 1].strip().count("|") - 1
            lines[index] = "|" + "---|" * columns
    return "\n".join(lines)
