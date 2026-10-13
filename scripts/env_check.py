"""Task 1: confirm the Kaggle environment (two T4 GPUs + package versions)."""

from __future__ import annotations

import importlib
import platform
import subprocess


def version(name: str) -> str:
    try:
        return getattr(importlib.import_module(name), "__version__", "installed")
    except Exception as error:  # pragma: no cover
        return f"NOT AVAILABLE ({error.__class__.__name__})"


def main():
    print(f"python       {platform.python_version()}")
    for name in ("torch", "transformers", "datasets", "accelerate", "peft", "trl"):
        print(f"{name:12s} {version(name)}")
    import torch

    print(f"cuda available: {torch.cuda.is_available()} | device count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"  cuda:{index} {props.name} {props.total_memory / 2**30:.1f} GiB | bf16 supported: {torch.cuda.is_bf16_supported()}")
    try:
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False).stdout)
    except FileNotFoundError:
        print("nvidia-smi not found (no GPU attached?)")
    if torch.cuda.device_count() < 2:
        print("WARNING: fewer than 2 GPUs visible. Select 'GPU T4 x2' in the Kaggle notebook settings.")


if __name__ == "__main__":
    main()
