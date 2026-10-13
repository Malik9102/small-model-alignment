"""Pieces shared by the SFT and DPO training loops."""

from __future__ import annotations

import contextlib
import math

import torch

from alignment.batching import build_example_ids, collate_examples, encode_many, get_eos_id, get_pad_id
from alignment.prompting import format_prompt


def resolve_device(name: str | None, fallback_index: int = 0) -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        index = min(fallback_index, torch.cuda.device_count() - 1)
        return torch.device(f"cuda:{index}")
    return torch.device("cpu")


def load_tokenizer(path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def dtype_kwarg() -> str:
    """``dtype`` on transformers>=4.56, ``torch_dtype`` on older versions."""

    import transformers
    from packaging.version import Version

    return "dtype" if Version(transformers.__version__) >= Version("4.56.0") else "torch_dtype"


def load_causal_lm(path: str, dtype: str = "float32", device=None, gradient_checkpointing: bool = False):
    from transformers import AutoModelForCausalLM

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    kwargs = {dtype_kwarg(): torch_dtype}
    if torch.cuda.is_available():
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if device is not None:
        model.to(device)
    return model


def autocast_context(device: torch.device, enabled: bool):
    """fp16 autocast on CUDA (T4 has no bf16); no-op elsewhere."""

    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_optimizer(model, name: str, lr: float, weight_decay: float = 0.0, betas=(0.9, 0.999), eps: float = 1e-8):
    """AdamW (decay only on >=2-D weights) or RMSprop."""

    decay, no_decay = [], []
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas), eps=eps)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, lr=lr, eps=eps)
    raise ValueError(f"Unknown optimizer {name!r}")


def build_scheduler(optimizer, schedule: str, total_steps: int, warmup_steps: int = 0, min_lr_ratio: float = 0.0):
    """constant | constant_with_warmup | cosine (linear warmup then cosine decay)."""

    schedule = schedule.lower()

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule in ("constant", "constant_with_warmup"):
            return 1.0
        if schedule == "cosine":
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unknown schedule {schedule!r}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ---------------------------------------------------------------------------
# Pre-tokenized datasets
# ---------------------------------------------------------------------------

class SFTDataset(torch.utils.data.Dataset):
    """Pre-tokenized (prompt_ids, response_ids) pairs."""

    def __init__(self, rows, tokenizer, max_length: int):
        self.max_length = max_length
        self.eos_id = get_eos_id(tokenizer)
        prompts = encode_many(tokenizer, [format_prompt(row["instruction"]) for row in rows])
        responses = encode_many(tokenizer, [row["response"] for row in rows])
        self.items = list(zip(prompts, responses))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        prompt_ids, response_ids = self.items[index]
        return build_example_ids(prompt_ids, response_ids, self.eos_id, self.max_length)


class PreferenceDataset(torch.utils.data.Dataset):
    """Pre-tokenized (prompt, chosen, rejected) triples with the shared prompt."""

    def __init__(self, rows, tokenizer, max_length: int, max_prompt_length: int):
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.eos_id = get_eos_id(tokenizer)
        prompts = encode_many(tokenizer, [format_prompt(row["instruction"]) for row in rows])
        chosen = encode_many(tokenizer, [row["chosen"] for row in rows])
        rejected = encode_many(tokenizer, [row["rejected"] for row in rows])
        self.items = list(zip(prompts, chosen, rejected))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        prompt_ids, chosen_ids, rejected_ids = self.items[index]
        chosen = build_example_ids(prompt_ids, chosen_ids, self.eos_id, self.max_length, self.max_prompt_length)
        rejected = build_example_ids(prompt_ids, rejected_ids, self.eos_id, self.max_length, self.max_prompt_length)
        return chosen, rejected


def make_sft_collate(tokenizer, max_length: int):
    pad_id = get_pad_id(tokenizer)

    def collate(examples):
        return collate_examples(examples, pad_id, max_length, pad_to="longest", pad_to_multiple_of=8)

    return collate


def make_preference_collate(tokenizer, max_length: int):
    """Collate to ``{"chosen": batch, "rejected": batch}`` with a shared width."""

    pad_id = get_pad_id(tokenizer)

    def collate(pairs):
        chosen = [pair[0] for pair in pairs]
        rejected = [pair[1] for pair in pairs]
        width = max(len(ids) for ids, _ in chosen + rejected)
        width = min(-(-width // 8) * 8, max_length)
        return {
            "chosen": collate_examples(chosen, pad_id, width, pad_to="max_length"),
            "rejected": collate_examples(rejected, pad_id, width, pad_to="max_length"),
        }

    return collate


def count_trainable(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
