"""Response-only log probabilities for causal LMs.

The model sees ``input_ids``; the logit at position ``t`` predicts token ``t+1``.
We therefore score ``labels[:, 1:]`` with ``logits[:, :-1]`` and only count
positions whose label is not ``-100`` (prompt and padding are excluded).

Two implementations:

* ``generic``: works with any model returning ``.logits`` (including the toy
  models used by the grader). Used by the submission adapter.
* ``fast``: for Hugging Face decoder-only models (Qwen2, Llama, ...). It runs the
  transformer body once, then applies the LM head *only* at response positions.
  With Qwen's ~152k vocabulary this skips most of the logits memory/compute for
  prompt and padding positions. It is numerically equivalent (tested in
  ``tests/test_core_extended.py``).

All log-softmax math is done in at least float32 (fp16/bf16 logits are upcast;
float64 is preserved).
"""

from __future__ import annotations

import torch

from alignment.batching import IGNORE_INDEX

# Model families whose CausalLM logits are exactly ``lm_head(last_hidden_state)``.
_FAST_PATH_MODEL_TYPES = {"qwen2", "qwen3", "llama", "mistral"}


def model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover
        return torch.device("cpu")


def move_batch(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def supports_fast_path(model) -> bool:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) not in _FAST_PATH_MODEL_TYPES:
        return False
    if not hasattr(model, "get_decoder") or not hasattr(model, "get_output_embeddings"):
        return False
    return model.get_output_embeddings() is not None


def _upcast(tensor: torch.Tensor) -> torch.Tensor:
    """fp16/bf16 -> fp32 for a stable log-softmax; fp32/fp64 keep their dtype."""

    return tensor.float() if tensor.dtype in (torch.float16, torch.bfloat16) else tensor


def _generic_response_logps(model, input_ids, attention_mask, labels):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    safe_targets = targets.masked_fill(~mask, 0)

    logits = _upcast(logits)
    selected = torch.gather(logits, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    token_logps = selected - torch.logsumexp(logits, dim=-1)
    token_logps = token_logps.masked_fill(~mask, 0.0)
    return token_logps.sum(dim=-1), mask.sum(dim=-1)


def _fast_response_logps(model, input_ids, attention_mask, labels):
    decoder = model.get_decoder()
    hidden = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
    targets = labels[:, 1:]
    mask = targets.ne(IGNORE_INDEX)
    batch_size = input_ids.shape[0]
    counts = mask.sum(dim=-1)
    sums = torch.zeros(batch_size, dtype=_upcast(hidden[:1, :1, :1]).dtype, device=input_ids.device)
    if mask.any():
        positions_hidden = hidden[:, :-1, :][mask]  # [N, H] response positions only
        logits = _upcast(model.get_output_embeddings()(positions_hidden))  # [N, V]
        token_targets = targets[mask]
        token_logps = logits.gather(-1, token_targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
        rows = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand_as(mask)[mask]
        sums = sums.index_add(0, rows, token_logps)
    return sums, counts


def response_logps(model, batch: dict[str, torch.Tensor], fast: bool | None = None):
    """Return ``(summed_logps[B], response_token_counts[B])`` for a labelled batch.

    ``batch`` must already be on the model's device. Row order is preserved.
    ``fast=None`` auto-selects the fast path for supported HF models.
    """

    if fast is None:
        fast = supports_fast_path(model)
    if fast:
        return _fast_response_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
    return _generic_response_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"])
