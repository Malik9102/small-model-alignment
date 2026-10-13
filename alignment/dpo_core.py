"""Policy/reference scoring shared by the DPO training loop and the adapter."""

from __future__ import annotations

import torch

from alignment.batching import stack_preference_batch
from alignment.logprobs import model_device, move_batch, response_logps


def preference_logps(model, preference_batch, pad_id: int = 0, fast: bool | None = None):
    """Summed response log-probs for chosen and rejected rows in ONE forward pass.

    Returns ``(chosen_logps[B], rejected_logps[B])`` on the model's device, with
    gradients if grad mode is enabled.
    """

    batch_size = preference_batch["chosen"]["input_ids"].shape[0]
    stacked = move_batch(stack_preference_batch(preference_batch, pad_id), model_device(model))
    summed, _ = response_logps(model, stacked, fast=fast)
    return summed[:batch_size], summed[batch_size:]


def reference_logps(reference_model, preference_batch, target_device, pad_id: int = 0, fast: bool | None = None):
    """Frozen-reference log-probs computed under ``no_grad`` and moved to the policy device."""

    was_training = getattr(reference_model, "training", False)
    if hasattr(reference_model, "eval"):
        reference_model.eval()  # frozen reference: deterministic (no dropout) scoring
    try:
        with torch.no_grad():
            chosen, rejected = preference_logps(reference_model, preference_batch, pad_id, fast)
    finally:
        if was_training:
            reference_model.train()
    return chosen.detach().to(target_device), rejected.detach().to(target_device)
