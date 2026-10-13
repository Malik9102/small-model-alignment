"""Stable grading interface for the SFT and DPO implementation.

Each function is a thin wrapper around the implementation in ``alignment/``:

* ``alignment/batching.py``  - tokenization, EOS appending, truncation, padding, -100 masking
* ``alignment/logprobs.py``  - response-only summed log probabilities
* ``alignment/losses.py``    - SFT token-mean loss and the DPO objective
* ``alignment/dpo_core.py``  - policy / frozen-reference scoring

Conventions (see ``alignment/batching.py`` for details):
* Rows are right-padded to ``max_length``; longer sequences are truncated to
  ``max_length`` after the EOS has been appended to the response.
* Prompt and padding positions have label ``-100``; labels are unshifted.
* SFT reduction: mean over all unmasked response tokens in the batch.
* Input row order is preserved everywhere.
"""

from __future__ import annotations

from alignment.batching import build_lm_batch, build_preference_batch, get_pad_id
from alignment.dpo_core import preference_logps, reference_logps
from alignment.logprobs import model_device, move_batch, response_logps
from alignment.losses import dpo_loss_from_logps, sft_token_mean_loss


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    return build_lm_batch(tokenizer, prompts, responses, max_length, pad_to="max_length")


def _summed_logps_and_counts(model, tokenizer, prompts, responses, max_length):
    batch = move_batch(build_sft_batch(tokenizer, prompts, responses, max_length), model_device(model))
    return response_logps(model, batch, fast=False)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    summed, _ = _summed_logps_and_counts(model, tokenizer, prompts, responses, max_length)
    return summed


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss (mean over all response tokens)."""

    summed, counts = _summed_logps_and_counts(model, tokenizer, prompts, responses, max_length)
    return sft_token_mean_loss(summed, counts)


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    return build_preference_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to="max_length"
    )


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    return dpo_loss_from_logps(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    batch = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    pad_id = get_pad_id(tokenizer)
    policy_chosen, policy_rejected = preference_logps(policy_model, batch, pad_id, fast=False)
    ref_chosen, ref_rejected = reference_logps(
        reference_model, batch, model_device(policy_model), pad_id, fast=False
    )
    return compute_dpo_loss_from_logps(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta)
