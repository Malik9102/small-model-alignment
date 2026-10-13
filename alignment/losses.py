"""SFT and DPO objectives.

SFT reduction (documented choice): **mean over all unmasked response tokens in
the batch**, i.e. ``-sum(response log probs) / number_of_response_tokens``. In
the training loop the normalizer is the token count of the whole effective
batch (all gradient-accumulation micro-batches), so every response token in an
optimizer step gets equal weight.

DPO (Rafailov et al., 2023):
    L = -log sigmoid(beta * [(pi_c - pi_r) - (ref_c - ref_r)])
where ``pi_*`` / ``ref_*`` are summed response log probabilities under the
policy / frozen reference. The implicit rewards are
``r_w = beta * (pi_c - ref_c)`` and ``r_l = beta * (pi_r - ref_r)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sft_token_mean_loss(summed_logps: torch.Tensor, token_counts: torch.Tensor) -> torch.Tensor:
    """Mean negative log-likelihood over all response tokens in the batch."""

    total_tokens = token_counts.sum().clamp_min(1).to(summed_logps.dtype)
    return -summed_logps.sum() / total_tokens


def _result_dtype(*values) -> torch.dtype:
    """float64 inputs stay float64; everything else (incl. fp16/bf16/ints) computes in float32."""

    dtypes = [v.dtype for v in values if isinstance(v, torch.Tensor) and v.is_floating_point()]
    return torch.float64 if torch.float64 in dtypes else torch.float32


def _as_float_tensor(value, dtype: torch.dtype, device=None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps):
    """(Delta_theta - Delta_ref) per example, before multiplying by beta."""

    values = (policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    dtype = _result_dtype(*values)
    device = policy_chosen_logps.device if isinstance(policy_chosen_logps, torch.Tensor) else None
    pc, pr, rc, rr = (_as_float_tensor(value, dtype, device) for value in values)
    return (pc - pr) - (rc - rr)


def dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float,
) -> torch.Tensor:
    """Scalar batch-mean DPO loss. Uses logsigmoid for numerical stability."""

    logits = dpo_logits(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps)
    return -F.logsigmoid(float(beta) * logits).mean()


@torch.no_grad()
def dpo_statistics(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta: float,
) -> dict[str, torch.Tensor]:
    """Per-example implicit rewards and diagnostics (detached)."""

    device = policy_chosen_logps.device
    pc = policy_chosen_logps.detach().float()
    pr = policy_rejected_logps.detach().float()
    rc = reference_chosen_logps.detach().float().to(device)
    rr = reference_rejected_logps.detach().float().to(device)
    chosen_rewards = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)
    losses = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return {
        "losses": losses,
        "chosen_rewards": chosen_rewards,
        "rejected_rewards": rejected_rewards,
        "margins": chosen_rewards - rejected_rewards,
        # Strict comparison, as in the manual: tied rewards count as incorrect.
        "accuracies": (chosen_rewards > rejected_rewards).float(),
        "policy_chosen_logps": pc,
        "policy_rejected_logps": pr,
    }
