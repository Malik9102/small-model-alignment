"""Tokenization and batch construction with response-only label masking.

Conventions (shared by SFT, DPO and the submission adapter):

* Prompt and response are tokenized *separately* with ``add_special_tokens=False``
  and concatenated, so no BOS/EOS is inserted in the middle of a sequence and the
  prompt token count is exactly ``len(encode(prompt))``.
* The EOS token is appended to the response *before* truncation (when the
  tokenizer has one). Truncation keeps the first ``max_length`` tokens, so an
  over-long response loses its EOS, and an over-long prompt can leave zero
  response tokens.
* Padding is always on the right and is done manually. The tokenizer's own
  ``padding_side`` is therefore irrelevant, and positions start at 0 for every row.
* ``labels`` are *not* shifted (Hugging Face convention): ``labels[i, t]`` is the
  token at position ``t`` if it is a response token, else ``-100``. The model
  scores ``labels[:, 1:]`` with ``logits[:, :-1]``.
* Masks are built from known lengths, never from token-id equality. This matters
  for Qwen, whose pad token *is* its EOS token (``<|endoftext|>``).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------

def get_eos_id(tokenizer):
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    return eos_id


def get_pad_id(tokenizer) -> int:
    """Pad id with fallbacks: pad -> eos -> 0. The value never affects labels."""

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = get_eos_id(tokenizer)
    if pad_id is None:
        pad_id = 0
    return int(pad_id)


def _to_id_list(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    value = list(value)
    if value and isinstance(value[0], (list, tuple)):
        if len(value) != 1:
            raise ValueError("Expected a single sequence of token ids")
        value = list(value[0])
    return [int(token_id) for token_id in value]


def encode_text(tokenizer, text: str) -> list[int]:
    """Encode text without any special tokens (no BOS, no EOS)."""

    if text is None:
        text = ""
    if text == "":
        return []
    if hasattr(tokenizer, "encode"):
        try:
            return _to_id_list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            pass
    return _to_id_list(tokenizer(text, add_special_tokens=False))


def encode_many(tokenizer, texts: Sequence[str]) -> list[list[int]]:
    """Batched variant used by data preparation (fast for HF fast tokenizers)."""

    texts = list(texts)
    try:
        output = tokenizer(texts, add_special_tokens=False)
        ids = output["input_ids"] if isinstance(output, dict) or hasattr(output, "keys") else output.input_ids
        if len(ids) == len(texts) and all(isinstance(row, list) for row in ids):
            return [[int(token) for token in row] if text else [] for row, text in zip(ids, texts)]
    except Exception:  # pragma: no cover - fall back to one-by-one encoding
        pass
    return [encode_text(tokenizer, text) for text in texts]


# ---------------------------------------------------------------------------
# Example / batch construction
# ---------------------------------------------------------------------------

def build_example_ids(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    eos_id,
    max_length: int,
    max_prompt_length: int | None = None,
):
    """Return ``(input_ids, labels)`` for one prompt/response pair (unpadded).

    ``max_prompt_length`` (optional, used by DPO training only) keeps the *last*
    tokens of an over-long prompt, the part closest to the response. The adapter
    contract does not use it.
    """

    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[len(prompt_ids) - max_prompt_length :] if max_prompt_length > 0 else []
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]

    input_ids = (prompt_ids + response_ids)[: max(int(max_length), 0)]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[: len(input_ids)]
    return input_ids, labels


def collate_examples(
    examples: Iterable[tuple[list[int], list[int]]],
    pad_id: int,
    max_length: int,
    pad_to: str = "max_length",
    pad_to_multiple_of: int | None = None,
) -> dict[str, torch.Tensor]:
    """Right-pad ``(input_ids, labels)`` pairs into tensors.

    ``pad_to="max_length"`` pads every row to ``max_length`` (adapter default);
    ``pad_to="longest"`` pads to the longest row (used during training for speed).
    """

    examples = list(examples)
    if pad_to == "max_length":
        width = int(max_length)
    elif pad_to == "longest":
        width = max((len(ids) for ids, _ in examples), default=0)
        if pad_to_multiple_of:
            width = -(-width // pad_to_multiple_of) * pad_to_multiple_of
        width = min(width, int(max_length)) if max_length else width
        width = max(width, 1)
    else:
        raise ValueError(f"Unknown pad_to={pad_to!r}")

    input_ids = torch.full((len(examples), width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, row_labels) in enumerate(examples):
        length = min(len(ids), width)
        if length == 0:
            continue
        input_ids[row, :length] = torch.tensor(ids[:length], dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(row_labels[:length], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    responses: Sequence[str],
    max_length: int,
    pad_to: str = "max_length",
    max_prompt_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Text -> response-masked LM batch (``input_ids``, ``attention_mask``, ``labels``)."""

    prompts = list(prompts)
    responses = list(responses)
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    eos_id = get_eos_id(tokenizer)
    examples = [
        build_example_ids(
            encode_text(tokenizer, prompt),
            encode_text(tokenizer, response),
            eos_id,
            max_length,
            max_prompt_length,
        )
        for prompt, response in zip(prompts, responses)
    ]
    return collate_examples(examples, get_pad_id(tokenizer), max_length, pad_to=pad_to)


def build_preference_batch(
    tokenizer,
    prompts: Sequence[str],
    chosen_responses: Sequence[str],
    rejected_responses: Sequence[str],
    max_length: int,
    pad_to: str = "max_length",
    max_prompt_length: int | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Separate chosen / rejected LM batches built with the identical prompt."""

    prompts = list(prompts)
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen and rejected must have the same length")
    return {
        "chosen": build_lm_batch(tokenizer, prompts, chosen_responses, max_length, pad_to, max_prompt_length),
        "rejected": build_lm_batch(tokenizer, prompts, rejected_responses, max_length, pad_to, max_prompt_length),
    }


def stack_preference_batch(batch: dict[str, dict[str, torch.Tensor]], pad_id: int = 0) -> dict[str, torch.Tensor]:
    """Concatenate chosen rows then rejected rows into one batch (one forward pass).

    Rows ``[0, B)`` are chosen, rows ``[B, 2B)`` are rejected. Widths are equalized
    by right-padding the narrower batch.
    """

    chosen, rejected = batch["chosen"], batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])

    def widen(tensor, fill):
        if tensor.shape[1] == width:
            return tensor
        pad = torch.full((tensor.shape[0], width - tensor.shape[1]), fill, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=1)

    return {
        "input_ids": torch.cat([widen(chosen["input_ids"], pad_id), widen(rejected["input_ids"], pad_id)]),
        "attention_mask": torch.cat([widen(chosen["attention_mask"], 0), widen(rejected["attention_mask"], 0)]),
        "labels": torch.cat([widen(chosen["labels"], IGNORE_INDEX), widen(rejected["labels"], IGNORE_INDEX)]),
    }
