"""Dataset parsing, filtering and deterministic subsampling.

All sources are converted to single-turn records:

* SFT:  ``{"id", "instruction", "response", "source"}``
* DPO:  ``{"id", "instruction", "chosen", "rejected", "source", ...}``

Sources (Hugging Face Hub):
* SFT            HuggingFaceH4/ultrachat_200k      train_sft -> train, test_sft -> validation
* HH-RLHF DPO    Anthropic/hh-rlhf                 train     -> train, test     -> validation
* UltraFeedback  HuggingFaceH4/ultrafeedback_binarized  train_prefs -> train, test_prefs -> validation
* AlpacaEval     tatsu-lab/alpaca_eval (805 prompts) -> fixed 300-prompt subset, seed 42

Filtering (applied identically to every run, documented in the report):
* single-turn only (first user turn / final assistant turn of a one-turn dialogue)
* empty prompts/responses removed; identical chosen/rejected removed
* UltraFeedback ties (score_chosen == score_rejected) removed
* formatted prompt <= ``max_prompt_tokens`` tokens
* SFT: prompt + response + EOS <= ``max_length`` (only complete demonstrations,
  so the model always sees where a response ends)
* DPO: responses are truncated at train time to fit ``max_length``; pairs whose
  chosen and rejected token windows become identical are removed
* exact-match overlap with evaluation prompts removed (contamination guard)
"""

from __future__ import annotations

import random
import re
from typing import Callable, Iterable, Iterator

from alignment.batching import encode_many, get_eos_id
from alignment.prompting import format_prompt

HH_HUMAN = "\n\nHuman:"
HH_ASSISTANT = "\n\nAssistant:"


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


# ---------------------------------------------------------------------------
# Per-source parsers (pure functions, unit tested)
# ---------------------------------------------------------------------------

def split_hh_transcript(text: str):
    """Split an HH transcript into (everything before the last assistant turn, last response)."""

    index = text.rfind(HH_ASSISTANT)
    if index == -1:
        return None
    return text[:index], text[index + len(HH_ASSISTANT) :].strip()


def parse_hh_pair(chosen_text: str, rejected_text: str):
    """Return ``{"instruction", "chosen", "rejected"}`` for single-turn HH pairs, else None."""

    chosen_split = split_hh_transcript(chosen_text)
    rejected_split = split_hh_transcript(rejected_text)
    if chosen_split is None or rejected_split is None:
        return None
    prompt_chosen, chosen = chosen_split
    prompt_rejected, rejected = rejected_split
    if prompt_chosen.strip() != prompt_rejected.strip():
        return None
    prompt = prompt_chosen
    if prompt.count(HH_HUMAN) != 1 or HH_ASSISTANT in prompt:
        return None  # multi-turn conversation
    instruction = prompt.split(HH_HUMAN, 1)[1].strip()
    if not instruction or not chosen or not rejected or chosen == rejected:
        return None
    return {"instruction": instruction, "chosen": chosen, "rejected": rejected}


def _single_turn_messages(messages):
    if not isinstance(messages, list) or len(messages) != 2:
        return None
    user, assistant = messages
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    return (user.get("content") or "").strip(), (assistant.get("content") or "").strip()


def parse_ultrafeedback_pair(row: dict):
    chosen = _single_turn_messages(row.get("chosen"))
    rejected = _single_turn_messages(row.get("rejected"))
    if chosen is None or rejected is None:
        return None
    instruction, chosen_text = chosen
    instruction_rejected, rejected_text = rejected
    if instruction != instruction_rejected or not instruction:
        return None
    if not chosen_text or not rejected_text or chosen_text == rejected_text:
        return None
    score_chosen, score_rejected = row.get("score_chosen"), row.get("score_rejected")
    if score_chosen is not None and score_rejected is not None and not score_chosen > score_rejected:
        return None  # drop ties (no preference signal)
    return {
        "instruction": instruction,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "score_chosen": score_chosen,
        "score_rejected": score_rejected,
    }


def parse_ultrachat_example(row: dict):
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None
    user, assistant = messages[0], messages[1]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    instruction = (user.get("content") or "").strip()
    response = (assistant.get("content") or "").strip()
    if not instruction or not response:
        return None
    return {"instruction": instruction, "response": response}


# ---------------------------------------------------------------------------
# Deterministic sampling with token-length filters
# ---------------------------------------------------------------------------

def _shuffled_indices(n: int, seed: int) -> list[int]:
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    return indices


def _chunks(items: list, size: int) -> Iterator[list]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def sample_sft_examples(
    rows,
    tokenizer,
    n: int,
    seed: int,
    max_length: int,
    max_prompt_tokens: int,
    blocked_instructions: set[str] | None = None,
    id_prefix: str = "sft",
    source: str = "",
    chunk_size: int = 2048,
):
    blocked = blocked_instructions or set()
    stats = {"seen": 0, "parse_fail": 0, "blocked": 0, "duplicate": 0, "prompt_too_long": 0, "too_long": 0, "kept": 0}
    eos_extra = 1 if get_eos_id(tokenizer) is not None else 0
    seen_instructions: set[str] = set()
    kept = []
    for chunk in _chunks(_shuffled_indices(len(rows), seed), chunk_size):
        parsed = []
        for index in chunk:
            stats["seen"] += 1
            example = parse_ultrachat_example(rows[index])
            if example is None:
                stats["parse_fail"] += 1
                continue
            key = normalize_text(example["instruction"])
            if key in blocked:
                stats["blocked"] += 1
                continue
            if key in seen_instructions:
                stats["duplicate"] += 1
                continue
            seen_instructions.add(key)
            parsed.append((index, example))
        if not parsed:
            continue
        prompt_lengths = [len(ids) for ids in encode_many(tokenizer, [format_prompt(e["instruction"]) for _, e in parsed])]
        response_lengths = [len(ids) for ids in encode_many(tokenizer, [e["response"] for _, e in parsed])]
        for (index, example), p_len, r_len in zip(parsed, prompt_lengths, response_lengths):
            if p_len > max_prompt_tokens:
                stats["prompt_too_long"] += 1
                continue
            if p_len + r_len + eos_extra > max_length:
                stats["too_long"] += 1
                continue
            kept.append({"id": f"{id_prefix}-{len(kept):06d}", "source": source, "source_index": index, **example})
            if len(kept) >= n:
                stats["kept"] = len(kept)
                return kept, stats
    stats["kept"] = len(kept)
    return kept, stats


def sample_preference_examples(
    rows,
    parser: Callable[[dict], dict | None],
    tokenizer,
    n: int,
    seed: int,
    max_length: int,
    max_prompt_tokens: int,
    blocked_instructions: set[str] | None = None,
    id_prefix: str = "pref",
    source: str = "",
    chunk_size: int = 2048,
):
    blocked = blocked_instructions or set()
    stats = {"seen": 0, "parse_fail": 0, "blocked": 0, "prompt_too_long": 0, "identical_after_truncation": 0, "kept": 0}
    eos_id = get_eos_id(tokenizer)
    kept = []
    for chunk in _chunks(_shuffled_indices(len(rows), seed), chunk_size):
        parsed = []
        for index in chunk:
            stats["seen"] += 1
            example = parser(rows[index])
            if example is None:
                stats["parse_fail"] += 1
                continue
            if normalize_text(example["instruction"]) in blocked:
                stats["blocked"] += 1
                continue
            parsed.append((index, example))
        if not parsed:
            continue
        prompts = encode_many(tokenizer, [format_prompt(e["instruction"]) for _, e in parsed])
        chosen = encode_many(tokenizer, [e["chosen"] for _, e in parsed])
        rejected = encode_many(tokenizer, [e["rejected"] for _, e in parsed])
        for (index, example), p_ids, c_ids, r_ids in zip(parsed, prompts, chosen, rejected):
            if len(p_ids) > max_prompt_tokens:
                stats["prompt_too_long"] += 1
                continue
            budget = max_length - len(p_ids)
            tail = [eos_id] if eos_id is not None else []
            if (c_ids + tail)[:budget] == (r_ids + tail)[:budget]:
                stats["identical_after_truncation"] += 1
                continue
            kept.append({"id": f"{id_prefix}-{len(kept):06d}", "source": source, "source_index": index, **example})
            if len(kept) >= n:
                stats["kept"] = len(kept)
                return kept, stats
    stats["kept"] = len(kept)
    return kept, stats


# ---------------------------------------------------------------------------
# AlpacaEval subset
# ---------------------------------------------------------------------------

def load_alpaca_eval_rows() -> list[dict]:
    """Load the 805 AlpacaEval prompts (tries the raw JSON file first)."""

    errors = []
    try:
        from huggingface_hub import hf_hub_download
        import json

        path = hf_hub_download("tatsu-lab/alpaca_eval", "alpaca_eval.json", repo_type="dataset")
        with open(path, "r", encoding="utf-8") as handle:
            return list(json.load(handle))
    except Exception as error:  # pragma: no cover - network dependent
        errors.append(f"hf_hub_download: {error}")
    from datasets import load_dataset

    for kwargs in ({"trust_remote_code": True}, {}):
        try:
            dataset = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", **kwargs)["eval"]
            return [dict(row) for row in dataset]
        except Exception as error:  # pragma: no cover - network dependent
            errors.append(f"load_dataset{kwargs}: {error}")
    raise RuntimeError("Could not load AlpacaEval:\n" + "\n".join(errors))


def make_alpaca_eval_subset(rows: Iterable[dict], n: int = 300, seed: int = 42) -> list[dict]:
    rows = list(rows)
    indices = random.Random(seed).sample(range(len(rows)), n)
    subset = []
    for position, index in enumerate(indices):
        row = rows[index]
        subset.append(
            {
                "id": f"alpaca-eval-{seed}-{position:04d}",
                "instruction": row["instruction"],
                "dataset": "alpaca_eval",
                "category": row.get("dataset", ""),
                "source_index": index,
            }
        )
    return subset
