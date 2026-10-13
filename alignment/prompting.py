"""Single source of truth for the prompt format used by SFT, DPO and evaluation.

Qwen/Qwen2.5-0.5B is a *base* model, so we use a plain-text instruction template
instead of the ChatML chat template. The response is terminated by the
tokenizer's EOS token (``<|endoftext|>`` for Qwen2.5 base), which is exactly the
token the submission-adapter contract appends. Using ChatML would end assistant
turns with ``<|im_end|>`` and create a mismatch with that appended EOS.

The template ends with a newline, so the prompt/response token boundary is clean:
we tokenize prompt and response separately during training, which is exactly how
the prompt is tokenized at generation time.
"""

from __future__ import annotations

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# Strings that mean "the model started a new turn"; generation stops on them and
# they are stripped from outputs during post-processing.
STOP_STRINGS = ["### Instruction:", "\n### Response:"]


def format_prompt(instruction: str) -> str:
    """Wrap a raw user instruction in the training/evaluation template."""

    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def strip_stop_strings(text: str, stop_strings=STOP_STRINGS) -> str:
    """Cut generated text at the first stop string and trim whitespace."""

    cut = len(text)
    for stop in stop_strings:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()
