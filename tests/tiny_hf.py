"""Build a tiny *offline* Hugging Face model + tokenizer that mimic Qwen2.5.

Used only by local tests: a byte-level BPE tokenizer where pad == eos ==
``<|endoftext|>`` (exactly like Qwen2.5 base), plus a randomly initialised
2-layer ``Qwen2ForCausalLM``. Lets the full SFT / DPO / generation / LoRA code
paths run on CPU without downloading anything.
"""

from __future__ import annotations

from pathlib import Path

CHATML_TEMPLATE = (
    "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)

CORPUS = [
    "### Instruction:\nExplain why the sky is blue.\n\n### Response:\nSunlight scatters off air molecules.",
    "### Instruction:\nWrite a haiku about rain.\n\n### Response:\nSoft rain on the roof.",
    "Two minutes in the microwave? You donkey! Use the oven and sear it in a hot pan.",
    "It is generally not recommended to microwave steak. Try reheating it slowly.",
    "def add(a, b):\n    return a + b\n# hash passwords with bcrypt, never md5",
    "I cannot help with that request, but here is a safer alternative.",
    "The quick brown fox jumps over the lazy dog. 0123456789 !?.,;:'\"()[]{}",
] * 20


def build_tiny_model(directory, model_type: str = "qwen2", seed: int = 0) -> str:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    directory = Path(directory)
    if (directory / "config.json").exists():
        return str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=400, special_tokens=special, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tokenizer.train_from_iterator(CORPUS, trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast.chat_template = CHATML_TEMPLATE
    fast.save_pretrained(directory)

    if model_type == "qwen2":
        config = AutoConfig.for_model(
            "qwen2",
            vocab_size=len(fast),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
            eos_token_id=fast.eos_token_id,
            pad_token_id=fast.pad_token_id,
        )
    else:  # llama-style stand-in for SmolLM2
        config = AutoConfig.for_model(
            "llama",
            vocab_size=len(fast),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            tie_word_embeddings=True,
            eos_token_id=fast.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=fast.pad_token_id,
        )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(directory)
    return str(directory)


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(build_tiny_model(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-qwen2"))
