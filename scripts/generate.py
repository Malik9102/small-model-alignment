"""Generate responses for a fixed evaluation set and save them as JSON.

Each saved entry has at least: id, instruction, output, generator, dataset
(+ category, finish_reason, num_new_tokens). A sidecar ``<output>.meta.json``
records the exact decoding settings so SFT and DPO are provably compared with
the same decoding policy.

Examples:
    # sanity check on 5 prompts (prints them)
    python scripts/generate.py --model_path outputs/sft/best --eval_set eval_sets/alpaca_eval_300_seed42.json \
        --generator qwen2.5-0.5b-sft --output outputs/generations/sanity_sft.json --limit 5 --print_samples

    # base model baseline (Task 2)
    python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --eval_set eval_sets/alpaca_eval_300_seed42.json \
        --generator qwen2.5-0.5b-base --output outputs/generations/base_alpaca.json

    # SmolLM2 persona model (chat template + LoRA adapter)
    python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path outputs/lora_persona/adapter \
        --prompt_format chat --eval_set eval_sets/persona_style_eval.json --generator smollm2-360m-lora-dpo \
        --output outputs/generations/lora_persona.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from alignment.prompting import STOP_STRINGS, format_prompt, strip_stop_strings  # noqa: E402
from alignment.train_common import dtype_kwarg  # noqa: E402
from alignment.utils import read_json, set_seed, write_json  # noqa: E402

DEFAULTS = {
    "batch_size": 16,
    "max_new_tokens": 512,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 0,
    "dtype": "float32",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT/LoRA adapter to apply and merge")
    parser.add_argument("--eval_set", required=True, help="JSON list with id + instruction")
    parser.add_argument("--generator", required=True, help="Name stored in every output row")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt_format", choices=["plain", "chat"], default="plain")
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--max_new_tokens", type=int, default=DEFAULTS["max_new_tokens"])
    parser.add_argument("--repetition_penalty", type=float, default=DEFAULTS["repetition_penalty"])
    parser.add_argument("--no_repeat_ngram_size", type=int, default=DEFAULTS["no_repeat_ngram_size"])
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default=DEFAULTS["dtype"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N prompts (sanity checks)")
    parser.add_argument("--print_samples", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model_and_tokenizer(model_path, adapter_path, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_source = adapter_path if adapter_path and (Path(adapter_path) / "tokenizer_config.json").exists() else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.padding_side = "left"  # decoder-only batched generation needs left padding
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(model_path, **{dtype_kwarg(): torch_dtype})
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.to(device)
    model.eval()
    return model, tokenizer


def build_prompt(tokenizer, instruction: str, prompt_format: str) -> str:
    if prompt_format == "plain":
        return format_prompt(instruction)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False, add_generation_prompt=True
    )


def stop_token_ids(tokenizer, prompt_format: str) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if prompt_format == "chat":
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end != tokenizer.unk_token_id and im_end not in ids:
            ids.append(im_end)
    return ids


@torch.no_grad()
def generate_all(model, tokenizer, rows, args, device):
    prompts = [build_prompt(tokenizer, row["instruction"], args.prompt_format) for row in rows]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]), reverse=True)
    eos_ids = stop_token_ids(tokenizer, args.prompt_format)
    outputs = [None] * len(rows)
    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if args.no_repeat_ngram_size:
        gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    if args.prompt_format == "plain":
        gen_kwargs.update(stop_strings=STOP_STRINGS, tokenizer=tokenizer)
    start = time.time()
    for batch_start in range(0, len(order), args.batch_size):
        indices = order[batch_start : batch_start + args.batch_size]
        encoded = tokenizer(
            [prompts[i] for i in indices], return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        generated = model.generate(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], **gen_kwargs
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        for row_index, tokens in zip(indices, new_tokens):
            token_list = tokens.tolist()
            terminators = set(eos_ids) | {tokenizer.pad_token_id}
            cut = next((i for i, token in enumerate(token_list) if token in terminators), None)
            content = token_list if cut is None else token_list[:cut]
            text = tokenizer.decode(content, skip_special_tokens=True)
            stripped = strip_stop_strings(text) if args.prompt_format == "plain" else text.strip()
            if stripped != text.strip():
                finish = "stop_string"  # model started a new "### Instruction:" turn
            elif cut is not None:
                finish = "eos"
            elif len(content) >= args.max_new_tokens:
                finish = "length"
            else:
                finish = "stop"
            outputs[row_index] = {"output": stripped, "finish_reason": finish, "num_new_tokens": len(content)}
        done = min(batch_start + args.batch_size, len(order))
        print(f"  generated {done}/{len(order)} ({time.time() - start:.0f}s)", flush=True)
    return outputs


def main():
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"[generate] {output_path} exists - skipping (use --overwrite to regenerate)")
        return
    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rows = read_json(args.eval_set)
    if args.limit:
        rows = rows[: args.limit]
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path, args.dtype, device)
    print(f"[generate] {args.generator}: {len(rows)} prompts from {args.eval_set} on {device}")
    generations = generate_all(model, tokenizer, rows, args, device)

    results = []
    for row, generation in zip(rows, generations):
        entry = {
            "id": row["id"],
            "instruction": row["instruction"],
            "output": generation["output"],
            "generator": args.generator,
            "dataset": row.get("dataset", Path(args.eval_set).stem),
        }
        for key in ("category", "expected_behavior"):
            if key in row:
                entry[key] = row[key]
        entry["finish_reason"] = generation["finish_reason"]
        entry["num_new_tokens"] = generation["num_new_tokens"]
        results.append(entry)
    write_json(output_path, results)
    meta = {
        "generator": args.generator,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "eval_set": args.eval_set,
        "num_prompts": len(rows),
        "prompt_format": args.prompt_format,
        "decoding": {
            "strategy": "greedy",
            "temperature": None,
            "top_p": None,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": None,  # only affects beam search; not used with greedy decoding
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "stop_strings": STOP_STRINGS if args.prompt_format == "plain" else None,
            "dtype": args.dtype,
        },
    }
    write_json(output_path.with_suffix(".meta.json"), meta)
    finish_counts = {}
    for result in results:
        finish_counts[result["finish_reason"]] = finish_counts.get(result["finish_reason"], 0) + 1
    print(f"[generate] wrote {len(results)} rows -> {output_path} | finish reasons {finish_counts}")
    if args.print_samples:
        for result in results:
            print("=" * 80)
            print(f"[{result['id']}] {result['instruction']}\n---\n{result['output']}\n(finish={result['finish_reason']}, tokens={result['num_new_tokens']})")


if __name__ == "__main__":
    main()
