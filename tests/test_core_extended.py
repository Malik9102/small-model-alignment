"""Extra correctness checks beyond the public smoke tests.

Covers the edge cases the manual warns about: truncation boundaries, unusual
tokenizer configurations, mixed lengths, row ordering, exact log-prob values,
DPO math, optimisation steps, data parsing, and judge parsing / retries.

Run:  PYTHONPATH=. python -m pytest -q tests/
"""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

import submission_adapter as adapter
from alignment.batching import IGNORE_INDEX
from alignment.data import parse_hh_pair, parse_ultrafeedback_pair
from alignment.judging import (
    MockJudge,
    build_judge_prompt,
    build_pairs,
    parse_judge_response,
    run_judging,
    summarize_judgments,
)
from alignment.logprobs import response_logps
from alignment.utils import read_jsonl, write_jsonl
from tests.toy_lm import ToyCausalLM, ToyTokenizer


# ---------------------------------------------------------------------------
# Tokenizer variants
# ---------------------------------------------------------------------------

class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None  # no pad id: implementation must fall back (to eos)


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class LeftPadTokenizer(ToyTokenizer):
    padding_side = "left"


class QwenLikeTokenizer(ToyTokenizer):
    """pad token id == eos token id, like Qwen2.5 (<|endoftext|>)."""

    pad_token_id = 1


def expected_logp(model, tokenizer, prompt, response, max_length):
    """Brute force: the toy LM's distribution is position independent."""

    logp = F.log_softmax(model.logit_bias.detach().float(), dim=-1)
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response)
    if tokenizer.eos_token_id is not None:
        response_ids = response_ids + [tokenizer.eos_token_id]
    sequence = (prompt_ids + response_ids)[:max_length]
    total = 0.0
    for position in range(max(len(prompt_ids), 1), len(sequence)):
        total += float(logp[sequence[position]])
    return total


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_eos_survives_exact_fit_and_is_dropped_when_truncated():
    tok = ToyTokenizer()
    prompt, response = "ab", "cd"  # 2 + 2 + eos = 5 tokens
    fit = adapter.build_sft_batch(tok, [prompt], [response], max_length=5)
    assert fit["labels"][0, 4] == tok.eos_token_id
    assert fit["input_ids"].shape == (1, 5)
    cut = adapter.build_sft_batch(tok, [prompt], [response], max_length=4)
    assert tok.eos_token_id not in cut["labels"][0].tolist()
    assert cut["labels"][0].tolist() == [IGNORE_INDEX, IGNORE_INDEX] + tok.encode("cd")


def test_prompt_longer_than_max_length_leaves_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["abcdefghij"], ["xyz"], max_length=6)
    assert batch["labels"].eq(IGNORE_INDEX).all()
    assert batch["attention_mask"].sum() == 6
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefghij"], ["xyz"], max_length=6)
    assert logps.tolist() == [0.0]
    loss = adapter.compute_sft_loss(model, tok, ["abcdefghij"], ["xyz"], max_length=6)
    assert torch.isfinite(loss)


def test_padding_is_masked_and_rows_padded_to_max_length():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["b", "defgh"], max_length=12)
    assert batch["input_ids"].shape == (2, 12)
    for row, (prompt, response) in enumerate([("a", "b"), ("abc", "defgh")]):
        real = len(prompt) + len(response) + 1
        assert batch["attention_mask"][row, :real].eq(1).all()
        assert batch["attention_mask"][row, real:].eq(0).all()
        assert batch["labels"][row, real:].eq(IGNORE_INDEX).all()
        assert batch["input_ids"][row, real:].eq(tok.pad_token_id).all()


@pytest.mark.parametrize("tokenizer_cls", [NoPadTokenizer, LeftPadTokenizer, QwenLikeTokenizer])
def test_unusual_tokenizers_keep_eos_label_and_mask_padding(tokenizer_cls):
    tok = tokenizer_cls()
    batch = adapter.build_sft_batch(tok, ["Q: ", "Question: "], ["yes", "no"], max_length=16)
    for row, (prompt, response) in enumerate([("Q: ", "yes"), ("Question: ", "no")]):
        end = len(prompt) + len(response) + 1
        assert batch["labels"][row, end - 1] == tok.eos_token_id  # EOS kept even if eos == pad
        assert batch["labels"][row, :len(prompt)].eq(IGNORE_INDEX).all()
        assert batch["labels"][row, end:].eq(IGNORE_INDEX).all()
        assert batch["attention_mask"][row, :end].eq(1).all()  # right padding regardless of padding_side
        assert batch["attention_mask"][row, end:].eq(0).all()


def test_tokenizer_without_eos_appends_nothing():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["Q: "], ["yes"], max_length=10)
    labels = batch["labels"][0]
    assert labels[labels.ne(IGNORE_INDEX)].tolist() == tok.encode("yes")


def test_dpo_batch_shares_prompt_masking():
    tok = ToyTokenizer()
    batch = adapter.build_dpo_batch(tok, ["Pick: ", "X"], ["good", "a"], ["bad answer", "bb"], max_length=20)
    for row, prompt in enumerate(["Pick: ", "X"]):
        n = len(prompt)
        for side in ("chosen", "rejected"):
            assert batch[side]["labels"][row, :n].eq(IGNORE_INDEX).all()
            assert batch[side]["input_ids"][row, :n].tolist() == tok.encode(prompt)


# ---------------------------------------------------------------------------
# Log probabilities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_length", [4, 7, 12, 30])
def test_logprobs_match_brute_force(max_length):
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.3)
    prompts = ["Hi: ", "", "longer prompt: ", "a"]
    responses = ["answer", "first token unscored", "x", "mixed length response"]
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length)
    expected = [expected_logp(model, tok, p, r, max_length) for p, r in zip(prompts, responses)]
    torch.testing.assert_close(logps.detach(), torch.tensor(expected), rtol=1e-5, atol=1e-5)


def test_row_order_is_preserved():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts = ["a: ", "bb: ", "ccc: "]
    responses = ["x", "yy yy", "zzz zzz zzz"]
    forward = adapter.compute_response_logprobs(model, tok, prompts, responses, 32)
    backward = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], 32)
    torch.testing.assert_close(forward, backward.flip(0))


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    prompts, responses = ["Do: ", "Say: "], ["a", "longer"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, 16)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, 16)
    tokens = adapter.build_sft_batch(tok, prompts, responses, 16)["labels"][:, 1:].ne(IGNORE_INDEX).sum()
    torch.testing.assert_close(loss, -logps.sum() / tokens)


def test_sft_step_reduces_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = adapter.compute_sft_loss(model, tok, ["Q: "], ["abc"], 16)
    before.backward()
    optimizer.step()
    after = adapter.compute_sft_loss(model, tok, ["Q: "], ["abc"], 16)
    assert after < before


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------

def test_dpo_loss_matches_formula_and_beta_scaling():
    pc, pr = torch.tensor([-1.0, -4.0, -2.0]), torch.tensor([-3.0, -1.0, -2.5])
    rc, rr = torch.tensor([-1.5, -3.0, -2.0]), torch.tensor([-2.0, -2.0, -2.0])
    for beta in (0.05, 0.1, 1.0):
        expected = -torch.log(torch.sigmoid(beta * ((pc - pr) - (rc - rr)))).mean()
        torch.testing.assert_close(adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_outputs_keep_input_precision(dtype):
    values = torch.tensor([-2.0, -5.0], dtype=dtype)
    loss = adapter.compute_dpo_loss_from_logps(values, values, values, values, 0.1)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0, dtype=dtype)))
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5).to(dtype)
    assert adapter.compute_response_logprobs(model, tok, ["a: "], ["bc"], 8).dtype == dtype
    assert adapter.compute_sft_loss(model, tok, ["a: "], ["bc"], 8).dtype == dtype
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.5).to(dtype)
    assert adapter.compute_dpo_loss(model, reference, tok, ["a: "], ["b"], ["c"], 0.1, 8).dtype == dtype


def test_half_precision_logps_are_upcast():
    values = torch.tensor([-2.0, -5.0], dtype=torch.float16)
    assert adapter.compute_dpo_loss_from_logps(values, values, values, values, 0.1).dtype == torch.float32


def test_dpo_loss_is_stable_for_extreme_logits():
    big = torch.tensor([1e4])
    loss = adapter.compute_dpo_loss_from_logps(-big, torch.zeros(1), torch.zeros(1), torch.zeros(1), 1.0)
    assert torch.isfinite(loss) and loss > 1e3


def test_dpo_end_to_end_matches_logps_composition_and_step_improves():
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    reference = ToyCausalLM(tok.vocab_size, bias_scale=0.3)
    prompts, chosen, rejected = ["Pick: ", "Q: "], ["aaa", "yes"], ["zzz", "no"]
    loss = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, 0.1, 16)
    manual = adapter.compute_dpo_loss_from_logps(
        adapter.compute_response_logprobs(policy, tok, prompts, chosen, 16),
        adapter.compute_response_logprobs(policy, tok, prompts, rejected, 16),
        adapter.compute_response_logprobs(reference, tok, prompts, chosen, 16).detach(),
        adapter.compute_response_logprobs(reference, tok, prompts, rejected, 16).detach(),
        0.1,
    )
    torch.testing.assert_close(loss, manual)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))  # identical models -> ln 2
    loss.backward()
    assert reference.logit_bias.grad is None
    with torch.no_grad():
        policy.logit_bias -= 5.0 * policy.logit_bias.grad
    after = adapter.compute_dpo_loss(policy, reference, tok, prompts, chosen, rejected, 0.1, 16)
    assert after < loss


# ---------------------------------------------------------------------------
# Fast HF log-prob path (tiny Qwen2 with pad == eos)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory):
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from tests.tiny_hf import build_tiny_model

    path = build_tiny_model(tmp_path_factory.mktemp("tiny") / "qwen2")
    return AutoModelForCausalLM.from_pretrained(path), AutoTokenizer.from_pretrained(path)


def test_fast_path_matches_generic_on_hf_model(tiny_qwen):
    model, tok = tiny_qwen
    assert tok.pad_token_id == tok.eos_token_id
    prompts = ["### Instruction:\nhi\n\n### Response:\n", "Q: "]
    responses = ["Sunlight scatters.", "a much longer answer with more tokens"]
    batch = adapter.build_sft_batch(tok, prompts, responses, max_length=40)
    assert batch["labels"][0][batch["labels"][0].ne(IGNORE_INDEX)][-1] == tok.eos_token_id
    generic, counts_g = response_logps(model, batch, fast=False)
    fast, counts_f = response_logps(model, batch, fast=True)
    torch.testing.assert_close(generic, fast, rtol=1e-4, atol=1e-4)
    assert counts_g.tolist() == counts_f.tolist()


def test_reference_scoring_is_deterministic_even_if_left_in_train_mode(tiny_qwen):
    from transformers import AutoModelForCausalLM

    model, tok = tiny_qwen
    path = model.name_or_path
    reference = AutoModelForCausalLM.from_pretrained(path, attention_dropout=0.5)
    reference.train()
    model.eval()
    losses = {float(adapter.compute_dpo_loss(model, reference, tok, ["Q: hi"], ["yes"], ["no"], 0.1, 32)) for _ in range(3)}
    assert len(losses) == 1
    assert reference.training  # caller's mode restored


# ---------------------------------------------------------------------------
# Data parsing
# ---------------------------------------------------------------------------

def test_hh_parser_keeps_single_turn_and_rejects_multi_turn():
    single_c = "\n\nHuman: How do I bake bread?\n\nAssistant: Mix flour, water, yeast."
    single_r = "\n\nHuman: How do I bake bread?\n\nAssistant: No idea."
    parsed = parse_hh_pair(single_c, single_r)
    assert parsed == {"instruction": "How do I bake bread?", "chosen": "Mix flour, water, yeast.", "rejected": "No idea."}
    multi = "\n\nHuman: hi\n\nAssistant: hello\n\nHuman: bread?\n\nAssistant: "
    assert parse_hh_pair(multi + "yes", multi + "no") is None
    assert parse_hh_pair(single_c, "\n\nHuman: other\n\nAssistant: x") is None
    assert parse_hh_pair(single_c, single_c) is None


def test_ultrafeedback_parser_drops_ties_and_multi_turn():
    row = {
        "chosen": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "good"}],
        "rejected": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "bad"}],
        "score_chosen": 8.0,
        "score_rejected": 4.0,
    }
    assert parse_ultrafeedback_pair(row)["chosen"] == "good"
    assert parse_ultrafeedback_pair({**row, "score_rejected": 8.0}) is None
    multi = row["chosen"] + [{"role": "user", "content": "more"}, {"role": "assistant", "content": "x"}]
    assert parse_ultrafeedback_pair({**row, "chosen": multi}) is None


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

def _generations(name, outputs):
    return [{"id": f"p{i}", "instruction": f"q{i}", "output": out, "generator": name, "category": "c"} for i, out in enumerate(outputs)]


def test_pairs_are_blinded_and_balanced():
    cand = _generations("dpo-secret-name", ["long answer"] * 40)
    base = _generations("sft-secret-name", ["short"] * 40)
    pairs, missing = build_pairs(cand, base, "dpo-secret-name", "sft-secret-name", seed=1)
    assert not missing
    positions = [p["hidden"]["candidate_position"] for p in pairs]
    assert 10 < positions.count("A") < 30
    prompt = build_judge_prompt(pairs[:5], "general")
    assert "secret" not in prompt and "dpo" not in prompt.lower() and "sft" not in prompt.lower()


def test_parse_judge_response_variants():
    text = '```json\n[{"comparison": 1, "winner": "Response A"}, {"comparison": 2, "winner": "TIE"}, {"comparison": 3, "winner": "C"}]\n```'
    parsed = parse_judge_response(text, 3)
    assert parsed[1]["winner"] == "A" and parsed[2]["winner"] == "tie" and 3 not in parsed
    assert parse_judge_response("no json here", 2) == {}


def test_run_judging_retries_logs_failures_and_resumes(tmp_path):
    cand = _generations("dpo", ["a much longer answer"] * 7)
    base = _generations("sft", ["short"] * 7)
    pairs, _ = build_pairs(cand, base, "dpo", "sft", seed=0)
    pairs_path, out_path = tmp_path / "pairs.jsonl", tmp_path / "judged.jsonl"
    write_jsonl(pairs_path, pairs)

    flaky = MockJudge(fail_first=1)
    summary = run_judging(pairs_path, out_path, batch_size=5, sleep_seconds=0, max_retries=3, backend="mock", judge=flaky)
    assert summary["overall"]["wins"] == 7 and summary["overall"]["failed"] == 0
    assert flaky.calls == 3  # 1 failed + 1 success for batch 1, 1 success for batch 2

    always_down = MockJudge(fail_first=10**6)
    out2 = tmp_path / "judged2.jsonl"
    summary = run_judging(pairs_path, out2, batch_size=5, sleep_seconds=0, max_retries=2, backend="mock", judge=always_down)
    assert summary["overall"]["failed"] == 7
    assert len(read_jsonl(tmp_path / "judged2_failures.jsonl")) == 7

    resumed_judge = MockJudge()
    run_judging(pairs_path, out_path, batch_size=5, sleep_seconds=0, reuse_judgments=True, backend="mock", judge=resumed_judge)
    assert resumed_judge.calls == 0  # everything reused


def test_summary_rates():
    records = [
        {"candidate_outcome": o, "status": "ok" if o in ("win", "loss", "tie") else o, "verdict": v, "candidate": "d", "baseline": "s", "category": "c", "attempts": 1}
        for o, v in [("win", "A"), ("win", "B"), ("loss", "A"), ("tie", "tie"), ("unknown", None)]
    ]
    s = summarize_judgments(records)["overall"]
    assert s["wins"] == 2 and s["losses"] == 1 and s["ties"] == 1 and s["unknown"] == 1
    assert s["win_rate_excluding_ties"] == pytest.approx(2 / 3)
    assert s["win_rate_ties_half"] == pytest.approx(2.5 / 4)
    json.dumps(s)
