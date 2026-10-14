# Small-Model Alignment: SFT + DPO (CS 5326 PA2)

Post-training pipeline for `Qwen/Qwen2.5-0.5B`:
- **Task 3:** supervised fine-tuning, written from scratch.
- **Task 4:** two from-scratch DPO runs, on HH-RLHF and UltraFeedback.
- **Task 5:** a TRL `DPOTrainer` + LoRA persona run on `SmolLM2-360M-Instruct` (the "Idiot Sandwich" task).
- **Tasks 6–7:** blinded pairwise evaluation with a Kaggle model-proxy judge.

No Hugging Face `Trainer`, TRL, or prebuilt DPO loop is used for SFT or the two from-scratch DPO runs. Only Transformers model and tokenizer loading and PyTorch utilities are used there.

## Layout

```text
submission_adapter.py        grading interface (6 functions) -> thin wrappers around alignment/
alignment/
  prompting.py               the single prompt template + stop strings
  batching.py                tokenization, EOS-before-truncation, right padding, -100 masks
  logprobs.py                response-only summed log-probs (generic + fast HF path)
  losses.py                  SFT token-mean loss, DPO loss, implicit rewards / accuracy / margin
  dpo_core.py                policy + frozen-reference scoring (one forward per side)
  sft_trainer.py             SFT loop: grad accumulation, fp16 autocast + GradScaler, clipping, cosine, best/final
  dpo_trainer.py             DPO loop: policy cuda:0, frozen reference cuda:1, cached val reference log-probs
  train_common.py            model loading, optimizers, schedules, pre-tokenized datasets
  data.py                    UltraChat / HH-RLHF / UltraFeedback parsing + filtering, AlpacaEval subset
  judging.py                 pair files, blinded judge prompts, parsing, kbench backend, resumable loop, summaries
  utils.py                   configs, seeding, atomic JSON/JSONL IO, metrics logger, checkpoints
configs/                     sft.yaml, dpo_hh.yaml, dpo_uf.yaml, lora_persona.yaml (guide values);
                             dpo_hh_lr5e6.yaml, lora_persona_beta05.yaml, lora_persona_1epoch.yaml (ablations)
scripts/                     env_check, prepare_data, train_sft, train_dpo, train_lora_dpo, generate,
                             build_pairs, judge_pairwise_kaggle, summarize_judgments, analyze_generations,
                             show_examples, plot_metrics, collect_results, analyze_length_bias
notebooks/                   01 ... 06 Kaggle notebooks (+ build_notebooks.py that generates them)
eval_sets/                   hh_rlhf_eval_300_seed42.json (given), persona_style_eval.json (24 prompts),
                             alpaca_eval_300_seed42.json (created by prepare_data.py)
tests/                       public smoke tests, 32 extended edge-case tests, tiny end-to-end pipeline test
report/report_template.md    report skeleton following manual Section 14 (MLA references)
```

## Key implementation decisions

- **Prompt format.** `### Instruction:\n{instruction}\n\n### Response:\n` followed by the response and `<|endoftext|>`. The same template is used for SFT, DPO and every Qwen generation.
- **Masking.** Prompt and response are tokenized separately with no special tokens. EOS is appended before right-truncation to `max_length`.
  - Prompt and padding positions get label `-100`; labels are unshifted.
  - Masks are built from lengths, never from token ids. Qwen's pad token *is* its EOS token.
  - Adapter batches are right-padded to `max_length`; the tokenizer's `padding_side` is ignored.
- **Log-probs.** `labels[:, 1:]` is scored with `logits[:, :-1]`, in fp32, and summed per row with row order preserved. The training loops use a fast path that applies the LM head only at response positions; tests show it is exactly equivalent.
- **SFT loss.** Mean over all response tokens in the effective batch.
- **Precision.** fp16 autocast with fp32 trainable weights and a GradScaler, because T4s lack bf16. The update order is: unscale, then clip, then step.
- **DPO reference.** A second SFT-best copy on `cuda:1`, kept in fp32 under the same autocast and frozen. At step 0 this gives loss = ln 2 and reward accuracy = 0 exactly.
- **DPO checkpoint selection.** Best checkpoint by validation reward accuracy, with reward margin as the tie-break, plus a final checkpoint.
- **Judging.** The judge sees "Response A/B" only, in random order; the model mapping is kept in `pair["hidden"]`.
  - 5 pairs per request, at least 5 s between attempts, 3 attempts per batch.
  - Failures go to a `*_failures.jsonl` file, and reruns resume from completed judgments.
  - An A/B-swapped subset measures the judge's position bias.

## Running on Kaggle

Push this repo to your GitHub, then put your repo URL into `REPO_URL` in the first cell of each notebook.

Run every training notebook with **Save Version → Save & Run All (Commit)**. Later notebooks attach earlier outputs via **Add Input → Your Work → Notebooks**.

| # | Notebook | Accelerator | Attach outputs of | Time |
|---|---|---|---|---|
| 01 | `01_setup_data_baseline` | T4 x2 | – | ~40 min |
| 02 | `02_train_sft` | T4 x2 | 01 | ~1.5–2 h |
| 03a | `03a_train_dpo_hh` | T4 x2 | 01, 02 | ~2–2.5 h |
| 03b | `03b_train_dpo_uf` | T4 x2 | 01, 02 | ~2–2.5 h |
| 03c | `03c_ablations` (HH lr 5e-6; LoRA β 0.5; LoRA 1 epoch) | T4 x2 | 01, 02 | ~1.5 h |
| 04 | `04_train_lora_persona` | T4 | – | ~20 min |
| 05 | `05_generate_eval_outputs` | T4 x2 | 01, 02, 03a, 03b | ~45 min |
| 06 | `06_judge_and_report_assets` | none (**Benchmark notebook**) | 02, 03a, 03b, 03c, 04, 05 | ~2–3 h |

Every notebook needs Internet ON. Notebook 01 runs all the tests before doing anything expensive. Notebooks 02 and 03 each do a few-step pilot run before the full run.

## After the Kaggle runs: make the repo auditable

The manual (§6.4, §14.7) requires the exact evaluation subsets and the key outputs to be findable in the repository.
1. Unzip `06_final_results.zip` into `outputs/`. Checkpoints are gitignored; metrics, generations, pairs, judgments, plots and tables are kept.
2. Copy `eval_sets/alpaca_eval_300_seed42.json` and `data/processed/manifest.json` from notebook 01's output into the repo.
3. Commit and push.

Notebooks are restartable: a training script skips a run whose `summary.json` already exists, `generate.py` skips existing output files, and the judge reuses completed judgments.

## Local checks (no GPU, no downloads)

```bash
PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py   # public tests
PYTHONPATH=. python -m pytest -q tests/                               # everything (~1 min)
```

## Reproducing the report tables

```bash
python scripts/plot_metrics.py --sft outputs/sft --dpo outputs/dpo_hh outputs/dpo_uf --lora outputs/lora_persona
python scripts/collect_results.py --runs_root outputs --output outputs/report_tables.md
```
