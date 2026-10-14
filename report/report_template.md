# Instruction Tuning and Direct Preference Optimization on a Small Language Model

**Name:** [FILL] · **Roll number:** [FILL] · **Course:** CS 5326 · **Repository:** [FILL: https://github.com/<user>/small-model-alignment]

> How to use this template: every `[FILL]` must be replaced with a number or observation from YOUR runs. The tables marked
> *auto* are produced by `python scripts/collect_results.py --runs_root outputs` (notebook 06 writes them to
> `outputs/report_tables.md`); paste them in. Delete this note and keep the PDF to **≤ 8 pages** including figures and references.

## 1. Repository, models, and data (§14.1)

**Base checkpoints.** `Qwen/Qwen2.5-0.5B` for SFT and both from-scratch DPO runs; `HuggingFaceTB/SmolLM2-360M-Instruct` for the LoRA persona run.

| Track | Source | Train | Validation | Evaluation set |
|---|---|---|---|---|
| SFT | `HuggingFaceH4/ultrachat_200k` (train_sft / test_sft), first user turn + first reply | 30,000 | 1,000 | – |
| HH-RLHF DPO | `Anthropic/hh-rlhf` (train / test), single-turn pairs | 15,000 | 1,000 | `eval_sets/hh_rlhf_eval_300_seed42.json` (300, 10 categories) |
| UltraFeedback DPO | `HuggingFaceH4/ultrafeedback_binarized` (train_prefs / test_prefs) | 15,000 | 1,000 | `eval_sets/alpaca_eval_300_seed42.json` (300 of 805, seed 42) |
| LoRA persona DPO | `data/persona_lora/ramsay_style_preferences.json` | 200 | 20 | `eval_sets/persona_style_eval.json` (24: cooking/coding/general/safety) |

**Seeds.** 42 for data shuffling, subset selection, the AlpacaEval subset, training, and A/B order randomization.

**Filtering** (`data/processed/manifest.json`):
- **All sets:** single-turn only; empty or identical responses removed; formatted prompt ≤ 256 tokens; exact-match overlap with any evaluation prompt removed.
- **SFT:** only complete demonstrations (prompt + response + EOS ≤ 512 tokens), so the model always sees where a response ends. 57,344 examples were examined: 11,130 rejected for long prompts and 16,101 for exceeding 512 tokens. This biases SFT towards shorter responses than raw UltraChat.
- **HH-RLHF:** 49,152 examined; 34,053 rejected as multi-turn or malformed, 58 for long prompts.
- **UltraFeedback:** 22,528 examined; 2,907 rejected (score ties or multi-turn), 3,336 for long prompts. Pairs whose chosen and rejected token windows become identical after truncation are also removed.

**Prompt format** (identical for SFT, DPO and all Qwen generations):
```
### Instruction:\n{instruction}\n\n### Response:\n{response}<|endoftext|>
```
A plain template was chosen because the base model has no chat tuning. The appended EOS (`<|endoftext|>`) is exactly the token the adapter contract appends. SmolLM2 uses its own ChatML chat template, applied by TRL and by `generate.py --prompt_format chat`.

## 2. Experimental configuration (§14.2)

[PASTE *auto* table "Experimental configuration" from `outputs/report_tables.md`]

**Generation settings** (all comparisons): greedy decoding (no temperature, no top-p), `max_new_tokens=512`, `repetition_penalty=1.1`, no n-gram constraint, fp32 weights, stop on EOS or on the string `### Instruction:`. Identical for every model within a comparison; per-file settings are in `outputs/generations/*.meta.json`. [PASTE *auto* generation-settings table]

**Implementation notes.**
- **SFT loss reduction:** the mean over all unmasked response tokens in the effective batch. The normaliser is the response-token count across all 16 accumulated micro-batches, so every token in an optimizer step has equal weight.
- **Precision:** fp16 autocast with fp32 master weights and a GradScaler. T4s do not support bf16.
- **DPO reference:** loaded in fp32 under the same autocast, so step-0 policy and reference log-probs are identical (loss = ln 2, reward accuracy = 0). Log-probs are summed over response tokens only, in fp32.
- **Exact reference checkpoint:** for both DPO runs, the policy init and the frozen reference are the SFT *best* checkpoint, step [FILL from `reference_checkpoint_state` in `outputs/dpo_*/summary.json`], validation loss [FILL].

**LoRA configuration justification (Task 5, required by §11).**
- **Rank r=16, α=32 (scale 2).** Large enough to express a consistent style shift across topics, while training only ~2% of parameters. [FILL: exact trainable count from `outputs/lora_persona/summary.json`]
- **Target modules: all attention and MLP projections.** Style lives in both *what* tokens are attended to and *how* they are transformed; attention-only LoRA tends to under-fit tone.
- **Learning rate 5e-5, cosine schedule, 10% warmup.** LoRA's B matrix starts at zero, so it needs a much larger LR than the 1e-6 used for full-parameter DPO.
- **Effective batch 8, 3 epochs (~75 steps).** With only 200 training pairs, one epoch is ~25 steps; three epochs give enough updates, while held-out reward accuracy is watched for overfitting.
- **Dropout 0.05.** Regularisation for the tiny dataset.
- **Reference model.** With `ref_model=None` and a PEFT config, TRL uses the base model with the adapter disabled.
- **Other.** β=0.1; fp16 AMP with fp32 weights.
- [FILL: anything you changed after looking at the curves/outputs, with the evidence]

## 3. Parameter choices and deviations (§14.3)

| Item | Guide value | Final value | Type | Reason / evidence |
|---|---|---|---|---|
| Reference precision | fp16 allowed | fp32 + fp16 autocast | numerical | exact policy/reference parity at step 0; GPU1 has ample memory [FILL: confirm step-0 val acc = 0.000] |
| DPO optimizer | RMSprop or AdamW | RMSprop | choice within guide | matches the DPO paper's setup; one state tensor per parameter (less memory than AdamW) |
| SFT data filter | not specified | complete examples ≤ 512 tokens | data | avoids training on truncated responses that never show an EOS [FILL: effect on stopping behaviour, e.g. % outputs hitting max tokens] |
| Weight decay scope | 0.1 | 0.1 on ≥2-D weights only | standard practice | biases and norm gains are not decayed |
| Decoding | greedy | greedy + repetition_penalty 1.1 | decoding | [FILL: sanity check showed loops without it / or remove if not needed] |
| transformers | ≥4.43,<5 | ≥4.56.2,<5 | resource / compatibility | required by TRL 0.29 for the LoRA task; same version used everywhere |
| [FILL any other change, e.g. batch size for OOM] | | | resource / performance | |

## 4. Training behaviour (§14.4)

![SFT loss](../outputs/plots/sft_loss.png)
![HH DPO](../outputs/plots/dpo_hh_training.png)
![UF DPO](../outputs/plots/dpo_uf_training.png)
![LoRA](../outputs/plots/lora_persona_training.png)

[PASTE *auto* table "Training outcomes"]

**Measured:**

*SFT:*
- Validation loss went from 1.3685 (base model, step 0) to 1.3335 at step 938, the best checkpoint and also the final one.
- It rose to about 1.39 over steps 100–200, at peak learning rate, before falling steadily.
- It was still decreasing at the end, so a second epoch would likely help.
- 0 skipped fp16 steps.

*HH-DPO (guide config):*
- Step 0: loss = ln 2 and margin 0.0000.
- Reward accuracy rose to about 0.60 by step 50, then plateaued at 0.60–0.62. The best was 0.616 at step 200, versus 0.615 at the final step (a difference of 1 pair in 1,000; SE ≈ ±1.5%).
- Validation loss fell throughout (0.660 → 0.639) and the margin grew (0.15 → 0.26).
- Both implicit rewards ended negative (r_w −0.31, r_l −0.58): likelihood displacement.

*UF-DPO (guide config):*
- Higher and cleaner signal: accuracy 0.686 at step 50 → 0.729 (best = final, step 469). Margin 0.48 → 0.76, loss 0.596 → 0.539.
- r_w stayed near zero (−0.16) while r_l fell to −0.93: DPO mostly pushed rejected responses down.

*LoRA-DPO:*
- Held-out accuracy reached 1.0 by step 10. Held-out loss fell 0.693 → 0.124, and margin rose to 2.15.
- Unlike HH, r_w *rose* (+1.09) while r_l fell (−1.06).
- Mean token entropy rose 2.25 → 2.53.

*Note on fp16:* 1–3 updates per DPO run were skipped by the GradScaler while it calibrated at the start (gradient norms of 40–70, versus about 5 in SFT).

**Interpretation:**
- [FILL/edit] After about 50 steps, DPO mainly increased confidence on pairs it already ranked correctly, rather than flipping new ones; accuracy plateaued while margins kept growing.
- UltraFeedback's GPT-4-scored pairs are more separable than HH's noisy human labels (0.73 vs 0.62).
- The persona pairs are separable almost immediately, so reward accuracy carries no information there after step 10; held-out loss is the useful signal.
- Do not infer response quality from these curves.

**Note on step-0 reward accuracy:** the logged step-0 accuracy is non-zero (0.09 for HH, 0.29 for UF) even though the loss is ln 2 and the margin prints as 0.0000. GPU floating-point summation isn't bit-deterministic, so the per-pair rewards differ by about 1e-7, and the strict `r_w > r_l` test turns that noise into "correct" pairs. Policy and reference are numerically identical at step 0.

## 5. DPO-versus-SFT results (§14.5)

Judge: [FILL: model actually used] via `kaggle_benchmarks`. Batches of 5 pairs, ≥5 s between attempts, up to 3 attempts per batch. Responses are labelled only "Response A/B", in a random order (seed 42). The HH rubric additionally rewards appropriate refusal, privacy protection, calibrated uncertainty and safe redirection. Unknown or failed judgments are excluded from win rates and reported separately.

[PASTE *auto* pairwise table. Keep UF-DPO (AlpacaEval) and HH-DPO (HH set) as separate rows. Do not combine them.]

**Uncertainty:** 95% CIs come from the Wilson interval (ties excluded) or a normal approximation (ties counted as ½). With n = 300, a CI is roughly ±5.5 points.

**Cross-set results (alignment tax) and the effect of SFT:** [PASTE the rows for HH-DPO on AlpacaEval, UF-DPO on the HH set, and SFT vs base]. Does safety-oriented DPO lose on general helpfulness? Does helpfulness DPO behave worse on the HH set?

**Length bias:** [PASTE the table from the notebook 06 length cell, or from `outputs/eval/length_bias.json`]. Report the mean words for candidate and baseline, how often the judge picked the longer answer, and the win rate on near-equal-length pairs, which is a crude length-controlled estimate (Dubois et al.).

**Position-bias check.** [FILL: consistency on 100 A/B-swapped pairs per track, and the "judge picked A" rate]

**LoRA persona (separate target).** Style rubric: LoRA vs base [FILL W/T/L]. Quality/safety rubric (style ignored): [FILL W/T/L]. [FILL: one sentence on whether style improved without losing usefulness or safety, citing the safety-sensitive prompts]

### Ablations (§14.3: performance-driven deviations; each changes one setting)

| Run | Change from guide | Val reward acc (best) | Val margin (final) | Judge vs SFT/base [95% CI] | Observations |
|---|---|---|---|---|---|
| HH-DPO (guide) | — | 0.616 | 0.264 | [FILL] | In 5 sanity samples, still complied with 2 of 3 clearly harmful requests |
| HH-DPO lr 5e-6 | lr 1e-6 → 5e-6 | [FILL] | [FILL] | [FILL] | [FILL: refusal rate by HH category; over-refusal on `benign_helpfulness`?] |
| LoRA (guide) | — | 1.00 | 2.15 | style [FILL] / quality [FILL] | Strong style; fixed sycophancy on "user is wrong" prompts; regressed on self-harm, knife and SQL prompts; one loop to max tokens |
| LoRA β 0.5 | β 0.1 → 0.5 | [FILL] | [FILL] | style [FILL] / quality [FILL] | [FILL] |
| LoRA 1 epoch | 3 → 1 epoch | [FILL] | [FILL] | style [FILL] / quality [FILL] | [FILL] |

Plots: `outputs/plots/dpo_ablation_comparison.png` and `outputs/plots/lora_ablation_comparison.png`.

## 6. Interpretation and error analysis (§14.6)

[FILL; separate *measured* from *interpretation*]
- **Which model won on each set, and by how much** (win rate, with and without ties).
- **Per-category breakdown** on the HH set ([PASTE per-category table]). Is the gain concentrated in refusal categories? Is there over-refusal on `benign_helpfulness` (alignment tax)? Use the `refusal_phrase_rate_by_category` values from `outputs/eval/generation_stats.json`.
- **Verbosity.** Mean words for SFT vs DPO (generation stats). Does the judge's preference track length?
- **Link to diagnostics.** Relate the results to reward accuracy, margin, β = 0.1, lr 1e-6 and about 469 steps. Small movement from the reference may explain small win-rate changes.
- **Observations already visible in the samples** (verify against the judge numbers):
  - **SFT taught format, not facts.** All 5 sanity answers ended with EOS, and the summary became concise (413 → 135 tokens). But "extinct bear species" got worse, listing the polar bear and a gorilla. DPO did not fix it either; UF-DPO added orangutans.
  - **UF-DPO lengthened answers** in the sanity samples (e.g. 241 → 309 and 365 → 512 tokens). This is the known verbosity effect of UltraFeedback-style preferences, and the reason for the length-bias check.
  - **The persona LoRA reduced sycophancy.** The base SmolLM2 agreed with bad practices: reusing a raw-chicken board, API keys in a public repo, bare `except: pass`, mixing bleach with ammonia, cooking chicken left out overnight. The persona model pushed back correctly on all of them, because blunt correction is exactly what the persona data rewards. At the same time it trivialised self-harm, gave knife-sharpening steps framed around harm, broke the SQL-injection answer, and looped once. The style and quality judges should be read together.
- **Examples** from `outputs/eval/examples_*.md`. Give 1-2 where DPO improves and 1-2 where it regresses or barely changes. Quote briefly and tie each to the aggregate numbers.

## 7. Limitations and reproducibility (§14.7)

**Limitations.**
- A single seed per run.
- Only 300 prompts per evaluation set (±[FILL] binomial standard error on the win rate).
- One LLM judge, with possible position, length and self-preference biases.
- One epoch per run, and only 24 persona prompts.
- SFT validation loss was still decreasing at the end; SFT was not retrained, because both DPO runs depend on that exact checkpoint.
- Training jobs are restartable at stage granularity (a finished stage is skipped on rerun), but a crashed run cannot resume mid-way: optimizer state is not checkpointed, to stay within Kaggle's 20 GB disk.
- Generation uses greedy decoding with repetition_penalty 1.1 for every model. The occasional degenerate loop was kept rather than hidden with n-gram blocking, which would also corrupt code answers.
- [FILL: any failed runs or Kaggle interruptions]

**Where everything lives.**
- Configs: `configs/*.yaml`. Actual values: `outputs/<run>/config_used.json`.
- Metrics: `outputs/<run>/metrics.jsonl` and `summary.json`.
- Generations: `outputs/generations/*.json` (+ `.meta.json`).
- Pairs and judgments: `outputs/eval/*_pairs.jsonl`, `*_judged.jsonl`, `*_summary.json`, `*_failures.jsonl`.
- Tables: `outputs/report_tables.md`.

**Reproduce.** Run notebooks `01` → `02` → `03a`/`03b`/`03c` → `04` → `05` → `06` in `notebooks/`. Offline tests: `PYTHONPATH=. python -m pytest -q tests/`.

## Works Cited

Bai, Yuntao, et al. "Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback." *arXiv*, 2022, arXiv:2204.05862.

Cui, Ganqu, et al. "UltraFeedback: Boosting Language Models with Scaled AI Feedback." *Proceedings of the 41st International Conference on Machine Learning*, 2024.

Ding, Ning, et al. "Enhancing Chat Language Models by Scaling High-Quality Instructional Conversations." *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing*, 2023.

Dubois, Yann, et al. "Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators." *arXiv*, 2024, arXiv:2404.04475.

Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *International Conference on Learning Representations*, 2022.

Ouyang, Long, et al. "Training Language Models to Follow Instructions with Human Feedback." *Advances in Neural Information Processing Systems*, vol. 35, 2022.

Qwen Team. "Qwen2.5 Technical Report." *arXiv*, 2024, arXiv:2412.15115.

Rafailov, Rafael, et al. "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model." *Advances in Neural Information Processing Systems*, vol. 36, 2023.

von Werra, Leandro, et al. *TRL: Transformer Reinforcement Learning*. GitHub, 2020, github.com/huggingface/trl.

Wolf, Thomas, et al. "Transformers: State-of-the-Art Natural Language Processing." *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*, 2020, pp. 38-45.

Zheng, Lianmin, et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." *Advances in Neural Information Processing Systems*, vol. 36, 2023.
