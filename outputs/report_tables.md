# Report tables (auto-generated)

## Experimental configuration (Section 14.2)

| Run | Params trained | Optimizer | Peak LR | Schedule / warmup | Weight decay | Clip | Per-device batch | Grad accum | Effective batch | Precision | Max len | Epochs / steps | Seed | β | Reference |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT | full | adamw (betas [0.9, 0.999]) | 2e-05 | cosine, 3% warmup | 0.1 | 1.0 | 2 | 16 | 32 | fp16 autocast, fp32 weights | 512 | 1 / 938 | 42 | – | – |
| dpo_hh | full | rmsprop | 1e-06 | constant, 0 warmup steps | 0.0 | 1.0 | 1 pair | 32 | 32 | fp16 autocast, fp32 policy; reference float32 | 512 (prompt ≤ 256) | 1 / 469 | 42 | 0.1 | SFT best, step 938 (frozen) |
| dpo_uf | full | rmsprop | 1e-06 | constant, 0 warmup steps | 0.0 | 1.0 | 1 pair | 32 | 32 | fp16 autocast, fp32 policy; reference float32 | 512 (prompt ≤ 256) | 1 / 469 | 42 | 0.1 | SFT best, step 938 (frozen) |
| dpo_hh_lr5e6 | full | rmsprop | 5e-06 | constant, 0 warmup steps | 0.0 | 1.0 | 1 pair | 32 | 32 | fp16 autocast, fp32 policy; reference float32 | 512 (prompt ≤ 256) | 1 / 469 | 42 | 0.1 | SFT best, step 938 (frozen) |
| lora_persona | LoRA r=16, α=32, dropout=0.05, q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj (8,683,520 params) | adamw_torch | 5e-05 | cosine, 10% warmup | 0.0 | 1.0 | 4 | 2 | 8 | fp16 AMP, fp32 weights | 512 | 3 / 75 | 42 | 0.1 | base model with adapter disabled |
| lora_persona_beta05 | LoRA r=16, α=32, dropout=0.05, q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj (8,683,520 params) | adamw_torch | 5e-05 | cosine, 10% warmup | 0.0 | 1.0 | 4 | 2 | 8 | fp16 AMP, fp32 weights | 512 | 3 / 75 | 42 | 0.5 | base model with adapter disabled |
| lora_persona_1epoch | LoRA r=16, α=32, dropout=0.05, q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj (8,683,520 params) | adamw_torch | 5e-05 | cosine, 10% warmup | 0.0 | 1.0 | 4 | 2 | 8 | fp16 AMP, fp32 weights | 512 | 1 / 25 | 42 | 0.1 | base model with adapter disabled |

## Training outcomes (Section 14.4)

| Run | Best step | Selection metric (best) | Final validation | Train time | Skipped fp16 steps |
|---|---|---|---|---|---|
| SFT | 938 | val loss 1.3335 | val loss 1.3335 | 1h23m26s | 0 |
| dpo_hh | 200 | reward acc 0.616, margin 0.2146 | loss 0.6393, acc 0.615, margin 0.2636 | 1h11m07s | 3 |
| dpo_uf | 469 | reward acc 0.729, margin 0.7621 | loss 0.5390, acc 0.729, margin 0.7621 | 1h39m32s | 3 |
| dpo_hh_lr5e6 | 350 | reward acc 0.620, margin 0.3841 | loss 0.7040, acc 0.615, margin 0.4044 | 1h13m00s | 3 |
| lora_persona | 75 (final) | – | loss 0.1244, acc 1.0, margin 2.14731502532959 | – | – |
| lora_persona_beta05 | 75 (final) | – | loss 0.0055, acc 1.0, margin 6.167818069458008 | – | – |
| lora_persona_1epoch | 25 (final) | – | loss 0.4845, acc 1.0, margin 0.47775182127952576 | – | – |

## Pairwise judgments (Section 14.5)

| Comparison | n | Wins | Ties | Losses | Win rate excl. ties [95% CI] | Win rate ties = ½ [95% CI] | Unknown | Failed | Pairs retried | Judge |
|---|---|---|---|---|---|---|---|---|---|---|
| dpo_hh_lr5e6_vs_sft_alpaca [general] | 300 | 102 | 20 | 178 | 36.4% [31.0, 42.2] | 37.3% [32.0, 42.6] | 0 | 0 | 0 | google/gemini-3.6-flash |
| dpo_hh_lr5e6_vs_sft [safety] | 300 | 135 | 13 | 152 | 47.0% [41.3, 52.8] | 47.2% [41.6, 52.7] | 0 | 0 | 0 | google/gemini-3.6-flash |
| dpo_hh_vs_sft_alpaca [general] | 300 | 121 | 58 | 121 | 50.0% [43.7, 56.3] | 50.0% [44.9, 55.1] | 0 | 0 | 0 | google/gemini-3.6-flash |
| dpo_hh_vs_sft [safety] | 300 | 163 | 28 | 109 | 59.9% [54.0, 65.6] | 59.0% [53.7, 64.3] | 0 | 0 | 0 | google/gemini-3.6-flash |
| dpo_hh_vs_sft_mirrored [safety] | 300 | 148 | 27 | 125 | 54.2% [48.3, 60.0] | 53.8% [48.4, 59.2] | 0 | 0 | 0 | google/gemini-3.6-flash |
| dpo_uf_vs_sft_hh [safety] | 300 | 126 | 21 | 153 | 45.2% [39.4, 51.0] | 45.5% [40.1, 50.9] | 0 | 0 | 0 | google/gemini-3.6-flash |
| dpo_uf_vs_sft [general] | 300 | 125 | 29 | 146 | 46.1% [40.3, 52.1] | 46.5% [41.1, 51.9] | 0 | 0 | 5 | google/gemini-3.6-flash |
| dpo_uf_vs_sft_mirrored [general] | 300 | 126 | 28 | 146 | 46.3% [40.5, 52.3] | 46.7% [41.3, 52.0] | 0 | 0 | 0 | google/gemini-3.6-flash |
| lora_1epoch_vs_base_persona_persona_quality [persona_quality] | 24 | 9 | 4 | 11 | 45.0% [25.8, 65.8] | 45.8% [27.3, 64.4] | 0 | 0 | 0 | google/gemini-3.6-flash |
| lora_1epoch_vs_base_persona_persona_style [persona_style] | 24 | 2 | 18 | 4 | 33.3% [9.7, 70.0] | 45.8% [35.8, 55.9] | 0 | 0 | 0 | google/gemini-3.6-flash |
| lora_beta05_vs_base_persona_persona_quality [persona_quality] | 24 | 11 | 1 | 12 | 47.8% [29.2, 67.0] | 47.9% [27.9, 67.9] | 0 | 0 | 0 | google/gemini-3.6-flash |
| lora_beta05_vs_base_persona_persona_style [persona_style] | 24 | 5 | 15 | 4 | 55.6% [26.7, 81.1] | 52.1% [39.6, 64.6] | 0 | 0 | 0 | google/gemini-3.6-flash |
| lora_vs_base_persona_persona_quality [persona_quality] | 24 | 11 | 1 | 12 | 47.8% [29.2, 67.0] | 47.9% [27.9, 67.9] | 0 | 0 | 0 | google/gemini-3.6-flash |
| lora_vs_base_persona_persona_style [persona_style] | 24 | 13 | 9 | 2 | 86.7% [62.1, 96.3] | 72.9% [59.8, 86.1] | 0 | 0 | 0 | google/gemini-3.6-flash |
| sft_vs_base_alpaca [general] | 300 | 150 | 22 | 128 | 54.0% [48.1, 59.7] | 53.7% [48.2, 59.1] | 0 | 0 | 0 | google/gemini-3.6-flash |

### Per category: dpo_hh_lr5e6_vs_sft_alpaca_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| helpful_base | 15 | 4 | 32 | 31.9% |
| koala | 17 | 5 | 38 | 30.9% |
| oasst | 22 | 5 | 27 | 44.9% |
| selfinstruct | 36 | 2 | 57 | 38.7% |
| vicuna | 12 | 4 | 24 | 33.3% |

### Per category: dpo_hh_lr5e6_vs_sft_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| benign_helpfulness | 8 | 1 | 21 | 27.6% |
| calibration_and_uncertainty | 15 | 0 | 15 | 50.0% |
| cyber_abuse | 12 | 4 | 14 | 46.2% |
| harmful_instructions | 19 | 2 | 9 | 67.9% |
| hate_or_harassment | 21 | 1 | 8 | 72.4% |
| legal_financial_safety | 10 | 0 | 20 | 33.3% |
| medical_safety | 12 | 3 | 15 | 44.4% |
| privacy_and_doxxing | 18 | 2 | 10 | 64.3% |
| safe_transformation | 12 | 0 | 18 | 40.0% |
| self_harm_support | 8 | 0 | 22 | 26.7% |

### Per category: dpo_hh_vs_sft_alpaca_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| helpful_base | 20 | 8 | 23 | 46.5% |
| koala | 29 | 9 | 22 | 56.9% |
| oasst | 18 | 12 | 24 | 42.9% |
| selfinstruct | 38 | 26 | 31 | 55.1% |
| vicuna | 16 | 3 | 21 | 43.2% |

### Per category: dpo_hh_vs_sft_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| benign_helpfulness | 16 | 2 | 12 | 57.1% |
| calibration_and_uncertainty | 12 | 3 | 15 | 44.4% |
| cyber_abuse | 20 | 6 | 4 | 83.3% |
| harmful_instructions | 17 | 1 | 12 | 58.6% |
| hate_or_harassment | 18 | 3 | 9 | 66.7% |
| legal_financial_safety | 18 | 2 | 10 | 64.3% |
| medical_safety | 21 | 2 | 7 | 75.0% |
| privacy_and_doxxing | 10 | 9 | 11 | 47.6% |
| safe_transformation | 17 | 0 | 13 | 56.7% |
| self_harm_support | 14 | 0 | 16 | 46.7% |

### Per category: dpo_hh_vs_sft_mirrored_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| benign_helpfulness | 12 | 2 | 16 | 42.9% |
| calibration_and_uncertainty | 10 | 2 | 18 | 35.7% |
| cyber_abuse | 17 | 7 | 6 | 73.9% |
| harmful_instructions | 14 | 4 | 12 | 53.8% |
| hate_or_harassment | 14 | 5 | 11 | 56.0% |
| legal_financial_safety | 15 | 2 | 13 | 53.6% |
| medical_safety | 23 | 2 | 5 | 82.1% |
| privacy_and_doxxing | 11 | 3 | 16 | 40.7% |
| safe_transformation | 16 | 0 | 14 | 53.3% |
| self_harm_support | 16 | 0 | 14 | 53.3% |

### Per category: dpo_uf_vs_sft_hh_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| benign_helpfulness | 19 | 1 | 10 | 65.5% |
| calibration_and_uncertainty | 12 | 2 | 16 | 42.9% |
| cyber_abuse | 12 | 7 | 11 | 52.2% |
| harmful_instructions | 10 | 2 | 18 | 35.7% |
| hate_or_harassment | 12 | 1 | 17 | 41.4% |
| legal_financial_safety | 8 | 1 | 21 | 27.6% |
| medical_safety | 11 | 3 | 16 | 40.7% |
| privacy_and_doxxing | 10 | 4 | 16 | 38.5% |
| safe_transformation | 14 | 0 | 16 | 46.7% |
| self_harm_support | 18 | 0 | 12 | 60.0% |

### Per category: dpo_uf_vs_sft_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| helpful_base | 18 | 9 | 24 | 42.9% |
| koala | 32 | 7 | 21 | 60.4% |
| oasst | 16 | 5 | 33 | 32.7% |
| selfinstruct | 43 | 5 | 47 | 47.8% |
| vicuna | 16 | 3 | 21 | 43.2% |

### Per category: dpo_uf_vs_sft_mirrored_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| helpful_base | 17 | 6 | 28 | 37.8% |
| koala | 25 | 5 | 30 | 45.5% |
| oasst | 24 | 5 | 25 | 49.0% |
| selfinstruct | 45 | 10 | 40 | 52.9% |
| vicuna | 15 | 2 | 23 | 39.5% |

### Per category: lora_1epoch_vs_base_persona_persona_quality_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| coding | 2 | 1 | 3 | 40.0% |
| cooking | 5 | 0 | 1 | 83.3% |
| general_help | 0 | 2 | 4 | 0.0% |
| safety_sensitive | 2 | 1 | 3 | 40.0% |

### Per category: lora_1epoch_vs_base_persona_persona_style_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| coding | 0 | 4 | 2 | 0.0% |
| cooking | 2 | 2 | 2 | 50.0% |
| general_help | 0 | 6 | 0 | n/a |
| safety_sensitive | 0 | 6 | 0 | n/a |

### Per category: lora_beta05_vs_base_persona_persona_quality_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| coding | 3 | 0 | 3 | 50.0% |
| cooking | 3 | 0 | 3 | 50.0% |
| general_help | 2 | 1 | 3 | 40.0% |
| safety_sensitive | 3 | 0 | 3 | 50.0% |

### Per category: lora_beta05_vs_base_persona_persona_style_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| coding | 0 | 6 | 0 | n/a |
| cooking | 3 | 1 | 2 | 60.0% |
| general_help | 0 | 6 | 0 | n/a |
| safety_sensitive | 2 | 2 | 2 | 50.0% |

### Per category: lora_vs_base_persona_persona_quality_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| coding | 4 | 0 | 2 | 66.7% |
| cooking | 3 | 0 | 3 | 50.0% |
| general_help | 1 | 1 | 4 | 20.0% |
| safety_sensitive | 3 | 0 | 3 | 50.0% |

### Per category: lora_vs_base_persona_persona_style_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| coding | 3 | 2 | 1 | 75.0% |
| cooking | 4 | 1 | 1 | 80.0% |
| general_help | 2 | 4 | 0 | 100.0% |
| safety_sensitive | 4 | 2 | 0 | 100.0% |

### Per category: sft_vs_base_alpaca_judged

| Category | Wins | Ties | Losses | Win rate (excl. ties) |
|---|---|---|---|---|
| helpful_base | 25 | 7 | 19 | 56.8% |
| koala | 32 | 5 | 23 | 58.2% |
| oasst | 26 | 1 | 27 | 49.1% |
| selfinstruct | 44 | 8 | 43 | 50.6% |
| vicuna | 23 | 1 | 16 | 59.0% |

## Generation settings

| File | Model | Prompt format | Decoding | Temperature / top-p | max_new_tokens | repetition_penalty | length_penalty | no_repeat_ngram | Stop |
|---|---|---|---|---|---|---|---|---|---|
| base_alpaca | Qwen/Qwen2.5-0.5B | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| base_hh | Qwen/Qwen2.5-0.5B | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| dpo_hh_alpaca | /kaggle/input/notebooks/sarmadnawazmalik/03a-train-dpo-hh/small-model-alignment/outputs/dpo_hh/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| dpo_hh_hh | /kaggle/input/notebooks/sarmadnawazmalik/03a-train-dpo-hh/small-model-alignment/outputs/dpo_hh/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| dpo_hh_lr5e6_alpaca | outputs/dpo_hh_lr5e6/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| dpo_hh_lr5e6_hh | outputs/dpo_hh_lr5e6/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| dpo_uf_alpaca | /kaggle/input/notebooks/sarmadnawazmalik/03b-train-dpo-uf/small-model-alignment/outputs/dpo_uf/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| dpo_uf_hh | /kaggle/input/notebooks/sarmadnawazmalik/03b-train-dpo-uf/small-model-alignment/outputs/dpo_uf/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| lora_persona | HuggingFaceTB/SmolLM2-360M-Instruct + outputs/lora_persona/adapter | chat | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS / <\|im_end\|> (chat template) |
| lora_persona_1epoch | HuggingFaceTB/SmolLM2-360M-Instruct + outputs/lora_persona_1epoch/adapter | chat | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS / <\|im_end\|> (chat template) |
| lora_persona_beta05 | HuggingFaceTB/SmolLM2-360M-Instruct + outputs/lora_persona_beta05/adapter | chat | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS / <\|im_end\|> (chat template) |
| sanity_dpo_hh | /kaggle/input/notebooks/sarmadnawazmalik/03a-train-dpo-hh/small-model-alignment/outputs/dpo_hh/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| sanity_dpo_uf | /kaggle/input/notebooks/sarmadnawazmalik/03b-train-dpo-uf/small-model-alignment/outputs/dpo_uf/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| sanity_sft | /kaggle/input/notebooks/sarmadnawazmalik/02-train-sft/small-model-alignment/outputs/sft/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| sft_alpaca | /kaggle/input/notebooks/sarmadnawazmalik/02-train-sft/small-model-alignment/outputs/sft/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| sft_hh | /kaggle/input/notebooks/sarmadnawazmalik/02-train-sft/small-model-alignment/outputs/sft/best | plain | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS + '### Instruction:', '\n### Response:' |
| smollm_base_persona | HuggingFaceTB/SmolLM2-360M-Instruct | chat | greedy (float32) | n/a (greedy) | 512 | 1.1 | none | 0 | EOS / <\|im_end\|> (chat template) |
