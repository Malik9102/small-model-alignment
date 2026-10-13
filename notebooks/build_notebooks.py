"""Generate the Kaggle notebooks in this folder (run: python notebooks/build_notebooks.py).

Edit REPO_URL in each generated notebook (first code cell) to point at YOUR fork.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

SETUP = r'''# ==== EDIT THIS: your GitHub fork of the repository ====
REPO_URL = "https://github.com/Malik9102/small-model-alignment.git"
REPO_DIR = "/kaggle/working/small-model-alignment"

import glob, os, shutil, subprocess, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)
print("cwd:", os.getcwd())
subprocess.run(["git", "log", "-1", "--oneline"], check=False)

def sh(command):
    """Run a shell command, stream its output, and STOP the notebook if it fails.
    (A plain `!command` keeps going after an error, so a crashed training run would go unnoticed.)"""
    print(f"$ {command}", flush=True)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {command}")

def find_input(pattern):
    """Locate a file/dir inside attached Kaggle inputs (any dataset/notebook slug)."""
    hits = sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(f"Nothing matching {pattern!r} under /kaggle/input - did you attach the right notebook output?")
    if len(hits) > 1:
        print(f"[find_input] {len(hits)} matches for {pattern!r}, using {hits[0]}")
    return hits[0]'''

INSTALL = r'''# Keep Kaggle's CUDA-matched PyTorch. Do NOT reinstall torch.
!python -m pip uninstall -y -q torchao || true
!python -m pip install -q "transformers>=4.56.2,<5" "datasets>=2.18,<5" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors matplotlib pytest
!python -m pip install -q -e . --no-deps'''

INSTALL_TRL = r'''# Keep Kaggle's CUDA-matched PyTorch. TRL 0.29 needs transformers>=4.56.2; pin <5 in the SAME command.
!python -m pip uninstall -y -q torchao || true
!python -m pip install -q "transformers>=4.56.2,<5" "datasets>=2.18,<5" "trl==0.29.0" "peft>=0.13,<0.19" huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors matplotlib pytest
!python -m pip install -q -e . --no-deps'''

ENV = "!python scripts/env_check.py"

TESTS = "!PYTHONPATH=. python -m pytest -q tests/"


def copy_data_cell():
    return r'''# Bring the fixed data files from notebook 01's output into the repo
os.makedirs("data/processed", exist_ok=True)
src = os.path.dirname(find_input("data/processed/manifest.json"))
for name in os.listdir(src):
    shutil.copy(os.path.join(src, name), "data/processed/")
shutil.copy(find_input("eval_sets/alpaca_eval_300_seed42.json"), "eval_sets/")
print(sorted(os.listdir("data/processed")))'''


def zip_cell(name, paths):
    joined = " ".join(paths)
    return f'''# Lightweight archive (metrics/logs/plots/generations - not the 2 GB checkpoints)
!cd {{REPO_DIR}} && zip -qr /kaggle/working/{name}.zip {joined} -x "*/best/*" "*/final/*" "*/trainer/*" && ls -lh /kaggle/working/{name}.zip
!df -h /kaggle/working | tail -1'''


def fail_fast(source: str) -> str:
    """Turn every `!command` line into `sh(f\"\"\"command\"\"\")` (IPython {var} syntax == f-string syntax)."""

    lines = []
    for line in source.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("!"):
            indent = line[: len(line) - len(stripped)]
            command = stripped[1:].replace("\\", "\\\\")
            lines.append(f'{indent}sh(f"""{command}""")')
        else:
            lines.append(line)
    return "\n".join(lines)


def notebook(title, cells):
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(title))
    for index, (kind, source) in enumerate(cells):
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            # the very first code cell defines sh(), so it cannot use it itself
            nb.cells.append(nbf.v4.new_code_cell(source if "def sh(" in source else fail_fast(source)))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


NOTEBOOKS = {
    "01_setup_data_baseline.ipynb": notebook(
        "# 01 - Setup (Task 1), data preparation, baseline generation (Task 2)\n"
        "**Accelerator:** GPU T4 x2 · **Internet:** ON · **Inputs:** none · **Runtime:** ~30-45 min\n\n"
        "When it finishes: **Save Version → Save & Run All (Commit)**. Later notebooks attach this notebook's output.",
        [
            ("code", SETUP),
            ("code", INSTALL),
            ("md", "## Task 1 - environment check (two T4s + package versions)"),
            ("code", ENV),
            ("md", "## Offline correctness tests (public smoke tests + extended edge cases + tiny end-to-end pipeline)"),
            ("code", TESTS),
            ("md", "## Fixed data files\nSFT 30k/1k (UltraChat), HH-RLHF 15k/1k, UltraFeedback 15k/1k, AlpacaEval 300 prompts (seed 42)."),
            ("code", "!python scripts/prepare_data.py --output_dir data/processed\n!cat data/processed/manifest.json"),
            ("md", "## Task 2 - baseline generations (Qwen/Qwen2.5-0.5B base)\nSanity check on 5 prompts first, then the full sets."),
            (
                "code",
                "!python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --eval_set eval_sets/alpaca_eval_300_seed42.json "
                "--generator qwen2.5-0.5b-base --output outputs/generations/sanity_base.json --limit 5 --print_samples --overwrite",
            ),
            (
                "code",
                "!python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --eval_set eval_sets/alpaca_eval_300_seed42.json "
                "--generator qwen2.5-0.5b-base --output outputs/generations/base_alpaca.json\n"
                "!python scripts/generate.py --model_path Qwen/Qwen2.5-0.5B --eval_set eval_sets/hh_rlhf_eval_300_seed42.json "
                "--generator qwen2.5-0.5b-base --output outputs/generations/base_hh.json",
            ),
            ("code", "!python scripts/analyze_generations.py outputs/generations/base_alpaca.json outputs/generations/base_hh.json"),
            ("code", zip_cell("01_data_and_baseline", ["data/processed", "eval_sets", "outputs/generations"])),
        ],
    ),
    "02_train_sft.ipynb": notebook(
        "# 02 - Supervised fine-tuning (Task 3)\n"
        "**Accelerator:** GPU T4 x2 (SFT uses cuda:0) · **Internet:** ON · **Inputs:** output of notebook 01 · **Runtime:** ~1.5-2 h\n\n"
        "Run with **Save Version → Save & Run All (Commit)** so it keeps running if your browser disconnects.",
        [
            ("code", SETUP),
            ("code", INSTALL),
            ("code", ENV),
            ("code", copy_data_cell()),
            ("md", "## Pilot run (5 optimizer steps) - catches OOM / path errors in ~2 minutes"),
            (
                "code",
                "!python scripts/train_sft.py --config configs/sft.yaml --set max_steps=5 --set limit_val=64 --set eval_every=5 "
                "--set log_every=1 --set save_final=false --set output_dir=/kaggle/working/pilot_sft\n"
                "shutil.rmtree('/kaggle/working/pilot_sft', ignore_errors=True)",
            ),
            ("md", "## Full SFT run: 30,000 examples, effective batch 32, lr 2e-5, cosine, 1 epoch"),
            ("code", "!python scripts/train_sft.py --config configs/sft.yaml --set output_dir=outputs/sft"),
            ("code", "!cat outputs/sft/summary.json\n!python scripts/plot_metrics.py --sft outputs/sft --output_dir outputs/plots"),
            ("code", "from IPython.display import Image\nImage('outputs/plots/sft_loss.png')"),
            ("md", "## Sanity generations from the SFT best checkpoint"),
            (
                "code",
                "!python scripts/generate.py --model_path outputs/sft/best --eval_set eval_sets/alpaca_eval_300_seed42.json "
                "--generator qwen2.5-0.5b-sft --output outputs/generations/sanity_sft.json --limit 5 --print_samples --overwrite",
            ),
            ("code", zip_cell("02_sft_run", ["outputs/sft", "outputs/plots"])),
        ],
    ),
}


def dpo_notebook(run: str) -> nbf.NotebookNode:
    label = {"hh": "HH-RLHF (helpful/harmless)", "uf": "UltraFeedback (general helpfulness)"}[run]
    eval_set = {"hh": "eval_sets/hh_rlhf_eval_300_seed42.json", "uf": "eval_sets/alpaca_eval_300_seed42.json"}[run]
    return notebook(
        f"# 03{'a' if run == 'hh' else 'b'} - From-scratch DPO on {label} (Task 4)\n"
        "**Accelerator:** GPU T4 x2 (policy on cuda:0, frozen reference on cuda:1) · **Internet:** ON · "
        "**Inputs:** outputs of notebooks 01 and 02 · **Runtime:** ~2-2.5 h\n\n"
        "Run with **Save Version → Save & Run All (Commit)**.",
        [
            ("code", SETUP),
            ("code", INSTALL),
            ("code", ENV),
            ("code", copy_data_cell()),
            (
                "code",
                "SFT_BEST = os.path.dirname(find_input('sft/best/model.safetensors'))\n"
                "print('SFT best checkpoint (policy init + frozen reference):', SFT_BEST)\n"
                "!cat {SFT_BEST}/training_state.json | head -c 300",
            ),
            ("md", "## Pilot (3 optimizer steps). Step-0 validation must show loss 0.6931 and reward accuracy 0.000."),
            (
                "code",
                f"!python scripts/train_dpo.py --config configs/dpo_{run}.yaml --set sft_checkpoint={{SFT_BEST}} --set max_steps=3 "
                "--set limit_val=32 --set eval_every=3 --set log_every=1 --set save_final=false --set output_dir=/kaggle/working/pilot_dpo\n"
                "shutil.rmtree('/kaggle/working/pilot_dpo', ignore_errors=True)",
            ),
            ("md", "## Full run: 15,000 pairs, β=0.1, lr 1e-6 constant, effective batch 32, best checkpoint by validation reward accuracy"),
            (
                "code",
                f"!python scripts/train_dpo.py --config configs/dpo_{run}.yaml --set sft_checkpoint={{SFT_BEST}} --set output_dir=outputs/dpo_{run}",
            ),
            ("code", f"!cat outputs/dpo_{run}/summary.json\n!python scripts/plot_metrics.py --dpo outputs/dpo_{run} --output_dir outputs/plots"),
            ("code", f"from IPython.display import Image\nImage('outputs/plots/dpo_{run}_training.png')"),
            (
                "code",
                f"!python scripts/generate.py --model_path outputs/dpo_{run}/best --eval_set {eval_set} "
                f"--generator qwen2.5-0.5b-dpo-{run} --output outputs/generations/sanity_dpo_{run}.json --limit 5 --print_samples --overwrite",
            ),
            ("code", zip_cell(f"03_dpo_{run}_run", [f"outputs/dpo_{run}", "outputs/plots"])),
        ],
    )


NOTEBOOKS["03a_train_dpo_hh.ipynb"] = dpo_notebook("hh")
NOTEBOOKS["03b_train_dpo_uf.ipynb"] = dpo_notebook("uf")

NOTEBOOKS["04_train_lora_persona.ipynb"] = notebook(
    "# 04 - Idiot Sandwich: TRL DPOTrainer + LoRA persona tuning (Task 5)\n"
    "**Accelerator:** GPU T4 (x1 or x2; training is pinned to one GPU) · **Internet:** ON · **Inputs:** none · **Runtime:** ~15-25 min",
    [
        ("code", SETUP),
        ("code", INSTALL_TRL),
        ("code", ENV),
        ("md", "## Train (r=16, α=32, all attention + MLP projections, lr 5e-5, β=0.1, 3 epochs on 200 pairs)"),
        ("code", "!python scripts/train_lora_dpo.py --config configs/lora_persona.yaml --set output_dir=outputs/lora_persona"),
        ("code", "!cat outputs/lora_persona/summary.json\n!python scripts/plot_metrics.py --lora outputs/lora_persona --output_dir outputs/plots"),
        ("code", "from IPython.display import Image\nImage('outputs/plots/lora_persona_training.png')"),
        ("md", "## Sanity check (3 prompts each): adapter loads, chat template applied, responses end cleanly"),
        (
            "code",
            "!python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path outputs/lora_persona/adapter "
            "--prompt_format chat --eval_set eval_sets/persona_style_eval.json --generator sanity-lora "
            "--output outputs/generations/sanity_lora.json --limit 3 --print_samples --overwrite",
        ),
        ("md", "## Generate on the fixed persona style set (cooking / coding / general help / safety-sensitive): base vs LoRA"),
        (
            "code",
            "!python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --prompt_format chat "
            "--eval_set eval_sets/persona_style_eval.json --generator smollm2-360m-instruct "
            "--output outputs/generations/smollm_base_persona.json --print_samples\n"
            "!python scripts/generate.py --model_path HuggingFaceTB/SmolLM2-360M-Instruct --adapter_path outputs/lora_persona/adapter "
            "--prompt_format chat --eval_set eval_sets/persona_style_eval.json --generator smollm2-360m-lora-dpo "
            "--output outputs/generations/lora_persona.json --print_samples",
        ),
        (
            "code",
            "!python scripts/analyze_generations.py outputs/generations/smollm_base_persona.json outputs/generations/lora_persona.json",
        ),
        ("md", "## Blinded pairwise file (LoRA = candidate, base SmolLM2 = baseline), written BEFORE judging (manual 5.5 / 13.1)"),
        (
            "code",
            "!python scripts/build_pairs.py --candidate outputs/generations/lora_persona.json --baseline outputs/generations/smollm_base_persona.json "
            "--output outputs/eval/lora_vs_base_persona_pairs.jsonl",
        ),
        ("code", zip_cell("04_lora_persona", ["outputs/lora_persona", "outputs/generations", "outputs/eval", "outputs/plots"]).replace(
            '-x "*/best/*" "*/final/*" "*/trainer/*"', '-x "*/trainer/*"')),
    ],
)

NOTEBOOKS["05_generate_eval_outputs.ipynb"] = notebook(
    "# 05 - Evaluation generations + pairwise files (Task 6)\n"
    "**Accelerator:** GPU T4 x2 · **Internet:** ON · **Inputs:** outputs of notebooks 01, 02, 03a, 03b · **Runtime:** ~30-60 min\n\n"
    "Greedy decoding, identical settings for every model in a comparison (see `*.meta.json`).",
    [
        ("code", SETUP),
        ("code", INSTALL),
        ("code", copy_data_cell()),
        (
            "code",
            "CKPT = {\n"
            "    'sft': os.path.dirname(find_input('sft/best/model.safetensors')),\n"
            "    'dpo_hh': os.path.dirname(find_input('dpo_hh/best/model.safetensors')),\n"
            "    'dpo_uf': os.path.dirname(find_input('dpo_uf/best/model.safetensors')),\n"
            "}\nprint(CKPT)\n"
            "os.makedirs('outputs/generations', exist_ok=True)\n"
            "for name in ('base_alpaca.json', 'base_alpaca.meta.json', 'base_hh.json', 'base_hh.meta.json'):\n"
            "    try: shutil.copy(find_input(f'outputs/generations/{name}'), 'outputs/generations/')\n"
            "    except FileNotFoundError as e: print(e)",
        ),
        ("md", "## Sanity check (2-5 prompts per checkpoint): path, tokenizer, prompt format, EOS behaviour, length"),
        (
            "code",
            "for key, path in CKPT.items():\n"
            "    !python scripts/generate.py --model_path {path} --eval_set eval_sets/alpaca_eval_300_seed42.json --generator sanity-{key} "
            "--output outputs/generations/sanity_{key}.json --limit 3 --print_samples --overwrite",
        ),
        ("md", "## Full generations\nSFT is generated on both sets (it is the baseline for both comparisons). Each DPO model on its own eval set; the cross-set generations are optional extras for error analysis."),
        (
            "code",
            "JOBS = [\n"
            "    ('sft',    'alpaca', 'qwen2.5-0.5b-sft'),\n"
            "    ('sft',    'hh',     'qwen2.5-0.5b-sft'),\n"
            "    ('dpo_uf', 'alpaca', 'qwen2.5-0.5b-dpo-uf'),\n"
            "    ('dpo_hh', 'hh',     'qwen2.5-0.5b-dpo-hh'),\n"
            "    # optional cross-set analysis (alignment tax / over-refusal):\n"
            "    ('dpo_hh', 'alpaca', 'qwen2.5-0.5b-dpo-hh'),\n"
            "    ('dpo_uf', 'hh',     'qwen2.5-0.5b-dpo-uf'),\n"
            "]\n"
            "EVAL = {'alpaca': 'eval_sets/alpaca_eval_300_seed42.json', 'hh': 'eval_sets/hh_rlhf_eval_300_seed42.json'}\n"
            "for key, eval_name, generator in JOBS:\n"
            "    out = f'outputs/generations/{key}_{eval_name}.json'\n"
            "    !python scripts/generate.py --model_path {CKPT[key]} --eval_set {EVAL[eval_name]} --generator {generator} --output {out}",
        ),
        ("md", "## Blinded pairwise files (DPO = candidate, SFT = baseline) + an A/B-swapped subset for position-bias checks"),
        (
            "code",
            "!python scripts/build_pairs.py --candidate outputs/generations/dpo_uf_alpaca.json --baseline outputs/generations/sft_alpaca.json --output outputs/eval/dpo_uf_vs_sft_pairs.jsonl\n"
            "!python scripts/build_pairs.py --candidate outputs/generations/dpo_hh_hh.json --baseline outputs/generations/sft_hh.json --output outputs/eval/dpo_hh_vs_sft_pairs.jsonl\n"
            "!python scripts/build_pairs.py --mirror_of outputs/eval/dpo_uf_vs_sft_pairs.jsonl --limit 100 --output outputs/eval/dpo_uf_vs_sft_mirrored_pairs.jsonl\n"
            "!python scripts/build_pairs.py --mirror_of outputs/eval/dpo_hh_vs_sft_pairs.jsonl --limit 100 --output outputs/eval/dpo_hh_vs_sft_mirrored_pairs.jsonl",
        ),
        ("code", "!python scripts/analyze_generations.py outputs/generations/*.json --output outputs/eval/generation_stats.json"),
        ("code", zip_cell("05_generations", ["outputs/generations", "outputs/eval", "eval_sets"])),
    ],
)

NOTEBOOKS["06_judge_and_report_assets.ipynb"] = notebook(
    "# 06 - Pairwise judging with Kaggle models (Task 7) + report assets (Task 8)\n"
    "**Create this as a Kaggle *Benchmark* notebook** (kaggle.com/benchmarks → create a task) so `kaggle_benchmarks` and the "
    "model proxy are available. **Accelerator:** none needed · **Inputs:** outputs of notebooks 02, 03a, 03b, 04, 05 · **Runtime:** ~30-60 min\n\n"
    "This notebook only *judges saved pairwise files*; it never regenerates model outputs.\n\n"
    "Judging is resumable: if the session dies, rerun - completed pairs are reused. Save a version when done.",
    [
        ("code", SETUP),
        ("code", "!python -m pip install -q pyyaml matplotlib"),
        (
            "code",
            "import kaggle_benchmarks as kbench\n"
            "print('models visible to kaggle_benchmarks:', sorted(kbench.llms))\n"
            "JUDGE = 'google/gemini-3.6-flash'   # if not listed above, pick a listed model and record it in the report",
        ),
        (
            "code",
            "# Gather everything from the attached inputs (metrics only - no checkpoints needed here)\n"
            "for sub in ('generations', 'eval'):\n"
            "    src = os.path.dirname(find_input(f'outputs/{sub}/' + ('sft_alpaca.json' if sub == 'generations' else 'dpo_uf_vs_sft_pairs.jsonl')))\n"
            "    shutil.copytree(src, f'outputs/{sub}', dirs_exist_ok=True)\n"
            "persona_src = os.path.dirname(find_input('outputs/generations/lora_persona.json'))\n"
            "for name in ('lora_persona.json', 'lora_persona.meta.json', 'smollm_base_persona.json', 'smollm_base_persona.meta.json'):\n"
            "    shutil.copy(os.path.join(persona_src, name), 'outputs/generations/')\n"
            "shutil.copy(find_input('outputs/eval/lora_vs_base_persona_pairs.jsonl'), 'outputs/eval/')  # built in notebook 04\n"
            "for run in ('sft', 'dpo_hh', 'dpo_uf', 'lora_persona'):\n"
            "    src = os.path.dirname(find_input(f'outputs/{run}/summary.json'))\n"
            "    os.makedirs(f'outputs/{run}', exist_ok=True)\n"
            "    for name in ('summary.json', 'config_used.json', 'metrics.jsonl'):\n"
            "        if os.path.exists(os.path.join(src, name)): shutil.copy(os.path.join(src, name), f'outputs/{run}/')\n"
            "!ls outputs/*",
        ),
        ("md", "## Judge (5 pairs/request, ≥5 s between attempts, 3 attempts per batch, failures logged, resumable)"),
        (
            "code",
            "from alignment.judging import run_judging\n"
            "JOBS = [\n"
            "    ('dpo_uf_vs_sft',            'general'),\n"
            "    ('dpo_hh_vs_sft',            'safety'),\n"
            "    ('lora_vs_base_persona',     'persona_style'),\n"
            "    ('lora_vs_base_persona',     'persona_quality'),\n"
            "    ('dpo_uf_vs_sft_mirrored',   'general'),   # position-bias check (100 swapped pairs)\n"
            "    ('dpo_hh_vs_sft_mirrored',   'safety'),\n"
            "]\n"
            "for name, rubric in JOBS:\n"
            "    suffix = '' if rubric in ('general', 'safety') else f'_{rubric}'\n"
            "    run_judging(\n"
            "        pairs_path=f'outputs/eval/{name}_pairs.jsonl',\n"
            "        output_path=f'outputs/eval/{name}{suffix}_judged.jsonl',\n"
            "        judge_model=JUDGE, batch_size=5, sleep_seconds=5, max_retries=3,\n"
            "        reuse_judgments=True, rubric=rubric, backend='kbench',\n"
            "    )",
        ),
        (
            "code",
            "!python scripts/summarize_judgments.py --judged outputs/eval/dpo_uf_vs_sft_judged.jsonl --name 'UF-DPO vs SFT (AlpacaEval 300)' --mirror_judged outputs/eval/dpo_uf_vs_sft_mirrored_judged.jsonl\n"
            "!python scripts/summarize_judgments.py --judged outputs/eval/dpo_hh_vs_sft_judged.jsonl --name 'HH-DPO vs SFT (HH eval 300)' --mirror_judged outputs/eval/dpo_hh_vs_sft_mirrored_judged.jsonl\n"
            "!python scripts/summarize_judgments.py --judged outputs/eval/lora_vs_base_persona_persona_style_judged.jsonl outputs/eval/lora_vs_base_persona_persona_quality_judged.jsonl --name 'LoRA vs base: style' 'LoRA vs base: quality/safety'",
        ),
        ("md", "## Report assets: plots, tables, examples"),
        (
            "code",
            "!python scripts/plot_metrics.py --sft outputs/sft --dpo outputs/dpo_hh outputs/dpo_uf --lora outputs/lora_persona --output_dir outputs/plots\n"
            "!python scripts/collect_results.py --runs_root outputs --output outputs/report_tables.md\n"
            "!python scripts/show_examples.py --pairs outputs/eval/dpo_uf_vs_sft_pairs.jsonl --judged outputs/eval/dpo_uf_vs_sft_judged.jsonl --k 3 --output outputs/eval/examples_dpo_uf.md > /dev/null\n"
            "!python scripts/show_examples.py --pairs outputs/eval/dpo_hh_vs_sft_pairs.jsonl --judged outputs/eval/dpo_hh_vs_sft_judged.jsonl --k 3 --output outputs/eval/examples_dpo_hh.md > /dev/null\n"
            "!python scripts/show_examples.py --pairs outputs/eval/lora_vs_base_persona_pairs.jsonl --judged outputs/eval/lora_vs_base_persona_persona_style_judged.jsonl --k 2 --output outputs/eval/examples_lora_style.md > /dev/null",
        ),
        ("code", "!python scripts/analyze_generations.py outputs/generations/*.json --output outputs/eval/generation_stats.json"),
        ("code", "!cd {REPO_DIR} && zip -qr /kaggle/working/06_final_results.zip outputs && ls -lh /kaggle/working/06_final_results.zip"),
    ],
)

if __name__ == "__main__":
    for name, nb in NOTEBOOKS.items():
        nbf.write(nb, HERE / name)
        print("wrote", HERE / name)
