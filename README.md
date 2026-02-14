# Math RL Tuning

Fine-tune and RL-train LLMs for mathematical reasoning using **SFT + GRPO**.

## Overview

This project trains **Mistral-7B-Instruct-v0.2** to solve math problems through a two-stage pipeline:

1. **SFT (Supervised Fine-Tuning)** — Train on the [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) dataset using QLoRA adapters to learn step-by-step mathematical reasoning with `\boxed{}` formatted answers.

2. **GRPO (Group Relative Policy Optimization)** — Reinforce the SFT model with a custom reward function that scores completions on format compliance, answer correctness (exact + symbolic LaTeX matching), and reasoning length.

### Key Features

- **Modular Python package** — All logic lives in `math_rl_tuning/`, easily importable and testable.
- **Configurable via YAML** — Every hyperparameter in one place (`configs/default.yaml`). Override at runtime.
- **Colab-ready notebooks** — Thin wrappers in `notebooks/` that clone the repo, install deps, and call the package. Run on Colab GPUs with zero local setup.
- **Memory-efficient** — QLoRA (4-bit) quantization + LoRA adapters. Fits on a T4 (16GB) for SFT and A100 for GRPO.
- **Symbolic answer checking** — Uses `latex2sympy2` to verify mathematical equivalence beyond string matching.

## Project Structure

```
math-rl-tuning/
├── configs/
│   └── default.yaml              # All hyperparameters and paths
├── math_rl_tuning/
│   ├── __init__.py
│   ├── config.py                 # YAML config loader with typed dataclasses
│   ├── data.py                   # Dataset loading, filtering, chat templates
│   ├── model.py                  # Model loading, QLoRA, LoRA, merging
│   ├── sft_trainer.py            # SFT training pipeline
│   ├── grpo_trainer.py           # GRPO/RL training pipeline
│   ├── rewards.py                # Reward functions for GRPO
│   ├── evaluation.py             # Accuracy evaluation and model comparison
│   ├── inference.py              # Streaming and batch generation
│   └── utils.py                  # Memory management, answer extraction, helpers
├── notebooks/
│   ├── 01_sft_training.ipynb     # SFT training (run on Colab)
│   ├── 02_grpo_training.ipynb    # GRPO training (run on Colab)
│   ├── 03_evaluation.ipynb       # Evaluation and comparison
│   └── 04_inference.ipynb        # Interactive inference
├── scripts/
│   ├── run_sft.py                # CLI: run SFT training
│   ├── run_grpo.py               # CLI: run GRPO training
│   └── run_eval.py               # CLI: run evaluation
├── setup.py
├── requirements.txt
├── MANIFEST.in
├── .gitignore
└── README.md
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/math-rl-tuning.git
cd math-rl-tuning
pip install -e .
```

### 2. Run on Google Colab

Open any notebook from `notebooks/` in Colab. Each notebook:
1. Clones this repo
2. Installs dependencies
3. Imports from the `math_rl_tuning` package
4. Runs the pipeline step

**Recommended GPU tiers:**
| Notebook | Minimum GPU | Recommended |
|----------|------------|-------------|
| 01 SFT Training | T4 (16GB) | A100 (40GB) |
| 02 GRPO Training | A100 (40GB) | A100 (80GB) |
| 03 Evaluation | T4 (16GB) | T4 (16GB) |
| 04 Inference | T4 (16GB) | T4 (16GB) |

### 3. Run from Command Line

CLI scripts are provided in `scripts/` for running each stage:

```bash
# SFT training
python scripts/run_sft.py
python scripts/run_sft.py --epochs 2 --batch-size 8 --no-wandb

# GRPO training (requires SFT adapter from step above)
python scripts/run_grpo.py --sft-adapter ./outputs/sft
python scripts/run_grpo.py --sft-adapter ./outputs/sft --sample-size 1200

# Evaluation
python scripts/run_eval.py --sft-adapter ./outputs/sft
python scripts/run_eval.py --sft-adapter ./outputs/sft --rl-adapter ./outputs/grpo --num-samples 100
```

All scripts accept `--config path/to/custom.yaml` and `--hf-token` / `--wandb-key`.

### 4. Run from Python

```python
from math_rl_tuning.config import load_config
from math_rl_tuning.sft_trainer import run_sft_training

cfg = load_config()
trainer, model, tokenizer = run_sft_training(cfg)
```

## Configuration

All settings are in `configs/default.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `model` | Base model name, max sequence length |
| `quantization` | QLoRA 4-bit settings |
| `lora.sft` / `lora.grpo` | LoRA rank, alpha, target modules |
| `dataset` | Source filtering, train/val sizes, system prompts |
| `sft_training` | Epochs, batch size, learning rate, WandB |
| `grpo_training` | RL hyperparams, vLLM, generation count |
| `rewards` | Reward function weights (format, correctness, length) |
| `evaluation` | Number of test samples, generation params |
| `inference` | Temperature, top-p, max tokens |

Override programmatically:
```python
cfg = load_config()
cfg.sft_training.num_train_epochs = 3
cfg.lora.sft.r = 32
cfg.dataset.sft_keep_sources = ["gsm8k", "math", "cn_k12"]
```

Or pass a custom YAML:
```python
cfg = load_config("my_custom_config.yaml")
```

## Pipeline Details

### SFT Stage
- **Dataset:** NuminaMath-CoT (filtered to `gsm8k` + `math` by default, ~12K train / 2K val)
- **System prompt:** Injected to encourage `\boxed{}` format
- **Chat template:** Mistral-Instruct `<s>[INST] ... [/INST]` format
- **Model:** Mistral-7B-Instruct-v0.2 with 4-bit QLoRA
- **LoRA:** r=16, alpha=64, all-linear target modules

### GRPO Stage
- **Prerequisite:** Merge the SFT adapter into the base model (automated)
- **Engine:** Unsloth + vLLM for fast generation during RL
- **Reward function:**
  - +0.1 for responses > 200 characters (encourages reasoning)
  - +0.1 for using `\boxed{}` format
  - +1.0 for correct answer (exact match or symbolic LaTeX equality)
  - +0.5 for partial match (ground truth appears in completion)
- **LoRA:** Fresh r=16 adapter on attention + MLP projections

### Evaluation
- Extracts `\boxed{}` answers from both ground truth and predictions
- Normalizes and compares with string matching + `latex2sympy2` symbolic equality
- Side-by-side comparison showing where RL improved over SFT

## Authentication

The project needs two API keys:

1. **Hugging Face** — For downloading Mistral-7B. Set via:
   - Colab secrets: Add `HF_TOKEN` in Colab's Secrets panel
   - Environment: `export HF_TOKEN=hf_...`
   - Code: `setup_hf_token("hf_...")`

2. **Weights & Biases** (optional) — For training logging. Set via:
   - Environment: `export WANDB_API_KEY=...`
   - Code: `setup_wandb("project-name", key="...")`
   - Set `report_to: "none"` in config to disable

## License

MIT
