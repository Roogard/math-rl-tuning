# Math RL Tuning


## Intro

In the past couple years, many different people, teams, and labs have been trying to see how accurate LLM's can get at math problems. When I saw this, I got really curious about how the process worked. What did the training look like? How can you verify if a math answer is correct? Because of this, I decided to try to make it myself to learn how it works. From the beginning, the plan was to take an LLM, and train it through supervised fine-tuning (SFT) and group relative policy optimization (GRPO). Also as a side note, before this project, I didn't have any experience with agentic engineering, so I wanted to use the project to also learn alot about how to effectively use it. 

## Tech

To do this, I used VSCode and Claude Code to generate a large amount of this project. However, I still coded soem things by hand, as well as made all the decisions, set up the structure, ran the training, and did a large part of the debugging. 

For running SFT and GRPO, I ended up using Google Colab to train the models. This ended up being pretty inefficient, since I had to push to github every time I wanted to make one small change. However, I have the google student plan, so I really wanted to use the free credits.

Also, compute and budget was a large influencer on how this project turned out. I am a broke college TA, so I don't really have access to GPU clusters. I however did spend $70 on google colab credits due to training and testing models. Today, I would have used a lot less money since I understand what's going on in the project a lot more now. 

## Design Decisions (Success and failure included)

This project went through a lot of phases. It started out on 3 different colab notebooks, with all the code inside those notebooks. It was very inefficient, mostly AI generated, and a huge mess. Eventually, I had enough and split all the code into a github repository instead, which ended up being far more efficient. 

After this, my next answer was verifying LLM answers to see if they found the right answer. At first, I had reward functions for if the right answer was in boxed, and if the right answer was there at all. However, this ended up being a terrible design. Beyond this, the actual parser that tried to find if the right answer was in boxed was broken for a while, so I only ever ended up getting the partial answer. Due to this, for a couple days I threw out any kind of training and just ensured that pretty much any answer was capable of being found. The library 



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
│   └── 03_evaluation.ipynb       # Evaluation and comparison
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

### 3. Run from Python

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

## License

MIT
