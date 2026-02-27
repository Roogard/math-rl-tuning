# System Architecture

## Overview

Math RL Tuning is a two-stage fine-tuning pipeline that takes a base instruction-following LLM and trains it to reason through math problems. Stage 1 (SFT) teaches the format and style. Stage 2 (GRPO) uses reinforcement learning to improve correctness through trial and reward.

---

## Pipeline Flow

```
[HuggingFace Hub]
  Qwen/Qwen2.5-7B-Instruct
        │
        │  load_model_and_tokenizer(cfg, stage="sft")
        │  - 4-bit NF4 QLoRA (bitsandbytes)
        │  - LoRA adapter: r=16, alpha=64, all-linear targets
        ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1: SFT  (sft_trainer.py)                     │
│                                                     │
│  Dataset: AI-MO/NuminaMath-CoT                      │
│    → filter to gsm8k + math sources                 │
│    → balanced_split: 6000 train / 1000 val          │
│    → inject system prompt as system role message    │
│      "reason step by step, put answer in \boxed{}"  │
│    → TRL SFTTrainer applies chat template via       │
│      dataset_text_field="messages"                  │
│                                                     │
│  Trainer: TRL SFTTrainer                            │
│    → completion-only loss (ignores prompt tokens)   │
│    → 1 epoch, lr=2e-5, cosine schedule              │
│    → NEFTune noise alpha=5.0                        │
│    → max_seq_length=1024                            │
│                                                     │
│  Output: outputs/sft/  (LoRA adapter)               │
└──────────────────────────┬──────────────────────────┘
                           │
                           │  merge_adapter(cfg, adapter_path, save_path)
                           │  - loads base model in fp16
                           │  - PeftModel.from_pretrained → merge_and_unload()
                           │  - saves full weights + tokenizer
                           │  - patch_vocab_size() fixes config.json mismatch
                           ▼
                    outputs/sft_merged/
                    (full model weights, ~14GB)
                           │
                           │  load_unsloth_model(cfg, model_path)
                           │  - FastLanguageModel.from_pretrained
                           │  - attaches fresh LoRA: r=32, alpha=32
                           │  - targets: q/k/v/o_proj + gate/up/down_proj
                           ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2: GRPO  (grpo_trainer.py)                   │
│                                                     │
│  Dataset: openai/gsm8k                              │
│    → format_grpo_prompt_gsm8k()                     │
│    → system prompt: use <reasoning> + <answer> XML  │
│    → up to 5000 samples                             │
│                                                     │
│  Trainer: TRL GRPOTrainer (NOT UnslothGRPOTrainer)  │
│    → 8 completions per prompt                       │
│    → max_completion_length=256                      │
│    → lr=5e-6, beta=0.01, max_grad_norm=0.1          │
│    → paged AdamW 8-bit                              │
│                                                     │
│  Reward (rewards.py — 5 functions):                 │
│    1. correctness_reward_func  → 2.0 (exact match)  │
│    2. int_reward_func          → 0.5 (is integer?)  │
│    3. strict_format_reward_func→ 0.5 (exact XML)    │
│    4. soft_format_reward_func  → 0.5 (relaxed XML)  │
│    5. xmlcount_reward_func     → 0.5 (per-tag score)│
│                                                     │
│  Output: outputs/grpo/  (LoRA adapter)              │
└──────────────────────────┬──────────────────────────┘
                           │
                           │  evaluate_model() / compare_models()
                           │  - greedy decoding (deterministic)
                           │  - extract \boxed{} from predictions
                           │  - compare: exact match OR latex2sympy2 equality
                           ▼
              Accuracy: base / SFT / GRPO on GSM8K test set
                           │
                           ▼
              Static Gradio demo → HuggingFace Spaces (v2)
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `config.py` | Load `configs/default.yaml`, deep-merge overrides, return typed `Config` dataclass |
| `data.py` | Dataset loading, filtering, balanced sampling, system prompt injection, GRPO formatting |
| `model.py` | QLoRA loading, LoRA attach/merge, Unsloth loading, vocab size patching |
| `sft_trainer.py` | End-to-end SFT: data → model → train → save |
| `grpo_trainer.py` | End-to-end GRPO: merge → Unsloth load → data → train → save |
| `rewards.py` | 5 GRPO reward functions; format: `[[{"role":"assistant","content":"..."}], ...]` |
| `evaluation.py` | Greedy generation, answer extraction, accuracy metrics, model comparison |
| `inference.py` | One-shot and streaming generation for interactive use |
| `utils.py` | Memory cleanup, Colab detection, HF/WandB auth, answer extraction helpers |

---

## Configuration System

All hyperparameters are in `configs/default.yaml`. Typed dataclasses in `config.py` mirror every section. Priority order when loading:

```
defaults (default.yaml) → custom YAML (--config flag) → CLI overrides (dict)
```

Key config sections and what they control:

| Section | Controls |
|---|---|
| `model` | Base model name, `max_seq_length` (1024) |
| `quantization` | 4-bit NF4, compute dtype (auto bfloat16/float16) |
| `lora.sft` | r=16, alpha=64, all-linear, dropout=0.05 |
| `lora.grpo` | r=32, alpha=32, specific attention+MLP modules |
| `dataset` | Source filters, train/val split sizes, system prompts |
| `sft_training` | Epochs, lr, batch size, NEFTune, WandB project |
| `grpo_training` | lr, beta, num_generations, max_completion_length, sample_size |
| `rewards` | Per-function reward weights (correctness=3.0, format bonuses) |
| `evaluation` | num_samples, max_new_tokens, decoding strategy |

---

## Answer Format by Stage

| Stage | Format | System Prompt |
|---|---|---|
| SFT | `\boxed{answer}` in free text | "Reason step by step, put final answer in \\boxed{}" |
| GRPO | `<reasoning>...</reasoning><answer>...</answer>` | Structured XML instructions |
| Evaluation | Extracts `\boxed{}` from ground truth; compares with exact match or symbolic equality | — |

---

## Key Technical Constraints

- **Unsloth import order**: `unsloth` must be imported before `trl` everywhere — it patches TRL internals for metric logging.
- **`max_seq_length` in GRPO**: Pass to `FastLanguageModel.from_pretrained`, NOT to `GRPOConfig`. The GRPOConfig only takes `max_completion_length`.
- **Pad token**: Always `eos_token` — adding a `[PAD]` token resizes embeddings and can cause CUDA asserts.
- **Post-merge patch**: `patch_vocab_size()` must be called after `merge_adapter()` to fix the `config.json` vocab mismatch.
- **Reward tracking**: `RewardTracker` + `RewardLoggingCallback` in `grpo_trainer.py` ensure all 5 reward function means appear in WandB at every training step.
- **Dead code**: `apply_mistral_chat_template()` in `data.py` is no longer used. TRL's `SFTTrainer` applies the chat template automatically via `dataset_text_field="messages"` and the tokenizer's built-in template.
