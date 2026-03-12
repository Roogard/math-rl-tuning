# Math RL Tuning — Claude Context

This file gives Claude Code a quick orientation to the project at the start of each session.
Full version-by-version details live in [project_spec.md](project_spec.md).

---

## Project Goals

This is a resume side project demonstrating applied RL for LLM fine-tuning. The core goal is to take Qwen2.5-Math-7B-Instruct and make it significantly better at math by:

1. Running SFT on curated math problems (NuminaMath-CoT)
2. Running GRPO reinforcement learning on GSM8K to improve reasoning
3. Proving the improvement with a before/after evaluation
4. Shipping a public demo that employers can view

The owner wants to understand every component well enough to discuss it in an interview, but the immediate priority is working results over deep dives.

---

## Project Architecture

Two-stage training pipeline, both stages run on **Google Colab**:

```
Qwen2.5-Math-7B-Instruct
        │
        ▼  SFT (TRL SFTTrainer, QLoRA r=32)
        │  Dataset: NuminaMath-CoT (5 sources, ~60k examples, 2 epochs)
        ▼
  outputs/sft/  ← LoRA adapter
        │
        ▼  merge_adapter()  ← required step before GRPO
        ▼
  outputs/sft_merged/  ← full weights
        │
        ▼  GRPO (TRL GRPOTrainer + Unsloth FastLanguageModel, LoRA r=32)
        │  Dataset: GSM8K  |  Reward: correctness + \boxed{} format
        ▼
  outputs/grpo/  ← LoRA adapter
        │
        ▼  Evaluation: base / SFT / GRPO accuracy on NuminaMath test set
        ▼
  Static Gradio demo on HuggingFace Spaces (pre-computed answers, free tier)
```

Key source files:
- `math_rl_tuning/sft_trainer.py` — SFT pipeline
- `math_rl_tuning/grpo_trainer.py` — GRPO pipeline
- `math_rl_tuning/model.py` — model loading, merging, Unsloth loading
- `math_rl_tuning/rewards.py` — 5 GRPO reward functions
- `math_rl_tuning/evaluation.py` — accuracy comparison
- `configs/default.yaml` — all hyperparameters
- `notebooks/` — Colab-ready thin wrappers over the package

---

## Constraints and Policies

**Compute**: All training runs on Google Colab. Do not suggest local GPU training or cloud alternatives — Colab credits are available. Notebooks in `notebooks/` are the interface.

**Commits**: Never commit automatically. Always ask before committing anything.

**Retraining decisions**: If evaluation results are poor (GRPO does not beat SFT by +5%), summarize findings and present options — do not retrain without user approval. See v1.5 in [project_spec.md](project_spec.md).

**Over-engineering**: This is a focused side project. Don't add abstractions, extra configs, or features beyond what the current version milestone requires.

**Critical GRPO gotchas** (do not change without good reason):
- Import `unsloth` before `trl` — required for metric logging
- Do NOT pass `max_seq_length` to `GRPOConfig` — causes tensor-shape bug
- Call `patch_vocab_size()` after any adapter merge
- Make sure reward is tracked in training so that the user can look at all info (reward, training accuracy, reward std, etc.) and make deductions based on them

---

## SFT Overhaul (Latest Changes)

Previous SFT was getting 59% accuracy while the base model got 65% — a performance degradation caused by catastrophic forgetting. After researching NuminaMath, OpenR1, Qwen2.5-Math, and DeepSeekMath, the following 10 changes were made:

| Change | Before | After | Why |
|---|---|---|---|
| Base model | Qwen2.5-7B-Instruct | **Qwen2.5-Math-7B-Instruct** | Math-specialized, avoids overwriting general alignment |
| LoRA rank | r=16, alpha=64, dropout=0.05 | **r=32, alpha=64, dropout=0.0** | More capacity for math; gentler scaling (alpha/r=2) |
| Dataset sources | 3 (gsm8k, math, cn_k12) | **5 (+orca_math, synthetic_math)** | More diversity |
| Dataset size | ~18k (6k/source) | **~60k (12k/source)** | Reference projects use 94k-860k |
| Epochs | 1 | **2** | Reference projects use 3-4 |
| Learning rate | 2e-5 | **5e-5** | LoRA benefits from higher LR; OpenR1 uses 5e-5 |
| Warmup | 10 steps (<1%) | **10% ratio** | Proper warmup scaling |
| NEFTune noise | alpha=5.0 | **Disabled** | Harmful for math precision; no reference project uses it |
| Eval during training | Disabled | **Every 100 steps + load_best_model_at_end** | Detect overfitting |
| Data quality filter | None | **Filter examples missing valid \boxed{} answer** | Remove garbage training data |

**Files changed:** `configs/default.yaml`, `math_rl_tuning/config.py`, `math_rl_tuning/sft_trainer.py`, `math_rl_tuning/data.py`

---

## Current Status

| Stage | Status |
|---|---|
| SFT training | Re-running with new config (targeting 70%+) |
| Merge SFT adapter | Not started |
| GRPO training | Not started |
| Baseline evaluation | Not started |
| v2 demo | Not started |

**Next steps:**
1. Run SFT with new config, verify accuracy > 65% (target 70%+)
2. If LR 5e-5 spikes eval loss, fall back to 3e-5
3. Run GRPO on the new SFT model
4. Full pipeline evaluation (base vs SFT vs GRPO)

Update this table as milestones are completed.

---

## Documentation Update Policy

**At the end of every version milestone, before closing out the work, update all three of these files:**

- [changelog.md](changelog.md) — add a new section for the completed version; list what changed, what was fixed, and what configs/code was modified
- [project_status.md](project_status.md) — mark the milestone as done, move to the next milestone, update the metrics table with any new accuracy numbers, and revise the "Immediate Next Steps" section
- [architecture.md](architecture.md) — update if any system design changed (new modules, changed data flow, new constraints)

Do not skip this step. These files are the project memory.

---

## Key References

- [project_spec.md](project_spec.md) — full version roadmap with task checklists and success criteria
- [architecture.md](architecture.md) — full system design, module responsibilities, data flow, config details
- [changelog.md](changelog.md) — history of all changes by version
- [project_status.md](project_status.md) — current milestone, accomplishments, next steps, accuracy metrics
- `.env` — credentials needed (HF_TOKEN, WANDB_API_KEY)
