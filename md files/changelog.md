# Changelog

Changes are listed newest-first. Update this file at the end of every version milestone.

---

## [In Progress] — MVP

### Model Switch: Mistral → Qwen2.5-7B-Instruct
- Switched base model from `mistralai/Mistral-7B-Instruct-v0.2` to `Qwen/Qwen2.5-7B-Instruct`
- Root cause: Unsloth GRPO kernel has hardcoded Llama-architecture assumptions (issue #1790: "New GRPO doesn't support models besides LLAMA - Mistral"), causing tensor shape mismatches at runtime
- Qwen2.5-7B-Instruct has full Unsloth GRPO support, higher math baseline (75.5% MATH benchmark vs 68.4% for Mistral), and is used in HuggingFace's Open-R1 (DeepSeek-R1 reproduction)
- Updated `configs/default.yaml`: `model.name`
- Updated `math_rl_tuning/data.py`: `inject_system_prompt` now prepends a proper `system` role message instead of prepending to user content — aligns with Qwen2.5's instruction-tuning format and ensures correct completion-only loss masking
- SFT adapter from previous Mistral run is discarded; SFT will be re-run on Qwen2.5
- No changes to `model.py`, `sft_trainer.py`, `grpo_trainer.py`, or `rewards.py` — all were already model-agnostic

### GRPO Bug Fixes (pre-run debugging)
- Fixed `load_unsloth_model`: now uses `cfg.model.max_seq_length` (1024) instead of computing from GRPO config (was incorrectly 512)
- Fixed tensor-shape bug: removed `max_seq_length` from `GRPOConfig`; now passes `max_completion_length` directly
- Restored Unsloth import order: `from unsloth import FastLanguageModel` must come before any `trl` import to enable metric logging
- Removed `UnslothGRPOTrainer` in favor of standard TRL `GRPOTrainer` (Unsloth subclass had tensor-shape bug)
- Added `RewardTracker` wrapper and `RewardLoggingCallback` to log per-function reward means to WandB
- Cleaned up `rewards.py`

### Config Updates
- `max_seq_length` set to 1024 (matched official Unsloth Mistral notebook)
- `max_completion_length` set to `max_seq_length - max_prompt_length` (256)
- Reverted aggressive A100 scaling settings back to stable baseline (batch=1, grad_accum=4, num_gen=4)

---

## [Done] — SFT Training

- SFT training completed on NuminaMath-CoT (gsm8k + math sources)
- LoRA adapter (r=16, alpha=64) saved to Google Drive
- Training ran for 1 epoch with NEFTune noise, cosine LR schedule
- Completion-only loss (prompt tokens masked)

---

## [Done] — Project Setup

- Initialized repository with full package structure (`math_rl_tuning/`)
- Config system: typed dataclasses + YAML loading with deep-merge overrides
- Colab support: Drive mounting, Colab secrets, `patch_colab_fileno()`
- Notebooks: 4 Colab-ready notebooks (SFT, GRPO, evaluation, inference)
- CLI scripts: `run_sft.py`, `run_grpo.py`, `run_eval.py`
- Documentation: `md files/CLAUDE.md`, `md files/project_spec.md`, `md files/architecture.md`
- `.env.example` added; `.env` added to `.gitignore`
