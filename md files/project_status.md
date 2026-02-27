# Project Status

---

## Current Milestone: MVP

**Goal**: Complete the first full end-to-end training run (SFT → merge → GRPO).

**Next action**: Open `notebooks/02_grpo_training.ipynb` on Google Colab and run GRPO training.

---

## Accomplishments

| # | Accomplishment | Notes |
|---|---|---|
| ✅ | Project scaffolding | Full Python package, config system, CLI scripts, Colab notebooks |
| ✅ | SFT training completed | LoRA adapter saved to Google Drive |
| ✅ | GRPO bugs debugged | Tensor-shape fix, Unsloth import order, reward logging confirmed |

---

## Milestone Tracker

| Milestone | Status | Done When |
|---|---|---|
| **MVP** — Full training run | 🟡 In progress | GRPO adapter saved, WandB shows training loss |
| **v1** — Measurable improvement | ⬜ Not started | GRPO beats SFT by +5 pts on GSM8K test set; all 3 baselines recorded |
| **v1.5** — Improvement (if needed) | ⬜ Conditional | Only if GRPO fails to beat SFT by +5%; skip if v1 succeeds |
| **v2** — Public static demo | ⬜ Not started | HuggingFace Spaces URL live with accuracy chart + answer comparison |
| **v3** — Harder dataset training | ⬜ Not started | Model trained on NuminaMath-CoT competition problems; accuracy recorded |
| **v4** — Competitive performance | ⬜ Not started | Competitive score on MATH/NuminaMath benchmark vs. published 7B results |
| **v5** — Enhanced website | ⬜ Not started | All model versions on demo; difficulty + dataset filter; preset browsing |

---

## Immediate Next Steps (MVP → v1)

1. **Run GRPO on Colab** — `notebooks/02_grpo_training.ipynb`
   - Ensure SFT adapter is accessible from Google Drive
   - Check WandB for reward signal and training loss during run
2. **Capture base model baseline** — run `run_eval.py` on unmodified Mistral-7B before any GRPO inference
3. **Run full evaluation** — `run_eval.py --sft-adapter ... --rl-adapter ...`
4. **Record all three accuracy numbers** — base / SFT / GRPO on GSM8K test set

---

## Metrics (fill in as runs complete)

| Model | GSM8K Accuracy | Date | Notes |
|---|---|---|---|
| Base Mistral-7B-Instruct-v0.2 | — | — | Pre-training baseline |
| After SFT | — | — | NuminaMath-CoT fine-tune |
| After GRPO | — | — | GSM8K RL fine-tune |
