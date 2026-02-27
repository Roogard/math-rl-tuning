# Math RL Tuning — Project Spec

**Goal**: Fine-tune Mistral-7B using SFT + GRPO reinforcement learning to produce a math-capable LLM, then demonstrate measurable improvement through a public demo.

**Audience**: Personal reference + Claude Code context at the start of sessions. Will eventually be a side project to display on a resume. 

**Current status**: MVP in progress — SFT completed, GRPO not yet run.

---

## Tech Stack

| Layer | Tool / Model |
|---|---|
| Base model | Mistral-7B-Instruct-v0.2 |
| Quantization | 4-bit NF4 QLoRA (bitsandbytes) |
| Adapter method | LoRA (PEFT) |
| SFT training | TRL SFTTrainer |
| RL training | TRL GRPOTrainer + Unsloth FastLanguageModel |
| SFT dataset | NuminaMath-CoT (gsm8k + math sources) |
| GRPO dataset | GSM8K |
| Compute | Google Colab (with credits) |
| Experiment tracking | Weights & Biases |
| Demo hosting | HuggingFace Spaces |

---

## Versions

### MVP — Complete first full training run

**What**: Run the full two-stage pipeline end-to-end: SFT (done) → merge adapter → GRPO training.

**Tech involved**: Mistral-7B-Instruct-v0.2 · QLoRA (4-bit NF4) · LoRA (r=32 for GRPO) · TRL GRPOTrainer · Unsloth FastLanguageModel · Google Colab

**Tasks**:
- [x] SFT training on NuminaMath-CoT
- [ ] Merge SFT LoRA adapter into base model weights
- [ ] Run GRPO training on GSM8K dataset
- [ ] Save GRPO adapter to Google Drive

**Done when**: GRPO adapter is saved to Drive with no errors; WandB run shows training loss curve.

---

### v1 — Measurable accuracy improvement

**What**: Prove that GRPO improved the model's math reasoning, with numbers to back it up.

**Tech involved**: `run_eval.py` · GSM8K test set (1,319 problems) · greedy decoding · latex2sympy2 symbolic equality

**Tasks**:
- [ ] Record base model (no training) accuracy on GSM8K test set — this is the "before" baseline
- [ ] Record SFT model accuracy on GSM8K test set
- [ ] Record GRPO model accuracy on GSM8K test set
- [ ] Save all three numbers to a results file

**Done when**:  All three checkpoint accuracies (base / SFT / GRPO) are recorded and saved.

---

### v1.5 — Improvement

**What**: Only do if the GRPO model is not as good as expected, or somehow worse.

**Tech involved**: Mistral-7B-Instruct-v0.2 · QLoRA (4-bit NF4) · LoRA (r=32 for GRPO) · TRL GRPOTrainer · Unsloth FastLanguageModel · Google Colab

**Tasks**:
- [ ] Find out why accuracy is not as high as it should be
- [ ] Retrain SFT/GRPO Model (whichever is needed) to raise accuracy
- [ ] Summarize before and after accuracy
- [ ] Report to user and wait for user input to decide whether to repeat this step or not.

**Done when** GRPO accuracy beats SFT accuracy by at least +5 percentage points on the GSM8K test set.

### v2 — Static comparison demo (public, live URL)

**What**: A public web demo hosted on HuggingFace Spaces that shows the accuracy improvement and lets viewers compare model outputs side-by-side. No live inference — all answers pre-computed.

**Tech involved**: Gradio · HuggingFace Spaces (free CPU tier) · JSON pre-computed results

**Tasks**:
- [ ] Select ~25 GSM8K questions spanning easy / medium / hard difficulty
- [ ] Pre-compute answers from all three model versions (base / SFT / GRPO) for those questions
- [ ] Build Gradio app: accuracy bar chart + side-by-side answer comparison with difficulty filter
- [ ] Deploy to HuggingFace Spaces
- [ ] Verify public URL loads correctly

**Done when**: A public HuggingFace Spaces URL is live. It shows an accuracy bar chart (base → SFT → GRPO) and allows browsing pre-computed answers for easy / medium / hard questions across all three model versions.

---

### v3 — Harder dataset training

**What**: Train a new or extended model on harder math problems (NuminaMath-CoT competition problems), going beyond GSM8K-level arithmetic.

**Tech involved**: NuminaMath-CoT dataset (full, including olympiad/competition sources) · SFT + GRPO pipeline (approach TBD)

**Tasks**:
- [ ] Decide approach: extend existing GRPO model vs. run a fresh SFT + GRPO pipeline on NuminaMath
- [ ] Train on chosen harder dataset
- [ ] Evaluate on NuminaMath test set and record accuracy

**Done when**: A model is trained on NuminaMath-level problems and accuracy on a NuminaMath test split is recorded.

---

### v4 — Competitive performance

**What**: Achieve benchmark-competitive performance on the MATH dataset or NuminaMath eval — scores that are meaningful when compared to published results for 7B-class models.

**Tech involved**: TBD — may require curriculum training, longer RL runs, larger LoRA rank, or a different base model.

**Tasks**:
- [ ] Identify a specific published benchmark score to target (e.g., MATH dataset accuracy for 7B models)
- [ ] Iterate on training strategy to approach that target
- [ ] Evaluate and record final benchmark score

**Done when**: Benchmark score on MATH or NuminaMath eval is competitive with published results for similarly-sized open-source models.

---

### v5 — Enhanced website

**What**: Update the demo to show all trained model versions (v1 + v3/v4 models), allow browsing by preset difficulty, and possibly support live user-submitted questions.

**Tech involved**: TBD — may require GPU Spaces, HuggingFace Inference API, or other live inference solution depending on cost/feasibility.

**Tasks**:
- [ ] Add all trained model versions to the demo
- [ ] Add preset question browsing with difficulty/dataset filters
- [ ] Decide: static pre-computed answers vs. live inference for user custom questions
- [ ] (If live inference) upgrade hosting to support GPU

**Done when**: Website shows all model versions with clear navigation. Users can browse preset examples across difficulty levels and model versions.
