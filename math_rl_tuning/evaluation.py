"""
Evaluation module.

Provides:
  - Single-model accuracy evaluation (boxed answer extraction + comparison)
  - Head-to-head comparison of SFT vs RL models
  - Result saving to JSON/CSV
"""

import gc
import os
import re
import json
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from math_rl_tuning.config import Config
from math_rl_tuning.utils import (
    extract_boxed,
    normalize_answer,
    extract_answer_letter,
    latex_equal,
    clean_memory,
)


# ---------------------------------------------------------------------------
# Greedy generation for evaluation
# ---------------------------------------------------------------------------

def generate_greedy(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
) -> str:
    """
    Generate a response using greedy decoding (deterministic).

    The question is wrapped in Mistral-Instruct format:
        <s>[INST] question [/INST]
    """
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluate a single model on the test set
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    num_samples: int = 50,
    max_new_tokens: int = 512,
    model_name: str = "model",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate a model on *num_samples* from *test_dataset*.

    Compares extracted \\boxed{} answers (with normalization and symbolic
    equality fallback).

    Args:
        model: The loaded model (already on device).
        tokenizer: The tokenizer.
        test_dataset: HuggingFace Dataset with 'problem' and 'solution' columns.
        num_samples: How many examples to evaluate.
        max_new_tokens: Max tokens to generate per example.
        model_name: Label for this model in the results DataFrame.
        random_state: Seed for reproducible sampling.

    Returns:
        (results_df, accuracy) — a DataFrame of per-example results and
        the overall accuracy as a float (0.0 – 1.0).
    """
    # Sample
    test_df = test_dataset.to_pandas() if hasattr(test_dataset, "to_pandas") else test_dataset
    if isinstance(test_df, pd.DataFrame):
        subset = test_df.sample(n=min(num_samples, len(test_df)), random_state=random_state)
    else:
        subset = test_df

    results = []
    correct = 0

    print(f"Evaluating {model_name} on {len(subset)} examples...")

    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        question = row["problem"]
        gold_raw = extract_boxed(row["solution"])
        gold = normalize_answer(gold_raw)

        # Generate
        output = generate_greedy(question, model, tokenizer, max_new_tokens)

        # Extract prediction
        pred_raw = extract_boxed(output)
        pred = normalize_answer(pred_raw)

        # Compare: exact match or symbolic equality
        is_correct = False
        if pred is not None and gold is not None:
            if pred == gold:
                is_correct = True
            elif latex_equal(pred, gold):
                is_correct = True

        if is_correct:
            correct += 1

        results.append({
            "model": model_name,
            "problem_snippet": str(question)[:80],
            "ground_truth": gold,
            "predicted": pred,
            "is_correct": is_correct,
            "full_output": output[:500],
        })

    accuracy = correct / len(subset) if len(subset) > 0 else 0.0
    print(f"{model_name} Accuracy: {accuracy:.4f}  ({correct}/{len(subset)})")

    return pd.DataFrame(results), accuracy


# ---------------------------------------------------------------------------
# Evaluate with adapter loading (loads model, evaluates, unloads)
# ---------------------------------------------------------------------------

def evaluate_adapter(
    adapter_path: str,
    test_dataset,
    cfg: Config,
    num_samples: int = 50,
    model_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, float]:
    """
    Load a LoRA adapter, evaluate it, then unload to free VRAM.

    Useful for comparing multiple models sequentially.
    """
    from math_rl_tuning.model import load_adapter

    name = model_name or os.path.basename(adapter_path)
    print(f"\nLoading adapter: {adapter_path}")

    model, tokenizer = load_adapter(cfg, adapter_path)

    results_df, accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_samples=num_samples,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        model_name=name,
    )

    # Cleanup
    del model, tokenizer
    clean_memory(verbose=True)

    return results_df, accuracy


# ---------------------------------------------------------------------------
# Head-to-head comparison
# ---------------------------------------------------------------------------

def compare_models(
    sft_adapter_path: str,
    rl_adapter_path: str,
    test_dataset,
    cfg: Config,
    num_samples: int = 50,
) -> Dict:
    """
    Evaluate SFT and RL models side-by-side and print a comparison.

    Returns a dict with results DataFrames and accuracy scores.
    """
    print("=" * 50)
    print("  HEAD-TO-HEAD MODEL COMPARISON")
    print("=" * 50)

    # Evaluate SFT
    sft_df, sft_acc = evaluate_adapter(
        adapter_path=sft_adapter_path,
        test_dataset=test_dataset,
        cfg=cfg,
        num_samples=num_samples,
        model_name="SFT",
    )

    # Evaluate RL
    rl_df, rl_acc = evaluate_adapter(
        adapter_path=rl_adapter_path,
        test_dataset=test_dataset,
        cfg=cfg,
        num_samples=num_samples,
        model_name="RL (GRPO)",
    )

    # Summary
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"SFT Accuracy:       {sft_acc:.2%}")
    print(f"RL (GRPO) Accuracy: {rl_acc:.2%}")
    print(f"Improvement:        {rl_acc - sft_acc:+.2%}")

    # Qualitative analysis
    if sft_df is not None and rl_df is not None:
        comparison = sft_df[["problem_snippet", "ground_truth"]].copy()
        comparison["SFT_answer"] = sft_df["predicted"]
        comparison["SFT_correct"] = sft_df["is_correct"]
        comparison["RL_answer"] = rl_df["predicted"]
        comparison["RL_correct"] = rl_df["is_correct"]

        # Cases where RL fixed an SFT mistake
        improved = comparison[
            (~comparison["SFT_correct"]) & comparison["RL_correct"]
        ]
        if not improved.empty:
            print(f"\nRL fixed {len(improved)} SFT errors:")
            print(improved.to_string(index=False))

        # Cases where RL regressed
        regressed = comparison[
            comparison["SFT_correct"] & (~comparison["RL_correct"])
        ]
        if not regressed.empty:
            print(f"\nRL regressed on {len(regressed)} examples.")

    return {
        "sft_results": sft_df,
        "sft_accuracy": sft_acc,
        "rl_results": rl_df,
        "rl_accuracy": rl_acc,
    }


# ---------------------------------------------------------------------------
# Save evaluation results
# ---------------------------------------------------------------------------

def save_results(
    results: Dict,
    output_dir: str,
):
    """Save evaluation results to disk as JSON and CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Save summary
    summary = {
        "sft_accuracy": results.get("sft_accuracy"),
        "rl_accuracy": results.get("rl_accuracy"),
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Save detailed results
    for key in ["sft_results", "rl_results"]:
        df = results.get(key)
        if df is not None:
            csv_path = os.path.join(output_dir, f"{key}.csv")
            df.to_csv(csv_path, index=False)
            print(f"{key} saved to: {csv_path}")
