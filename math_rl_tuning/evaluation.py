"""
Evaluation module.

Provides:
  - Single-model accuracy evaluation (boxed answer extraction + comparison)
  - Head-to-head comparison of SFT vs RL models
  - Result saving to JSON/CSV
"""

import os
import json
import torch
import pandas as pd
from typing import Dict, Optional, Tuple
from tqdm import tqdm

from math_rl_tuning.config import Config
from math_rl_tuning.utils import extract_boxed, clean_memory, math_verify_equal


# ---------------------------------------------------------------------------
# Greedy generation for evaluation
# ---------------------------------------------------------------------------

def generate_greedy(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 512,
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
) -> str:
    """
    Generate a response using greedy decoding (deterministic).

    Uses the tokenizer's chat template so the prompt format matches
    whatever model is loaded (Qwen, Mistral, Llama, etc.).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


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

        # Generate
        output = generate_greedy(question, model, tokenizer, max_new_tokens)

        # Extract answers from \boxed{} first, then compare semantically
        gold_answer = extract_boxed(row["solution"])
        pred_answer = extract_boxed(output)

        # Compare using math-verify (semantic equality)
        is_correct = math_verify_equal(gold_answer, pred_answer)

        if is_correct:
            correct += 1

        results.append({
            "model": model_name,
            "problem_snippet": str(question)[:80],
            "ground_truth": gold_answer,
            "predicted": pred_answer,
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
# Evaluate the raw base model (no adapter)
# ---------------------------------------------------------------------------

def evaluate_base_model(
    cfg: Config,
    test_dataset,
    num_samples: int = 50,
    model_name: str = "Base",
) -> Tuple[pd.DataFrame, float]:
    """
    Load the raw base model (no LoRA adapter) and evaluate it.

    Useful as a baseline to measure how much SFT and GRPO improved things.
    """
    from math_rl_tuning.model import load_model_and_tokenizer

    print(f"\nLoading base model: {cfg.model.name}")
    model, tokenizer = load_model_and_tokenizer(cfg, stage="sft", inference=True)

    results_df, accuracy = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_samples=num_samples,
        max_new_tokens=cfg.evaluation.max_new_tokens,
        model_name=model_name,
    )

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
    include_base: bool = True,
) -> Dict:
    """
    Evaluate Base, SFT, and RL models side-by-side and print a comparison.

    Returns a dict with results DataFrames and accuracy scores.
    Set include_base=False to skip the base model evaluation.
    """
    print("=" * 50)
    print("  HEAD-TO-HEAD MODEL COMPARISON")
    print("=" * 50)

    results = {}

    # Evaluate Base (optional)
    if include_base:
        base_df, base_acc = evaluate_base_model(
            cfg=cfg,
            test_dataset=test_dataset,
            num_samples=num_samples,
        )
        results["base_results"] = base_df
        results["base_accuracy"] = base_acc

    # Evaluate SFT
    sft_df, sft_acc = evaluate_adapter(
        adapter_path=sft_adapter_path,
        test_dataset=test_dataset,
        cfg=cfg,
        num_samples=num_samples,
        model_name="SFT",
    )
    results["sft_results"] = sft_df
    results["sft_accuracy"] = sft_acc

    # Evaluate RL
    rl_df, rl_acc = evaluate_adapter(
        adapter_path=rl_adapter_path,
        test_dataset=test_dataset,
        cfg=cfg,
        num_samples=num_samples,
        model_name="RL (GRPO)",
    )
    results["rl_results"] = rl_df
    results["rl_accuracy"] = rl_acc

    # Summary
    print("\n" + "=" * 50)
    print("  RESULTS")
    print("=" * 50)
    if include_base:
        print(f"Base Accuracy:      {base_acc:.2%}")
        print(f"SFT Accuracy:       {sft_acc:.2%}  ({sft_acc - base_acc:+.2%} vs Base)")
    else:
        print(f"SFT Accuracy:       {sft_acc:.2%}")
    print(f"RL (GRPO) Accuracy: {rl_acc:.2%}  ({rl_acc - sft_acc:+.2%} vs SFT)")

    # Qualitative analysis: Base → SFT
    if include_base and base_df is not None and sft_df is not None:
        comp_base_sft = base_df[["problem_snippet", "ground_truth"]].copy()
        comp_base_sft["Base_correct"] = base_df["is_correct"].values
        comp_base_sft["SFT_correct"] = sft_df["is_correct"].values
        results["base_vs_sft"] = comp_base_sft

        sft_gains = comp_base_sft[(~comp_base_sft["Base_correct"]) & comp_base_sft["SFT_correct"]]
        sft_regressions = comp_base_sft[comp_base_sft["Base_correct"] & (~comp_base_sft["SFT_correct"])]
        print(f"\nSFT fixed {len(sft_gains)} Base errors, regressed on {len(sft_regressions)}.")

    # Qualitative analysis: SFT → RL
    if sft_df is not None and rl_df is not None:
        comp_sft_rl = sft_df[["problem_snippet", "ground_truth"]].copy()
        comp_sft_rl["SFT_correct"] = sft_df["is_correct"].values
        comp_sft_rl["RL_correct"] = rl_df["is_correct"].values
        results["sft_vs_rl"] = comp_sft_rl

        rl_gains = comp_sft_rl[(~comp_sft_rl["SFT_correct"]) & comp_sft_rl["RL_correct"]]
        rl_regressions = comp_sft_rl[comp_sft_rl["SFT_correct"] & (~comp_sft_rl["RL_correct"])]
        print(f"RL fixed {len(rl_gains)} SFT errors, regressed on {len(rl_regressions)}.")

    return results


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
        k: results[k]
        for k in ("base_accuracy", "sft_accuracy", "rl_accuracy")
        if k in results
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Save detailed results
    for key in ["base_results", "sft_results", "rl_results"]:
        df = results.get(key)
        if df is not None:
            csv_path = os.path.join(output_dir, f"{key}.csv")
            df.to_csv(csv_path, index=False)
            print(f"{key} saved to: {csv_path}")
