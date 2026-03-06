"""
Evaluation module.

Provides:
  - Batched greedy generation for fast evaluation
  - Single-model accuracy evaluation (boxed answer extraction + comparison)
  - Head-to-head comparison of SFT vs RL models (with adapter swapping)
  - Result saving to JSON/CSV
"""

import os
import json
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from math_rl_tuning.config import Config
from math_rl_tuning.utils import extract_boxed, clean_memory, math_verify_equal


# ---------------------------------------------------------------------------
# Greedy generation (single example — kept for interactive use)
# ---------------------------------------------------------------------------

def generate_greedy(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
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
# Batched greedy generation (for fast evaluation)
# ---------------------------------------------------------------------------

def generate_greedy_batch(
    questions: List[str],
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
) -> List[str]:
    """
    Generate responses for multiple questions using batched greedy decoding.

    Left-pads inputs (required for correct batched generation) and processes
    questions in chunks of *batch_size*.
    """
    # Save original padding config
    orig_padding_side = tokenizer.padding_side
    orig_pad_token = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_outputs = []

    for i in tqdm(range(0, len(questions), batch_size), desc="Batches"):
        batch_questions = questions[i : i + batch_size]

        # Format each question as a chat message
        batch_prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch_questions
        ]

        # Tokenize with left-padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        prompt_lengths = [
            (inputs["attention_mask"][j] == 1).sum().item()
            for j in range(len(batch_questions))
        ]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode each output, skipping prompt tokens
        for j in range(len(batch_questions)):
            # The input length includes padding; use attention mask to find real prompt length
            input_len = inputs["input_ids"].shape[1]
            new_tokens = output_ids[j][input_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_outputs.append(text)

    # Restore original padding config
    tokenizer.padding_side = orig_padding_side
    tokenizer.pad_token_id = orig_pad_token

    return all_outputs


# ---------------------------------------------------------------------------
# Evaluate a single model on the test set
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    num_samples: int = 100,
    max_new_tokens: int = 1024,
    model_name: str = "model",
    random_state: int = 42,
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
    batch_size: int = 4,
) -> Tuple[pd.DataFrame, float]:
    """
    Evaluate a model on *num_samples* from *test_dataset*.

    Compares extracted \\boxed{} answers (with normalization and symbolic
    equality fallback).  Uses batched generation for speed.

    Args:
        model: The loaded model (already on device).
        tokenizer: The tokenizer.
        test_dataset: HuggingFace Dataset with 'problem' and 'solution' columns.
        num_samples: How many examples to evaluate.
        max_new_tokens: Max tokens to generate per example.
        model_name: Label for this model in the results DataFrame.
        random_state: Seed for reproducible sampling.
        system_prompt: System instruction prepended to the conversation.
        batch_size: Number of examples to process per batch.

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

    print(f"Evaluating {model_name} on {len(subset)} examples (batch_size={batch_size})...")

    # Collect all questions and generate in batches
    rows = list(subset.iterrows())
    questions = [row["problem"] for _, row in rows]
    outputs = generate_greedy_batch(
        questions, model, tokenizer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        system_prompt=system_prompt,
    )

    # Score results
    results = []
    correct = 0

    for (_, row), output in zip(rows, outputs):
        gold_answer = extract_boxed(row["solution"])
        pred_answer = extract_boxed(output)
        is_correct = math_verify_equal(gold_answer, pred_answer)

        if is_correct:
            correct += 1

        results.append({
            "model": model_name,
            "problem_snippet": str(row["problem"])[:80],
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
        system_prompt=cfg.evaluation.system_prompt,
        batch_size=cfg.evaluation.batch_size,
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
    num_samples: int = 100,
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
        system_prompt=cfg.evaluation.system_prompt,
        batch_size=cfg.evaluation.batch_size,
    )

    del model, tokenizer
    clean_memory(verbose=True)

    return results_df, accuracy


# ---------------------------------------------------------------------------
# Head-to-head comparison (loads base model once, swaps adapters)
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

    Loads the base model once and swaps LoRA adapters to avoid redundant
    model reloads.  Returns a dict with results DataFrames and accuracy scores.
    """
    from peft import PeftModel
    from math_rl_tuning.model import load_model_and_tokenizer

    print("=" * 50)
    print("  HEAD-TO-HEAD MODEL COMPARISON")
    print("=" * 50)

    ec = cfg.evaluation
    results = {}

    # Load base model once
    print(f"\nLoading base model: {cfg.model.name}")
    base_model, tokenizer = load_model_and_tokenizer(cfg, stage="sft", inference=True)

    # --- Evaluate Base (optional) ---
    if include_base:
        base_df, base_acc = evaluate_model(
            model=base_model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            num_samples=num_samples,
            max_new_tokens=ec.max_new_tokens,
            model_name="Base",
            system_prompt=ec.system_prompt,
            batch_size=ec.batch_size,
        )
        results["base_results"] = base_df
        results["base_accuracy"] = base_acc

    # --- Evaluate SFT (swap adapter onto base model) ---
    print(f"\nLoading SFT adapter: {sft_adapter_path}")
    sft_model = PeftModel.from_pretrained(base_model, sft_adapter_path)
    sft_model.eval()

    sft_df, sft_acc = evaluate_model(
        model=sft_model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_samples=num_samples,
        max_new_tokens=ec.max_new_tokens,
        model_name="SFT",
        system_prompt=ec.system_prompt,
        batch_size=ec.batch_size,
    )
    results["sft_results"] = sft_df
    results["sft_accuracy"] = sft_acc

    # Free SFT adapter, keep base model
    del sft_model
    clean_memory(verbose=False)

    # --- Evaluate RL (swap adapter onto base model) ---
    print(f"\nLoading RL adapter: {rl_adapter_path}")
    rl_model = PeftModel.from_pretrained(base_model, rl_adapter_path)
    rl_model.eval()

    rl_df, rl_acc = evaluate_model(
        model=rl_model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        num_samples=num_samples,
        max_new_tokens=ec.max_new_tokens,
        model_name="RL (GRPO)",
        system_prompt=ec.system_prompt,
        batch_size=ec.batch_size,
    )
    results["rl_results"] = rl_df
    results["rl_accuracy"] = rl_acc

    # Final cleanup
    del rl_model, base_model, tokenizer
    clean_memory(verbose=True)

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
