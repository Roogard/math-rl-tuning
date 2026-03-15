"""
Dataset loading, filtering, preprocessing, and chat-template formatting.

Handles:
  - Loading AI-MO/NuminaMath-CoT from Hugging Face
  - Source-based filtering (keep/drop specific math sources)
  - Balanced per-source sampling for train/val splits
  - System prompt injection into message lists
  - GRPO prompt formatting
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from math_rl_tuning.config import Config, DatasetConfig
from math_rl_tuning.utils import extract_boxed


# ---------------------------------------------------------------------------
# Core dataset loading
# ---------------------------------------------------------------------------

def load_numina_dataset(dataset_name: str = "AI-MO/NuminaMath-CoT"):
    """Load the NuminaMath-CoT dataset from Hugging Face."""
    return load_dataset(dataset_name)


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_sources(df: pd.DataFrame, drop_sources: List[str]) -> pd.DataFrame:
    """Remove rows whose 'source' column is in *drop_sources*."""
    return df[~df["source"].isin(drop_sources)].copy()


def keep_sources(df: pd.DataFrame, keep: List[str]) -> pd.DataFrame:
    """Keep only rows whose 'source' column is in *keep*."""
    return df[df["source"].isin(keep)].copy()


# ---------------------------------------------------------------------------
# Balanced per-source sampling
# ---------------------------------------------------------------------------

def balanced_split(
    df: pd.DataFrame,
    train_per_source: int = 6000,
    val_per_source: int = 1000,
    random_state: int = 42,
    per_source_overrides: Optional[Dict[str, int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample up to *train_per_source* training rows and *val_per_source*
    validation rows from each source group.

    Per-source caps can be set via *per_source_overrides* dict, which
    maps source names to their specific training cap. Sources not in
    the dict use the global *train_per_source* default.

    Returns (train_df, val_df) — both shuffled.
    """
    train_blocks, val_blocks = [], []
    overrides = per_source_overrides or {}

    for src, grp in df.groupby("source"):
        grp = grp.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(grp)

        cap = overrides.get(src, train_per_source)
        n_train = min(cap, n)
        n_val = min(val_per_source, n - n_train)

        train_blocks.append(grp.iloc[:n_train])
        val_blocks.append(grp.iloc[n_train : n_train + n_val])

        print(f"  {src}: train={n_train}, val={n_val}")

    train_df = pd.concat(train_blocks, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    )
    val_df = pd.concat(val_blocks, ignore_index=True).sample(
        frac=1.0, random_state=random_state
    )

    return train_df, val_df


# ---------------------------------------------------------------------------
# Answer format normalization
# ---------------------------------------------------------------------------

def _ensure_boxed_in_messages(example: dict) -> dict:
    r"""Ensure the assistant's final answer is wrapped in ``\boxed{}``.

    Handles three cases in order:
    1. Already has ``\boxed{}`` → no-op.
    2. Has GSM8K ``####`` marker → replaces with ``\boxed{}``.
    3. Neither → falls back to extracting ``\boxed{}`` from the ``solution``
       field (available as a separate column in NuminaMath-CoT).
    """
    msgs = example["messages"]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "assistant":
            content = msgs[i]["content"]
            if "\\boxed{" in content:
                break  # already correct
            if "####" in content:
                parts = content.rsplit("####", 1)
                answer = parts[1].strip()
                msgs[i]["content"] = (
                    parts[0].rstrip()
                    + f"\n\nThe answer is $\\boxed{{{answer}}}$."
                )
                break
            # Fallback: pull the answer from the raw solution field
            fallback = extract_boxed(example.get("solution", ""))
            if fallback:
                msgs[i]["content"] = (
                    content.rstrip()
                    + f"\n\nThe answer is $\\boxed{{{fallback}}}$."
                )
            break
    return {"messages": msgs}


def _has_valid_boxed_answer(example: dict) -> bool:
    r"""Return True if the assistant message contains a non-empty ``\boxed{}`` answer."""
    msgs = example.get("messages", [])
    for msg in reversed(msgs):
        if msg["role"] == "assistant":
            answer = extract_boxed(msg["content"])
            return answer is not None and len(answer.strip()) > 0
    return False


# ---------------------------------------------------------------------------
# System prompt injection
# ---------------------------------------------------------------------------

def inject_system_prompt(example: dict, system_prompt: str) -> dict:
    """
    Prepend a system message to the example's ``messages`` field.

    Uses a proper ``system`` role so the tokenizer's chat template places
    the instruction in the system slot (supported by Qwen2.5 and Llama-3).
    Skips if a system message is already present.
    """
    msgs = example["messages"]
    if not msgs or msgs[0]["role"] != "system":
        return {"messages": [{"role": "system", "content": system_prompt}] + msgs}
    return {"messages": msgs}


# ---------------------------------------------------------------------------
# GRPO prompt formatting
# ---------------------------------------------------------------------------

def _extract_gsm8k_answer(text: str) -> str:
    """Extract the numerical answer from GSM8K format (#### marker)."""
    if "####" not in text:
        return text.strip()
    return text.split("####")[-1].strip()


def format_grpo_prompt_gsm8k(
    example: dict,
    system_prompt: str,
) -> dict:
    """
    Format a GSM8K example for GRPO training.

    Returns a dict with:
      - ``prompt``: list of message dicts (TRL applies chat template)
      - ``answer``: the extracted ground-truth numerical answer

    NOTE: prompt MUST be message dicts, not a pre-rendered string.
    TRL's GRPOTrainer applies the chat template internally.
    Passing a pre-rendered string causes double-templating issues.
    """
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["question"]},
        ],
        "answer": _extract_gsm8k_answer(example["answer"]),
    }


def format_grpo_prompt_numina(
    example: dict,
    system_prompt: str,
) -> dict:
    """
    Format a NuminaMath example for GRPO training.

    Returns a dict with:
      - ``prompt``: list of message dicts (TRL applies chat template)
      - ``answer``: the extracted ground-truth answer from \\boxed{}
    """
    solution = example["solution"]
    answer = extract_boxed(solution) or solution.strip()

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ],
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# High-level convenience: prepare everything for SFT
# ---------------------------------------------------------------------------

def prepare_sft_data(cfg: Config):
    """
    End-to-end data preparation for SFT training.

    Returns (train_ds, val_ds) as HuggingFace ``Dataset`` objects,
    already filtered, balanced, system-prompt-injected, and
    chat-template-formatted.
    """
    dc = cfg.dataset

    print("Loading dataset...")
    ds = load_numina_dataset(dc.name)

    # Inject system prompt
    print("Injecting system prompt...")
    ds = ds.map(
        lambda ex: inject_system_prompt(ex, dc.system_prompt)
    )

    # Ensure all assistant answers use \boxed{} (converts GSM8K's #### format)
    print("Normalizing answer format to \\boxed{}...")
    ds = ds.map(_ensure_boxed_in_messages)

    # Filter out examples without valid \boxed{} answers
    before_count = len(ds["train"])
    ds["train"] = ds["train"].filter(_has_valid_boxed_answer)
    after_count = len(ds["train"])
    print(f"Data quality filter: {before_count} → {after_count} ({before_count - after_count} removed)")

    df = ds["train"].to_pandas()

    # Filter to keep only clean sources
    all_sources = set(df["source"].unique())
    drop_sources = list(all_sources - set(dc.sft_keep_sources))
    df = filter_sources(df, drop_sources)

    print(f"Sources kept: {sorted(df['source'].unique())}")
    print("Building balanced splits...")

    train_df, val_df = balanced_split(
        df,
        train_per_source=dc.train_per_source,
        val_per_source=dc.val_per_source,
        random_state=dc.random_state,
        per_source_overrides=getattr(dc, "train_per_source_overrides", None),
    )

    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # Note: SFTTrainer handles chat template application internally when
    # dataset_text_field="messages". No manual template step needed here.

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# High-level convenience: prepare everything for GRPO
# ---------------------------------------------------------------------------

def prepare_grpo_data(cfg: Config, tokenizer=None):
    """
    End-to-end data preparation for GRPO training.

    Loads GSM8K (default) or NuminaMath depending on
    ``cfg.dataset.grpo_dataset_name``.

    Returns a shuffled HuggingFace ``Dataset`` of size
    ``cfg.grpo_training.grpo_sample_size`` with ``prompt`` and ``answer``
    columns.  ``prompt`` is a list of message dicts (not a string).
    """
    dc = cfg.dataset
    gc = cfg.grpo_training
    grpo_ds_name = getattr(dc, "grpo_dataset_name", "openai/gsm8k")

    if "gsm8k" in grpo_ds_name.lower():
        return _prepare_grpo_gsm8k(dc, gc)
    else:
        return _prepare_grpo_numina(dc, gc)


def _prepare_grpo_gsm8k(dc, gc):
    """Load and format GSM8K for GRPO training."""
    print(f"Loading GSM8K dataset for GRPO...")
    ds = load_dataset(dc.grpo_dataset_name, "main")
    train_ds = ds["train"]

    print(f"GRPO candidate pool: {len(train_ds)} examples")
    print("Formatting prompts...")

    dataset = train_ds.map(
        lambda ex: format_grpo_prompt_gsm8k(ex, dc.grpo_system_prompt),
        remove_columns=train_ds.column_names,
    )

    dataset = dataset.shuffle(seed=dc.random_state).select(
        range(min(gc.grpo_sample_size, len(dataset)))
    )

    print(f"GRPO dataset size: {len(dataset)}")
    return dataset


def _prepare_grpo_numina(dc, gc):
    """Load and format NuminaMath-CoT for GRPO training (fallback)."""
    print("Loading NuminaMath dataset for GRPO...")
    ds = load_numina_dataset(dc.name)

    train_df = ds["train"].to_pandas()
    train_df = filter_sources(train_df, dc.grpo_drop_sources)
    train_ds = Dataset.from_pandas(train_df)

    print(f"GRPO candidate pool: {len(train_ds)} examples")
    print("Formatting prompts...")

    dataset = train_ds.map(
        lambda ex: format_grpo_prompt_numina(ex, dc.grpo_system_prompt),
        remove_columns=train_ds.column_names,
    )

    dataset = dataset.shuffle(seed=dc.random_state).select(
        range(min(gc.grpo_sample_size, len(dataset)))
    )

    print(f"GRPO dataset size: {len(dataset)}")
    return dataset


# ---------------------------------------------------------------------------
# High-level convenience: prepare test data
# ---------------------------------------------------------------------------

def prepare_test_data(cfg: Config) -> Dataset:
    """Load and filter the test split."""
    dc = cfg.dataset

    ds = load_numina_dataset(dc.name)
    test_df = ds["test"].to_pandas()
    test_df = filter_sources(test_df, dc.test_drop_sources)

    # Optionally filter to only sources the model was trained on
    eval_keep = getattr(cfg.evaluation, "eval_keep_sources", [])
    if eval_keep:
        test_df = keep_sources(test_df, eval_keep)
        print(f"Eval source filter active — keeping only: {eval_keep}")

    if len(test_df) < cfg.evaluation.num_samples:
        print(
            f"WARNING: Test set ({len(test_df)} examples) is smaller than "
            f"evaluation.num_samples ({cfg.evaluation.num_samples}). "
            f"All {len(test_df)} examples will be used."
        )

    print(f"Test set size: {len(test_df)}")
    print(f"Test set sources: {sorted(test_df['source'].unique())}")
    return Dataset.from_pandas(test_df)
