"""
Dataset loading, filtering, preprocessing, and chat-template formatting.

Handles:
  - Loading AI-MO/NuminaMath-CoT from Hugging Face
  - Source-based filtering (keep/drop specific math sources)
  - Balanced per-source sampling for train/val splits
  - System prompt injection into message lists
  - Mistral-style chat template formatting
  - GRPO prompt formatting
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from math_rl_tuning.config import Config, DatasetConfig


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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample up to *train_per_source* training rows and *val_per_source*
    validation rows from each source group.

    Returns (train_df, val_df) — both shuffled.
    """
    train_blocks, val_blocks = [], []

    for src, grp in df.groupby("source"):
        grp = grp.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        n = len(grp)

        n_train = min(train_per_source, n)
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
# Mistral chat template
# ---------------------------------------------------------------------------

def apply_mistral_chat_template(example: dict, **kwargs) -> dict:
    """
    Convert a list of messages into Mistral-Instruct chat format::

        <s>[INST] user message [/INST] assistant message</s>
    """
    messages = example["messages"]
    text_parts = []
    current = ""

    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "user":
            if current:
                text_parts.append(current)
            current = f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            current = current + f" {content}</s>"

    if current:
        text_parts.append(current)

    return {"text": "\n".join(text_parts)}


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
    from math_rl_tuning.utils import extract_xml_answer

    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["problem"]},
        ],
        "answer": extract_xml_answer(example["solution"]),
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

    print(f"Test set size: {len(test_df)}")
    return Dataset.from_pandas(test_df)
