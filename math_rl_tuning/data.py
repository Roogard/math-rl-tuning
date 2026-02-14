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
    Prepend *system_prompt* to the first user message in the example's
    ``messages`` field.
    """
    msgs = example["messages"]
    if msgs and msgs[0]["role"] == "user":
        new_content = system_prompt + "\n\n" + msgs[0]["content"]
        new_msgs = [{"role": "user", "content": new_content}] + msgs[1:]
        return {"messages": new_msgs}
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

def format_grpo_prompt(
    example: dict,
    tokenizer,
    system_prompt: str,
) -> dict:
    """
    Format a single example for GRPO training.

    Returns a dict with:
      - ``prompt``: the tokenized prompt string (with generation prompt)
      - ``answer``: the extracted ground-truth answer from \\boxed{}
    """
    from math_rl_tuning.utils import extract_xml_answer

    raw_problem = example["problem"]
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{raw_problem}"}
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    ground_truth = example["solution"]
    ground_truth = extract_xml_answer(ground_truth)

    return {"prompt": prompt_text, "answer": ground_truth}


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

    # Apply Mistral chat template
    print("Applying chat template...")
    train_ds = train_ds.map(apply_mistral_chat_template)
    val_ds = val_ds.map(apply_mistral_chat_template)

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# High-level convenience: prepare everything for GRPO
# ---------------------------------------------------------------------------

def prepare_grpo_data(cfg: Config, tokenizer):
    """
    End-to-end data preparation for GRPO training.

    Returns a shuffled HuggingFace ``Dataset`` of size
    ``cfg.grpo_training.grpo_sample_size`` with ``prompt`` and ``answer``
    columns.
    """
    dc = cfg.dataset
    gc = cfg.grpo_training

    print("Loading dataset for GRPO...")
    ds = load_numina_dataset(dc.name)

    train_df = ds["train"].to_pandas()
    train_df = filter_sources(train_df, dc.grpo_drop_sources)
    train_ds = Dataset.from_pandas(train_df)

    print(f"GRPO candidate pool: {len(train_ds)} examples")
    print("Formatting prompts...")

    dataset = train_ds.map(
        lambda ex: format_grpo_prompt(ex, tokenizer, dc.grpo_system_prompt),
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
