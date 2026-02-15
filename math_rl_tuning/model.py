"""
Model loading, quantization, and LoRA adapter setup.

Supports two backends:
  - Standard HuggingFace (transformers + peft + bitsandbytes)
  - Unsloth / FastLanguageModel (for GRPO with vLLM acceleration)
"""

import os
import json
import torch
from typing import Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from math_rl_tuning.config import Config


# ---------------------------------------------------------------------------
# BitsAndBytes quantization config builder
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_bnb_config(cfg: Config) -> BitsAndBytesConfig:
    """
    Create a BitsAndBytesConfig from the project config.

    The compute dtype is aligned with the GPU's precision capability:
    bfloat16 on Ampere+ GPUs, float16 otherwise.  This prevents the
    fp16 GradScaler from encountering bfloat16 tensors.
    """
    from math_rl_tuning.utils import is_bf16_supported

    qc = cfg.quantization
    # Auto-align compute dtype with GPU capability
    if qc.bnb_4bit_compute_dtype == "auto" or qc.bnb_4bit_compute_dtype == "bfloat16":
        compute_dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
    else:
        compute_dtype = _DTYPE_MAP.get(qc.bnb_4bit_compute_dtype, torch.float16)

    print(f"BnB compute dtype: {compute_dtype}")
    return BitsAndBytesConfig(
        load_in_4bit=qc.load_in_4bit,
        bnb_4bit_quant_type=qc.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qc.bnb_4bit_use_double_quant,
    )


# ---------------------------------------------------------------------------
# LoRA config builder
# ---------------------------------------------------------------------------

def build_lora_config(cfg: Config, stage: str = "sft") -> LoraConfig:
    """
    Build a PEFT LoraConfig for the given *stage* ('sft' or 'grpo').
    """
    lc = getattr(cfg.lora, stage)
    return LoraConfig(
        r=lc.r,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        bias=lc.bias,
        task_type=lc.task_type,
        target_modules=lc.target_modules,
    )


# ---------------------------------------------------------------------------
# Standard HF model loading (for SFT)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    cfg: Config,
    stage: str = "sft",
    inference: bool = False,
    model_path: Optional[str] = None,
) -> Tuple:
    """
    Load base model with QLoRA quantization and attach a LoRA adapter.

    Args:
        cfg: Project configuration.
        stage: 'sft' or 'grpo' — selects the LoRA config.
        inference: If True, skip LoRA wrapping (for loading a saved adapter later).
        model_path: Override the model name/path. Defaults to ``cfg.model.name``.
                    Use this to load a merged model for GRPO.

    Returns:
        (model, tokenizer)
    """
    mc = cfg.model
    name_or_path = model_path or mc.name

    print(f"Loading tokenizer: {name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Quantization
    bnb_config = build_bnb_config(cfg)

    print(f"Loading model: {name_or_path} (4-bit quantized)")
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        use_cache=inference,  # disable cache during training
    )

    # Resize embeddings if we added tokens
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    if not inference:
        model = prepare_model_for_kbit_training(model)

        # Attach LoRA adapter
        lora_config = build_lora_config(cfg, stage=stage)
        model = get_peft_model(model, lora_config)

        model.gradient_checkpointing_enable()
        model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Load a saved LoRA adapter onto the base model (for inference / evaluation)
# ---------------------------------------------------------------------------

def load_adapter(
    cfg: Config,
    adapter_path: str,
    base_model_name: Optional[str] = None,
    load_in_4bit: bool = True,
) -> Tuple:
    """
    Load a base model + a saved LoRA adapter for inference.

    Returns (model, tokenizer).
    """
    base_name = base_model_name or cfg.model.name

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = build_bnb_config(cfg) if load_in_4bit else None

    print(f"Loading base model: {base_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16 if not load_in_4bit else None,
    )

    # Resize to match adapter tokenizer
    base_model.resize_token_embeddings(len(tokenizer))

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Merge LoRA adapter into base model (needed before GRPO with Unsloth)
# ---------------------------------------------------------------------------

def merge_adapter(
    cfg: Config,
    adapter_path: str,
    save_path: Optional[str] = None,
) -> str:
    """
    Merge a LoRA adapter into the base model weights and save the result.

    This is required before GRPO training with Unsloth/vLLM because they
    need a clean model (no adapter mismatch).

    Args:
        cfg: Project configuration.
        adapter_path: Path to the saved LoRA adapter.
        save_path: Where to save the merged model. Defaults to
                   ``cfg.paths.sft_merged_dir``.

    Returns:
        The path where the merged model was saved.
    """
    save_path = save_path or cfg.paths.sft_merged_dir

    if os.path.exists(save_path) and os.listdir(save_path):
        print(f"Merged model already exists at {save_path} — skipping.")
        return save_path

    print("Merging SFT adapter into base model...")
    base_name = cfg.model.name

    # Load tokenizer from adapter to get correct vocab size
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    vocab_size = len(tokenizer)

    # Load base model in fp16 for accurate merging
    print(f"  Loading base model: {base_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"  Resizing embeddings to {vocab_size}...")
    base_model.resize_token_embeddings(vocab_size)

    print(f"  Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("  Merging weights...")
    model = model.merge_and_unload()

    os.makedirs(save_path, exist_ok=True)
    print(f"  Saving merged model to {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Cleanup
    del model, base_model
    torch.cuda.empty_cache()

    print("  Merge complete.")
    return save_path


# ---------------------------------------------------------------------------
# Unsloth model loading (for GRPO)
# ---------------------------------------------------------------------------

def load_unsloth_model(
    cfg: Config,
    model_path: str,
    for_training: bool = True,
) -> Tuple:
    """
    Load a model using Unsloth's FastLanguageModel for GRPO training
    with vLLM acceleration.

    Args:
        cfg: Project configuration.
        model_path: Path to the merged model (or HF model name).
        for_training: If True, attach a fresh LoRA adapter for RL.

    Returns:
        (model, tokenizer)
    """
    from unsloth import FastLanguageModel

    gc = cfg.grpo_training
    lc = cfg.lora.grpo

    print(f"Loading model with Unsloth: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=cfg.model.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if for_training:
        print("Attaching fresh LoRA adapter for RL...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lc.r,
            target_modules=lc.target_modules if isinstance(lc.target_modules, list)
                else [lc.target_modules],
            lora_alpha=lc.alpha,
            lora_dropout=lc.dropout,
            bias=lc.bias,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    return model, tokenizer


# ---------------------------------------------------------------------------
# Vocab size hotfix (for Unsloth config.json mismatch)
# ---------------------------------------------------------------------------

def patch_vocab_size(model_path: str, vocab_size: int = 32001):
    """
    Patch config.json to fix vocab_size mismatch that can occur
    after merging adapters with added tokens.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        print(f"  config.json not found at {model_path} — skipping patch.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    if config.get("vocab_size") != vocab_size:
        print(f"  Patching vocab_size: {config.get('vocab_size')} -> {vocab_size}")
        config["vocab_size"] = vocab_size
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        print(f"  vocab_size already correct ({vocab_size}).")
