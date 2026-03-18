"""
Shared utilities: memory management, answer extraction, environment helpers.
"""

import gc
import os
import re
import shutil
import sys
import torch
from typing import Optional


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def clean_memory(verbose: bool = True):
    """Aggressively clear GPU VRAM and Python garbage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if verbose:
        print("Memory cleared.")


def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_bf16_supported() -> bool:
    """Check if the current GPU supports bfloat16 (Ampere+ architecture)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.is_bf16_supported()


# ---------------------------------------------------------------------------
# Environment / Colab helpers
# ---------------------------------------------------------------------------

def is_colab() -> bool:
    """Detect whether we are running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_google_drive(mount_point: str = "/content/drive"):
    """Mount Google Drive if running in Colab."""
    if not is_colab():
        print("Not running in Colab — skipping Drive mount.")
        return
    from google.colab import drive
    drive.mount(mount_point)


def setup_hf_token(token: Optional[str] = None):
    """
    Set the HF_TOKEN environment variable.

    Looks for the token in this order:
        1. Explicit *token* argument
        2. ``HF_TOKEN`` environment variable (already set)
        3. Google Colab ``userdata`` secrets
    """
    if token:
        os.environ["HF_TOKEN"] = token
        return

    if os.environ.get("HF_TOKEN"):
        return

    if is_colab():
        try:
            from google.colab import userdata
            os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
            return
        except Exception:
            pass

    print(
        "Warning: No HF_TOKEN found. Set it via environment variable, "
        "Colab secrets, or pass it explicitly."
    )


def setup_wandb(project: str, key: Optional[str] = None):
    """Login to Weights & Biases and set the project."""
    import wandb

    if key:
        wandb.login(key=key)
    elif os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    else:
        wandb.login()  # interactive or from ~/.netrc

    os.environ["WANDB_PROJECT"] = project


def patch_colab_fileno():
    """
    Patch sys.stdout/stderr.fileno for Colab environments
    that break it (needed by some Unsloth / vLLM internals).
    """
    for stream, fd in [(sys.stdout, 1), (sys.stderr, 2)]:
        if not hasattr(stream, "fileno"):
            stream.fileno = lambda _fd=fd: _fd
        else:
            try:
                stream.fileno()
            except Exception:
                stream.fileno = lambda _fd=fd: _fd


# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    r"""
    Extract the LAST ``\boxed{...}`` content from *text*.

    We use a brace-counting loop rather than a regex because regex can't
    handle nested braces like ``\boxed{\dfrac{1}{2}}`` — a greedy pattern
    would close too early at the first ``}``. The loop tracks brace depth
    explicitly, so arbitrarily nested LaTeX expressions work correctly.

    Returns the LAST match (not the first) because some CoT responses
    revise their answer, and the final \boxed{} is the intended one.

    Returns None if no boxed answer is found.
    """
    if text is None:
        return None
    result = None
    start_tag = "\\boxed{"
    idx = text.find(start_tag)
    while idx != -1:
        depth = 1           # We're one level deep after consuming the opening '{'
        i = idx + len(start_tag)
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1  # Entering a nested brace group
            elif text[i] == '}':
                depth -= 1  # Closing a brace group
            i += 1
        if depth == 0:
            # depth == 0 means we found the matching closing brace
            result = text[idx + len(start_tag):i - 1].strip()
        idx = text.find(start_tag, idx + 1)
    return result


def copy_to_drive(source_dir: str, subdir: str, cfg):
    """Copy saved model to Google Drive (Colab only)."""
    if not is_colab():
        return
    mount_google_drive(cfg.paths.drive_mount)
    dest = os.path.join(cfg.paths.drive_save_dir, subdir)
    if os.path.exists(dest):
        print(f"Drive destination already exists: {dest}. Skipping.")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copytree(source_dir, dest)
    print(f"Copied to Drive: {dest}")


def math_verify_equal(gold_text: str, pred_text: str) -> bool:
    """
    Compare gold solution and model output using math-verify.

    Simple string comparison fails for mathematically equivalent expressions:
    "1/2", "0.5", and "\\frac{1}{2}" are all the same answer but != as strings.
    math-verify uses SymPy to verify symbolic equivalence, handling fractions,
    decimals, expressions, and even multiple-choice letters (via StringExtractionConfig).

    Pass the FULL text (not pre-extracted answers). math-verify handles
    extraction from \\boxed{} internally via LatexExtractionConfig.
    """
    if not gold_text or not pred_text:
        return False
    try:
        from math_verify import parse, verify
        from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig, StringExtractionConfig
        from latex2sympy2_extended.math_normalization import NormalizationConfig

        norm_config = NormalizationConfig(basic_latex=True, units=True)
        latex_config = LatexExtractionConfig(
            boxed_match_priority=0,
            normalization_config=norm_config,
        )
        configs = [latex_config, ExprExtractionConfig()]

        # Try LaTeX + Expr extraction (handles most math answers)
        gold_parsed = parse(gold_text, extraction_config=configs)
        pred_parsed = parse(pred_text, extraction_config=configs)
        if gold_parsed and pred_parsed:
            if verify(gold_parsed, pred_parsed):
                return True

        # Fallback: StringExtractionConfig for MCQ (A/B/C/D) answers
        gold_parsed = parse(gold_text, extraction_config=[StringExtractionConfig()])
        pred_parsed = parse(pred_text, extraction_config=[StringExtractionConfig()])
        if gold_parsed and pred_parsed:
            return bool(verify(gold_parsed, pred_parsed))

        return False
    except Exception:
        return False



