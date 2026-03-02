"""
Shared utilities: memory management, answer extraction, environment helpers.
"""

import gc
import os
import re
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

    Handles nested braces correctly, e.g. ``\boxed{\dfrac{17}{32}}``
    extracts ``\dfrac{17}{32}`` instead of just ``\dfrac{17``.

    Returns None if no boxed answer is found.
    """
    if text is None:
        return None
    result = None
    start_tag = "\\boxed{"
    idx = text.find(start_tag)
    while idx != -1:
        depth = 1
        i = idx + len(start_tag)
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            result = text[idx + len(start_tag):i - 1].strip()
        idx = text.find(start_tag, idx + 1)
    return result


def extract_xml_answer(text: str) -> str:
    r"""
    Extract content from the last ``\boxed{...}`` in *text*.

    Handles nested braces correctly. Falls back to the raw text
    if no ``\boxed{`` is found.
    """
    if "\\boxed{" not in text:
        return text.strip()
    idx = text.rfind("\\boxed{")
    depth = 1
    i = idx + len("\\boxed{")
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[idx + len("\\boxed{"):i - 1].strip()
    return text.strip()


def extract_xml_tag_answer(text: str) -> Optional[str]:
    """
    Extract the content from ``<answer>...</answer>`` XML tags.

    Returns None if no answer tag is found.
    """
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def normalize_answer(ans: Optional[str]) -> Optional[str]:
    """Strip whitespace and remove common LaTeX wrappers for comparison."""
    if ans is None:
        return None
    ans = ans.replace(" ", "")
    ans = re.sub(r"\\textbf\{\((.)\)\}", r"\1", ans)
    return ans


def extract_answer_letter(text: str) -> str:
    r"""
    Extract a multiple-choice letter (a-e) from text containing
    'Answer: X' or 'Final Answer: X'.  Returns the LAST match.
    """
    matches = re.findall(r"(?:final\s*|)answer\s*[:\-]?\s*([a-e])", text.lower())
    if matches:
        return matches[-1]
    return "None"


def latex_equal(a: str, b: str) -> bool:
    """
    Try to determine symbolic equality of two LaTeX expressions
    using latex2sympy2.  Returns False on any failure.
    """
    try:
        from latex2sympy2 import latex2sympy
        return latex2sympy(a).equals(latex2sympy(b))
    except Exception:
        return False
