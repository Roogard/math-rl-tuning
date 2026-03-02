"""
Inference module.

Provides:
  - Streaming text generation (threaded, for interactive use)
  - Simple one-shot generation
  - Batch generation for test sets
"""

import threading
import torch
from typing import Optional

from math_rl_tuning.config import Config
from math_rl_tuning.utils import extract_boxed, extract_xml_tag_answer


# ---------------------------------------------------------------------------
# Streaming generation (for interactive / notebook use)
# ---------------------------------------------------------------------------

def generate_stream(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    print_output: bool = True,
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
) -> str:
    """
    Generate a response with token-by-token streaming.

    Uses the tokenizer's chat template so the prompt format matches
    whatever model is loaded (Qwen, Mistral, Llama, etc.).

    Args:
        question: The math question to ask.
        model: The loaded model.
        tokenizer: The tokenizer.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling threshold.
        do_sample: Whether to sample (True) or use greedy decoding (False).
        print_output: If True, print tokens as they are generated.
        system_prompt: System instruction prepended to the conversation.

    Returns:
        The full generated text (response only, without prompt).
    """
    from transformers import TextIteratorStreamer

    model.eval()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    streamer = TextIteratorStreamer(
        tokenizer, skip_special_tokens=True, skip_prompt=True
    )

    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    # Run generation in background thread with no gradient tracking
    def _generate_no_grad():
        with torch.no_grad():
            model.generate(**generation_kwargs)

    thread = threading.Thread(target=_generate_no_grad)
    thread.start()

    output_text = ""
    for token in streamer:
        if print_output:
            print(token, end="", flush=True)
        output_text += token

    thread.join()

    if print_output:
        print()  # newline after streaming

    return output_text.strip()


# ---------------------------------------------------------------------------
# Simple one-shot generation (no streaming)
# ---------------------------------------------------------------------------

def generate(
    question: str,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
) -> str:
    """
    Generate a response without streaming.

    Uses the tokenizer's chat template so the prompt format matches
    whatever model is loaded (Qwen, Mistral, Llama, etc.).

    Returns the model's response (prompt stripped).
    """
    model.eval()
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
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Generation from config
# ---------------------------------------------------------------------------

def generate_from_config(
    question: str,
    model,
    tokenizer,
    cfg: Config,
    stream: bool = False,
) -> str:
    """
    Generate using parameters from the config's inference section.

    Args:
        question: The math question.
        model: The loaded model.
        tokenizer: The tokenizer.
        cfg: Project configuration.
        stream: If True, use streaming generation with live printing.

    Returns:
        The generated response text.
    """
    ic = cfg.inference

    if stream:
        return generate_stream(
            question=question,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=ic.max_new_tokens,
            temperature=ic.temperature,
            top_p=ic.top_p,
            do_sample=ic.do_sample,
        )
    else:
        return generate(
            question=question,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=ic.max_new_tokens,
            temperature=ic.temperature,
            top_p=ic.top_p,
            do_sample=ic.do_sample,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_final_answer(response: str) -> Optional[str]:
    """
    Extract the final answer from a model response.

    Tries <answer> XML tags first (GRPO format), then \\boxed{},
    then falls back to the last line.
    """
    xml_ans = extract_xml_tag_answer(response)
    if xml_ans:
        return xml_ans

    boxed = extract_boxed(response)
    if boxed:
        return boxed

    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else None
