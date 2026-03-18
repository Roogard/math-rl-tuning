from setuptools import setup, find_packages

setup(
    name="math-rl-tuning",
    version="0.1.0",
    description="Fine-tune and RL-train LLMs for mathematical reasoning using SFT + GRPO",
    author="",
    packages=find_packages(),
    include_package_data=True,
    # Ship the default YAML config so it's accessible after `pip install`
    package_data={
        "": ["../configs/*.yaml"],
    },
    data_files=[("configs", ["configs/default.yaml"])],
    python_requires=">=3.10",  # Requires 3.10+ for match-case and | union type hints
    install_requires=[
        # Core ML stack
        "torch>=2.3.0",
        "transformers>=4.56.1",   # GRPOConfig and SFTConfig APIs changed frequently; pin recent
        "datasets>=3.0.0",
        "accelerate>=1.4.0",
        "peft>=0.7.1",            # LoRA / QLoRA adapter management
        "trl>=0.26.2",            # 0.26.2+ defers llm_blender import (PR #4598)
        "bitsandbytes>=0.45.5",   # 4-bit NF4 quantization (QLoRA)
        "safetensors>=0.4.3",     # Fast, safe model weight serialization
        "sentencepiece>=0.2.0",   # Tokenizer dependency for Qwen2.5
        # Math answer verification
        "latex2sympy2>=1.9.0",    # LaTeX → SymPy conversion (used by math-verify)
        # Data / utilities
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.23.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        # Experiment tracking and model hub
        "wandb>=0.15.0",
        "huggingface-hub>=0.21.0",
    ],
    extras_require={
        # Optional: Unsloth + vLLM for faster GRPO training with speculative decoding
        "unsloth": [
            "unsloth",
            "vllm",
        ],
        # Optional: visualization for reward curves and evaluation results
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
)
