from setuptools import setup, find_packages

setup(
    name="math-rl-tuning",
    version="0.1.0",
    description="Fine-tune and RL-train LLMs for mathematical reasoning using SFT + GRPO",
    author="",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["../configs/*.yaml"],
    },
    data_files=[("configs", ["configs/default.yaml"])],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.56.1",
        "datasets>=3.0.0",
        "accelerate>=1.4.0",
        "peft>=0.7.1",
        "trl>=0.24.0",
        "bitsandbytes>=0.45.5",
        "safetensors>=0.4.3",
        "sentencepiece>=0.2.0",
        "latex2sympy2>=1.9.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.23.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "wandb>=0.15.0",
        "huggingface-hub>=0.21.0",
    ],
    extras_require={
        "unsloth": [
            "unsloth",
            "vllm",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
        ],
    },
)
