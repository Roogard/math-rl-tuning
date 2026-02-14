#!/usr/bin/env python3
"""
Run SFT training from the command line.

Usage:
    python scripts/run_sft.py
    python scripts/run_sft.py --config configs/custom.yaml
    python scripts/run_sft.py --epochs 2 --batch-size 8
"""

import argparse
import sys
import os

# Ensure the project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_rl_tuning.config import load_config
from math_rl_tuning.utils import setup_hf_token, setup_wandb, get_device
from math_rl_tuning.sft_trainer import run_sft_training


def main():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom YAML config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override per-device train batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable Weights & Biases logging")
    parser.add_argument("--save-to-drive", action="store_true",
                        help="Copy saved model to Google Drive (Colab only)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face API token")
    parser.add_argument("--wandb-key", type=str, default=None,
                        help="Weights & Biases API key")
    args = parser.parse_args()

    # Load config
    cfg = load_config(config_path=args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        cfg.sft_training.num_train_epochs = args.epochs
    if args.batch_size is not None:
        cfg.sft_training.per_device_train_batch_size = args.batch_size
    if args.lr is not None:
        cfg.sft_training.learning_rate = args.lr
    if args.output_dir is not None:
        cfg.paths.sft_output_dir = args.output_dir
    if args.no_wandb:
        cfg.sft_training.report_to = "none"

    # Setup
    print(f"Device: {get_device()}")
    setup_hf_token(args.hf_token)

    if cfg.sft_training.report_to == "wandb":
        setup_wandb(cfg.sft_training.wandb_project, key=args.wandb_key)

    # Train
    trainer, model, tokenizer = run_sft_training(
        cfg, save_to_drive=args.save_to_drive
    )

    print("\nSFT training complete.")


if __name__ == "__main__":
    main()
