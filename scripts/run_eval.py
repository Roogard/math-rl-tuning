#!/usr/bin/env python3
"""
Run evaluation and model comparison from the command line.

Usage:
    python scripts/run_eval.py --sft-adapter ./outputs/sft
    python scripts/run_eval.py --sft-adapter ./outputs/sft --rl-adapter ./outputs/grpo
    python scripts/run_eval.py --sft-adapter ./outputs/sft --num-samples 100
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from math_rl_tuning.config import load_config
from math_rl_tuning.utils import setup_hf_token, get_device
from math_rl_tuning.data import prepare_test_data
from math_rl_tuning.evaluation import evaluate_adapter, compare_models, save_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to custom YAML config file")
    parser.add_argument("--sft-adapter", type=str, required=True,
                        help="Path to the saved SFT LoRA adapter")
    parser.add_argument("--rl-adapter", type=str, default=None,
                        help="Path to the saved RL LoRA adapter (for comparison)")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of test examples to evaluate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override evaluation output directory")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="Hugging Face API token")
    parser.add_argument("--all-sources", action="store_true",
                        help="Evaluate on all test sources (ignore eval_keep_sources filter)")
    args = parser.parse_args()

    # Load config
    cfg = load_config(config_path=args.config)

    if args.num_samples is not None:
        cfg.evaluation.num_samples = args.num_samples
    if args.output_dir is not None:
        cfg.paths.eval_output_dir = args.output_dir
    if args.all_sources:
        cfg.evaluation.eval_keep_sources = []

    # Setup
    print(f"Device: {get_device()}")
    setup_hf_token(args.hf_token)

    # Load test data
    test_ds = prepare_test_data(cfg)

    if args.rl_adapter:
        # Head-to-head comparison
        results = compare_models(
            sft_adapter_path=args.sft_adapter,
            rl_adapter_path=args.rl_adapter,
            test_dataset=test_ds,
            cfg=cfg,
            num_samples=cfg.evaluation.num_samples,
        )
    else:
        # Single model evaluation
        sft_df, sft_acc = evaluate_adapter(
            adapter_path=args.sft_adapter,
            test_dataset=test_ds,
            cfg=cfg,
            num_samples=cfg.evaluation.num_samples,
            model_name="SFT",
        )
        results = {"sft_results": sft_df, "sft_accuracy": sft_acc}

    # Save
    save_results(results, cfg.paths.eval_output_dir)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
