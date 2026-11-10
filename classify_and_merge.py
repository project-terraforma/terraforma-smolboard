#!/usr/bin/env python3
"""CLI entry point for multi-model classification and merger."""

import argparse
import logging
import sys
import torch

from multi_model_classification.config import ClassificationConfig, DEFAULT_MODELS
from multi_model_classification.orchestrator import run_full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run multi-model LLM classification on medium pairs and generate consensus datasets"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"HuggingFace model names to use (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs to classify (default: all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Reserved for future batching; inference currently handles one pair at a time",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Max concurrent model runs (default: 1, sequential)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference (default: auto)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoints, start fresh",
    )
    parser.add_argument(
        "--no-int8",
        action="store_true",
        help="Disable CUDA int8 quantization (requires more GPU memory)",
    )

    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if len(args.models) < 2 or len(set(args.models)) != len(args.models):
        parser.error("--models requires at least two distinct models")

    config = ClassificationConfig(
        model_names=args.models,
        limit=args.limit,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        device=("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device,
        resume=not args.no_resume,
        use_int8=not args.no_int8,
    )

    try:
        agreed_path, disagreed_path = run_full_pipeline(config)
        logger.info(f"\n✓ Classification complete!")
        logger.info(f"✓ Agreed pairs (all models unanimous): {agreed_path}")
        logger.info(f"✓ Disagreed pairs (team review): {disagreed_path}")
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
