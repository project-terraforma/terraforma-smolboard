"""Orchestrate parallel classification across multiple models."""

from __future__ import annotations

import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .classifier import HuggingFaceClassifier
from .config import DATASETS_DIR, ClassificationConfig
from .merger import merge_results, save_consensus_datasets

logger = logging.getLogger(__name__)


def run_classifications(config: ClassificationConfig) -> list[Path]:
    """Run classifications for all models in parallel."""
    input_parquet = DATASETS_DIR / "medium_pairs.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")
    if len(config.model_names) < 2 or len(set(config.model_names)) != len(config.model_names):
        raise ValueError("Provide at least two distinct models")
    if len({name.replace("/", "_") for name in config.model_names}) != len(config.model_names):
        raise ValueError("Model names collide when converted to result directories")

    logger.info(f"Starting classification with {len(config.model_names)} models")
    logger.info(f"Input: {input_parquet}")

    output_paths = []

    def run_model(model_name: str) -> Path:
        logger.info(f"Starting classification for {model_name}")
        classifier = HuggingFaceClassifier(model_name, config)
        try:
            output_path = classifier.run(input_parquet)
        finally:
            del classifier
            gc.collect()
        logger.info(f"Completed classification for {model_name}")
        return output_path

    if config.max_workers == 1:
        for model_name in config.model_names:
            output_path = run_model(model_name)
            output_paths.append(output_path)
    else:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(run_model, model_name): model_name for model_name in config.model_names}
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    output_path = future.result()
                    output_paths.append(output_path)
                except Exception as e:
                    logger.error(f"Error running {model_name}: {e}")
                    raise

    logger.info(f"All classifications complete: {len(output_paths)} models")
    return output_paths


def run_merger(config: ClassificationConfig) -> tuple[Path, Path]:
    """Merge results from all models and generate consensus datasets."""
    logger.info("Starting merger phase")

    agreed_df, disagreed_df = merge_results(config.model_names)
    save_consensus_datasets(agreed_df, disagreed_df, config.model_names)

    consensus_dir = DATASETS_DIR / "multi_model_results" / "consensus"
    agreed_path = consensus_dir / "agreed_pairs.parquet"
    disagreed_path = consensus_dir / "disagreed_pairs.parquet"

    logger.info(f"Merger complete: {len(agreed_df)} agreed, {len(disagreed_df)} disagreed")
    return agreed_path, disagreed_path


def run_full_pipeline(config: ClassificationConfig) -> tuple[Path, Path]:
    """Run the full classification and merging pipeline."""
    logger.info("=" * 80)
    logger.info("Starting Multi-Model Classification Pipeline")
    logger.info("=" * 80)
    logger.info(f"Models: {config.model_names}")
    logger.info(f"Max workers: {config.max_workers}")
    logger.info("=" * 80)

    classification_paths = run_classifications(config)
    agreed_path, disagreed_path = run_merger(config)

    logger.info("=" * 80)
    logger.info("Pipeline Complete!")
    logger.info(f"Agreed pairs: {agreed_path}")
    logger.info(f"Disagreed pairs: {disagreed_path}")
    logger.info("=" * 80)

    return agreed_path, disagreed_path
