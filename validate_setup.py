#!/usr/bin/env python3
"""Validation script to test the multi-model classification system."""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_setup():
    """Validate that all required files and modules exist."""
    errors = []

    # Check multi_model_classification module
    mmc_path = Path("multi_model_classification")
    if not mmc_path.exists():
        errors.append("multi_model_classification directory not found")

    required_files = [
        "multi_model_classification/__init__.py",
        "multi_model_classification/config.py",
        "multi_model_classification/utils.py",
        "multi_model_classification/classifier.py",
        "multi_model_classification/merger.py",
        "multi_model_classification/orchestrator.py",
        "classify_and_merge.py",
    ]

    for file_path in required_files:
        if not Path(file_path).exists():
            errors.append(f"Missing: {file_path}")

    # Check input data
    input_parquet = Path("datasets/geo/medium_pairs.parquet")
    if not input_parquet.exists():
        errors.append(f"Input data not found: {input_parquet}")

    if errors:
        logger.error("Validation failed:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        return False
    else:
        logger.info("✓ All files present")
        logger.info("✓ Input data found")
        return True


def validate_imports():
    """Validate that all required packages can be imported."""
    required_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("pandas", "pd"),
        ("pyarrow", "pyarrow"),
    ]

    errors = []
    for import_name, display_name in required_packages:
        try:
            __import__(import_name)
            logger.info(f"✓ {display_name} installed")
        except ImportError:
            errors.append(f"Missing package: {display_name}")

    if errors:
        logger.error("Import validation failed:")
        for error in errors:
            logger.error(f"  ✗ {error}")
        logger.error("\nInstall missing packages with:")
        logger.error("  pip install -r multi_model_classification/requirements.txt")
        return False

    return True


def validate_module_imports():
    """Validate that multi_model_classification modules can be imported."""
    try:
        from multi_model_classification.config import ClassificationConfig, DEFAULT_MODELS
        from multi_model_classification.orchestrator import run_full_pipeline
        logger.info("✓ multi_model_classification modules importable")
        logger.info(f"✓ Default models configured: {len(DEFAULT_MODELS)} models")
        return True
    except ImportError as e:
        logger.error(f"✗ Failed to import multi_model_classification: {e}")
        return False


def main():
    logger.info("=" * 80)
    logger.info("Multi-Model Classification System - Validation")
    logger.info("=" * 80)

    checks = [
        ("File structure", validate_setup),
        ("Required packages", validate_imports),
        ("Module imports", validate_module_imports),
    ]

    results = []
    for check_name, check_func in checks:
        logger.info(f"\nRunning: {check_name}...")
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            logger.error(f"✗ {check_name} check failed: {e}")
            results.append(False)

    logger.info("\n" + "=" * 80)
    if all(results):
        logger.info("✓ All validation checks passed!")
        logger.info("=" * 80)
        logger.info("\nYou can now run:")
        logger.info("  python classify_and_merge.py --limit 100")
        return 0
    else:
        logger.error("✗ Some validation checks failed")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
