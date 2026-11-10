"""Merge results from multiple models and generate consensus datasets."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR
from .config import FALLBACK, LABELS
from .utils import get_output_parquet_path

logger = logging.getLogger(__name__)


def load_all_results(model_names: list[str]) -> dict[str, pd.DataFrame]:
    """Load classification results from all models."""
    results = {}
    for model_name in model_names:
        parquet_path = get_output_parquet_path(model_name)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Results not found for {model_name}: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        results[model_name] = df
        logger.info(f"Loaded {len(df)} results from {model_name}")
    return results


def merge_results(model_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge results from all models.
    Returns: (agreed_df, disagreed_df)
    """
    results = load_all_results(model_names)
    if len(model_names) < 2 or len(set(model_names)) != len(model_names):
        raise ValueError("Provide at least two distinct models")
    columns = [f"{model.replace('/', '_')}_type" for model in model_names]
    if len(set(columns)) != len(columns):
        raise ValueError("Model names collide when converted to output column names")
    base = results[model_names[0]].copy()
    if "pair_id" not in base or base["pair_id"].isna().any() or base["pair_id"].astype(str).duplicated().any():
        raise ValueError("Results must have unique, non-null pair IDs")
    base_ids = base["pair_id"].astype(str)
    if any(column in base.columns for column in columns + ["majority_type", "agreement_count", "confidence_score", "disagreement_pattern"]):
        raise ValueError("Input columns conflict with consensus output columns")
    for model_name, column in zip(model_names, columns):
        df = results[model_name]
        if "pair_id" not in df or "llm_pair_type" not in df or df["pair_id"].isna().any():
            raise ValueError(f"Missing pair IDs or classifications for {model_name}")
        ids = df["pair_id"].astype(str)
        if ids.duplicated().any() or set(ids) != set(base_ids):
            raise ValueError(f"Pair IDs do not match for {model_name}")
        predictions = pd.Series(df["llm_pair_type"].to_numpy(), index=ids)
        if not predictions.isin([*LABELS, FALLBACK]).all():
            raise ValueError(f"Invalid classifications for {model_name}")
        base[column] = predictions.reindex(base_ids).to_numpy()

    counts = [Counter(row) for row in base[columns].itertuples(index=False, name=None)]
    base["agreement_count"] = [max(c.values()) for c in counts]
    base["majority_type"] = [
        winners[0] if len(winners := [label for label, count in c.items() if count == max(c.values())]) == 1 else None
        for c in counts
    ]
    base["confidence_score"] = base["agreement_count"] / len(model_names)
    base["disagreement_pattern"] = [
        " vs ".join(f"{model}:{prediction}" for model, prediction in zip(model_names, row))
        for row in base[columns].itertuples(index=False, name=None)
    ]
    unanimous = base["agreement_count"] == len(model_names)
    agreed_df = base.loc[unanimous].copy()
    disagreed_df = base.loc[~unanimous].copy()

    logger.info(f"Generated {len(agreed_df)} agreed pairs and {len(disagreed_df)} disagreed pairs")
    return agreed_df, disagreed_df


def save_consensus_datasets(agreed_df: pd.DataFrame, disagreed_df: pd.DataFrame, model_names: list[str]) -> None:
    """Save consensus datasets to parquet files."""
    consensus_dir = RESULTS_DIR / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    agreed_path = consensus_dir / "agreed_pairs.parquet"
    disagreed_path = consensus_dir / "disagreed_pairs.parquet"

    agreed_df.to_parquet(agreed_path, index=False)
    disagreed_df.to_parquet(disagreed_path, index=False)
    logger.info("Saved %s agreed pairs and %s review pairs", len(agreed_df), len(disagreed_df))

    generate_statistics(agreed_df, disagreed_df, model_names, consensus_dir)


def generate_statistics(agreed_df: pd.DataFrame, disagreed_df: pd.DataFrame, model_names: list[str], consensus_dir: Path) -> None:
    """Generate statistical reports."""
    total_pairs = len(agreed_df) + len(disagreed_df)
    agreement_rate = len(agreed_df) / total_pairs if total_pairs > 0 else 0

    summary = {
        "total_pairs_analyzed": total_pairs,
        "pairs_with_full_agreement": len(agreed_df),
        "pairs_with_disagreement": len(disagreed_df),
        "overall_agreement_rate": agreement_rate,
        "models": model_names,
    }

    summary_path = consensus_dir / "agreement_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")

    if len(agreed_df) > 0:
        agreed_types = agreed_df["majority_type"].value_counts().to_dict()
        agreed_by_category = {
            t: {
                "count": int(agreed_df[agreed_df["majority_type"] == t].shape[0]),
                "percentage": float((agreed_df[agreed_df["majority_type"] == t].shape[0] / len(agreed_df) * 100)),
            }
            for t in agreed_types.keys()
        }
    else:
        agreed_by_category = {}

    if len(disagreed_df) > 0:
        disagreed_types = disagreed_df["majority_type"].value_counts().to_dict()
        disagreed_by_category = {
            t: {
                "count": int(disagreed_df[disagreed_df["majority_type"] == t].shape[0]),
                "percentage": float((disagreed_df[disagreed_df["majority_type"] == t].shape[0] / len(disagreed_df) * 100)),
            }
            for t in disagreed_types.keys()
        }
    else:
        disagreed_by_category = {}

    category_report = {
        "agreed_pairs_by_category": agreed_by_category,
        "disagreed_pairs_by_category": disagreed_by_category,
    }

    category_path = consensus_dir / "category_breakdown.json"
    with open(category_path, "w") as f:
        json.dump(category_report, f, indent=2)
    logger.info(f"Saved category breakdown to {category_path}")
