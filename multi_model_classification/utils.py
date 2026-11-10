"""Utility functions for classification."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

from .config import FALLBACK, LABELS, SHOWN_FIELDS


def render_record(row: pd.Series, side: str) -> str:
    """Render a single record as JSON for the prompt."""
    out = {}
    for f in SHOWN_FIELDS:
        v = row.get(f"{side}_{f}")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if hasattr(v, "tolist"):
            v = v.tolist()
        out[f] = v
    return json.dumps(out, ensure_ascii=False, default=str)[:1500]


def parse_label(text: str | None) -> str:
    """Extract label from model output."""
    t = (text or "").strip().lower().strip("`\"' \n\r\t")
    if t not in {*LABELS, FALLBACK}:
        raise ValueError(f"Model returned an invalid label: {text!r}")
    return t


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_label_descriptions() -> str:
    """Build label descriptions for prompt."""
    lines = []
    for label, desc in LABELS.items():
        lines.append(f"- {label}: {desc}")
    return "\n".join(lines)


def ensure_results_dir(model_name: str) -> Path:
    """Ensure model results directory exists."""
    from .config import RESULTS_DIR
    model_dir = RESULTS_DIR / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_checkpoint_path(model_name: str) -> Path:
    """Get checkpoint file path for a model."""
    model_dir = ensure_results_dir(model_name)
    return model_dir / "pairtype_checkpoint_v2.jsonl"


def get_output_parquet_path(model_name: str) -> Path:
    """Get output parquet path for a model."""
    model_dir = ensure_results_dir(model_name)
    return model_dir / "medium_pairs_typed.parquet"


def get_stats_path(model_name: str) -> Path:
    """Get stats JSON path for a model."""
    model_dir = ensure_results_dir(model_name)
    return model_dir / "pairtype_stats.json"


def load_checkpoint(model_name: str) -> dict[str, dict[str, str]]:
    """Load existing checkpoint or return empty dict."""
    checkpoint_path = get_checkpoint_path(model_name)
    if not checkpoint_path.exists():
        return {}

    checkpoints = {}
    with open(checkpoint_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                label = parse_label(record["llm_pair_type"])
                pair_id = record["pair_id"]
                digest = record["prompt_hash"]
                if not isinstance(pair_id, str) or not isinstance(digest, str):
                    raise ValueError("Invalid pair ID or prompt hash")
                if pair_id in checkpoints:
                    raise ValueError(f"Duplicate checkpoint pair ID: {pair_id}")
                checkpoints[pair_id] = {"llm_pair_type": label, "prompt_hash": digest}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid checkpoint {checkpoint_path}, line {line_number}: {exc}") from exc
    return checkpoints


def save_checkpoint(model_name: str, pair_id: str, classification: str, digest: str, raw_output: str | None = None) -> None:
    """Append a single classification to checkpoint."""
    checkpoint_path = get_checkpoint_path(model_name)
    record = {
        "pair_id": pair_id,
        "llm_pair_type": classification,
        "prompt_hash": digest,
        "raw_output": raw_output,
    }
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_statistics(parquet_path: Path, column: str = "llm_pair_type") -> dict:
    """Compute statistics for typed pairs."""
    df = pd.read_parquet(parquet_path)

    counts = df[column].value_counts().to_dict()
    total = len(df)
    percentages = {k: (v / total * 100) for k, v in counts.items()}

    return {
        "total_pairs": total,
        "counts": counts,
        "percentages": percentages,
    }
