"""Export benchmark runs into repo-readable JSON for the website."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics") or {}
    model_name = run.get("model_name") or run.get("model") or "unknown"
    run_inputs = run.get("run_inputs") or {}
    return {
        "model": model_name,
        "model_name": model_name,
        "dataset": run_inputs.get("dataset") or "unknown",
        "prompt": run_inputs.get("prompt") or run.get("prompt") or "baseline",
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "average_latency": metrics.get("average_latency_seconds"),
        "tokens_per_second": metrics.get("tokens_per_second"),
        "benchmark_date": run.get("completed_at") or run.get("started_at") or "unknown",
        "parameter_count": run.get("parameter_count") or run.get("model_parameter_count"),
        "parameter_bucket": _bucket_for_parameter_count(run.get("parameter_count") or run.get("model_parameter_count")),
    }


def _bucket_for_parameter_count(value: Any) -> str:
    if value is None:
        return "all"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "all"
    if numeric < 5_000_000_000:
        return "<5B"
    if numeric < 10_000_000_000:
        return "<10B"
    if numeric < 20_000_000_000:
        return "<20B"
    return "all"


def export_leaderboard(rows: Iterable[dict[str, Any]], output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [normalize_run(row) for row in rows]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
