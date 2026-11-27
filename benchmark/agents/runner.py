"""Single-model benchmark runner used by the LangGraph agent."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _benchmark_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _latest_run_dir(model_name: str) -> Path | None:
    result_root = _benchmark_dir() / "results" / "benchmark_runs"
    if not result_root.exists():
        return None
    candidates = []
    for run_dir in sorted(result_root.iterdir()):
        if run_dir.is_dir() and model_name in run_dir.name:
            candidates.append(run_dir)
    if not candidates:
        return None
    return candidates[-1]


def run_model_benchmark(
    model_name: str,
    *,
    dataset: str = "medium_pairs_geo",
    prompt: str = "baseline",
    limit: int | None = None,
    results_dir: str | None = None,
) -> dict[str, Any]:
    """Invoke the benchmark CLI for a single model and capture the output."""
    benchmark_dir = _benchmark_dir()
    cmd = [
        sys.executable,
        "benchmark.py",
        model_name,
        "--dataset",
        dataset,
        "--prompt",
        prompt,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if results_dir is not None:
        cmd.extend(["--results-dir", str(results_dir)])

    completed = subprocess.run(
        cmd,
        cwd=str(benchmark_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    run_dir = _latest_run_dir(model_name)
    metrics_path = run_dir / "metrics.json" if run_dir else None
    metrics: dict[str, Any] = {}
    if metrics_path and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    return {
        "model_name": model_name,
        "dataset": dataset,
        "prompt": prompt,
        "limit": limit,
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "run_dir": str(run_dir) if run_dir else None,
        "metrics": metrics,
    }
