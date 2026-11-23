"""Export benchmark runs into repo-readable JSON for the website."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping


def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("metrics") or {}
    model_name = run.get("model_display_name") or run.get("model_name") or run.get("model") or "unknown"
    run_inputs = run.get("run_inputs") or {}
    dataset = run.get("dataset") or run_inputs.get("dataset") or "unknown"
    prompt = run.get("prompt") or run_inputs.get("prompt") or "baseline"
    if isinstance(dataset, Mapping):
        dataset = dataset.get("dataset_name") or "unknown"
    if isinstance(prompt, Mapping):
        prompt = prompt.get("name") or "baseline"
    return {
        "model": model_name,
        "model_name": model_name,
        "dataset": dataset,
        "prompt": prompt,
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


def publish_run(
    run_path: str | Path,
    output_path: str | Path,
    *,
    replace_models: Iterable[str] = (),
) -> Path:
    """Replace selected model rows with one completed benchmark result."""
    run_file = Path(run_path)
    run = json.loads(run_file.read_text(encoding="utf-8"))

    output = Path(output_path)
    existing = json.loads(output.read_text(encoding="utf-8")) if output.exists() else []
    names = {_normalize_name(name) for name in replace_models}
    new_row = normalize_run(run)
    names.add(_normalize_name(new_row["model"]))
    kept = [row for row in existing if _normalize_name(row.get("model", "")) not in names]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps([*kept, new_row], indent=2) + "\n", encoding="utf-8")
    return output


def latest_run_for_model(model_name: str, results_root: str | Path) -> Path:
    root = Path(results_root)
    candidates = [
        run_dir / "run_result.json"
        for run_dir in root.iterdir()
        if run_dir.is_dir()
        and model_name.lower() in run_dir.name.lower()
        and (run_dir / "run_result.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run found for model '{model_name}' in {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _normalize_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, nargs="?", help="completed run_result.json file")
    parser.add_argument("--model", help="publish the latest completed run for this model")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("benchmark/results/benchmark_runs"),
        help="benchmark runs directory used with --model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/data/leaderboard.json"),
        help="leaderboard JSON output path",
    )
    parser.add_argument(
        "--replace-model",
        action="append",
        default=[],
        help="existing leaderboard model name to remove before publishing",
    )
    args = parser.parse_args()
    if args.run is None and args.model is None:
        parser.error("provide a run_result.json path or --model")
    run = args.run or latest_run_for_model(args.model, args.results_root)
    publish_run(run, args.output, replace_models=args.replace_model)
    print(f"Published {run} to {args.output}")


if __name__ == "__main__":
    main()
