"""Command-line entrypoint for the benchmark agent."""

from __future__ import annotations

import argparse
import json

from agents.runner import run_model_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark job for a single model.")
    parser.add_argument("model", help="Configured benchmark model name")
    parser.add_argument("--dataset", default="medium_pairs_geo", help="Configured dataset name")
    parser.add_argument("--prompt", default="baseline", help="Configured prompt name")
    parser.add_argument("--limit", type=int, help="Deterministic partial run limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_model_benchmark(
        args.model,
        dataset=args.dataset,
        prompt=args.prompt,
        limit=args.limit,
    )
    print(json.dumps({
        "model_name": result["model_name"],
        "success": result["success"],
        "returncode": result["returncode"],
        "run_dir": result["run_dir"],
        "accuracy": result["metrics"].get("accuracy"),
        "f1": result["metrics"].get("f1"),
    }, indent=2))


if __name__ == "__main__":
    main()
