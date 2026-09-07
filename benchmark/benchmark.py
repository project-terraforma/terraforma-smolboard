"""Run a standardized zero-shot place-conflation benchmark."""

from __future__ import annotations

import argparse
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from smolboard_benchmark.adapters import create_adapter
from smolboard_benchmark.config import fingerprint, get_named, load_config, resolve_workspace_path
from smolboard_benchmark.datasets import ParquetConflationDataset
from smolboard_benchmark.evaluator import run_evaluation
from smolboard_benchmark.prompts import load_prompt
from smolboard_benchmark.results import FileResultWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="configured model name")
    parser.add_argument("--dataset", default="overture-3k", help="configured dataset name")
    parser.add_argument("--prompt", default="baseline", help="configured prompt name")
    parser.add_argument("--limit", type=int, metavar="N", help="deterministic balanced partial run")
    parser.add_argument("--batch-size", type=int, help="examples processed per adapter call")
    parser.add_argument("--run-id", help="new run ID, or an interrupted run ID to resume")
    parser.add_argument("--results-dir", type=Path, help="override configured results root")
    parser.add_argument("--no-download", action="store_true", help="fail if the configured model file is absent")
    parser.add_argument("--config", type=Path, help="benchmark YAML configuration path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 2:
        raise SystemExit("--limit must be at least 2")
    config = load_config(args.config)
    model_config = get_named(config, "models", args.model)
    dataset_config = get_named(config, "datasets", args.dataset)
    get_named(config, "prompts", args.prompt)
    adapter_config = get_named(config, "adapters", model_config["adapter"])
    dataset = ParquetConflationDataset(args.dataset, dataset_config)
    prompt = load_prompt(args.prompt, config)
    dataset_manifest = dataset.manifest()
    run_id = args.run_id or _default_run_id(args.model, args.dataset, args.prompt)
    _validate_run_id(run_id)
    result_root = args.results_dir.resolve() if args.results_dir else resolve_workspace_path(config["benchmark"]["result_root"])
    run_dir = result_root / run_id
    batch_size = args.batch_size or int(config["benchmark"]["evaluator_batch_size"])
    run_inputs = {
        "model": args.model,
        "model_config": model_config,
        "dataset": args.dataset,
        "dataset_manifest": dataset_manifest,
        "prompt": {"name": prompt.name, "version": prompt.version, "sha256": prompt.sha256},
        "generation": config["generation"],
        "adapter": adapter_config,
        "limit": args.limit,
        "evaluator_batch_size": batch_size,
        "seed": config["benchmark"]["seed"],
    }
    run_fingerprint = fingerprint(run_inputs)
    manifest = {
        "run_id": run_id,
        "started_at": _utc_now(),
        "run_fingerprint": run_fingerprint,
        "run_inputs": run_inputs,
        "output_contract_version": "1",
    }
    writer = FileResultWriter(run_dir, manifest)
    adapter = create_adapter(args.model, model_config, adapter_config, config["generation"], allow_download=not args.no_download)
    run_metadata = {
        "run_id": run_id,
        "model_name": args.model,
        "model_identifier": model_config["model_id"],
        "model_display_name": model_config["display_name"],
        "model_revision": model_config["revision"],
        "model_parameter_count": model_config.get("parameter_count"),
        "dataset": dataset_manifest,
        "prompt": {"name": prompt.name, "version": prompt.version, "sha256": prompt.sha256},
        "generation": config["generation"],
        "seed": config["benchmark"]["seed"],
        "run_fingerprint": run_fingerprint,
        "hardware": _hardware_metadata(),
        "software": _software_metadata(),
        "git": _git_metadata(),
    }
    print(f"Run {run_id}: model={args.model}, dataset={args.dataset}, prompt={args.prompt}, limit={args.limit or 'full'}")
    result = run_evaluation(
        dataset=dataset, prompt=prompt, model_name=args.model, adapter=adapter, writer=writer,
        dataset_batch_size=4096, evaluator_batch_size=batch_size, limit=args.limit,
        expected_examples=args.limit or dataset_manifest["total_examples"], run_metadata=run_metadata,
    )
    metrics = result["metrics"]
    print(f"Completed {metrics['total_examples']} examples: accuracy={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}. Wrote {run_dir}")


def _default_run_id(model: str, dataset: str, prompt: str) -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{model}_{dataset}_{prompt}"


def _validate_run_id(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise SystemExit("--run-id may contain only letters, numbers, dots, underscores, and hyphens")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hardware_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(), "python_architecture": platform.architecture()[0],
        "cpu_count_logical": psutil.cpu_count(), "system_ram_total_mib": psutil.virtual_memory().total / (1024**2),
        "gpus": _nvidia_gpus(),
    }


def _nvidia_gpus() -> list[dict[str, str]] | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return [
        dict(zip(("name", "memory_total", "driver_version"), (part.strip() for part in line.split(",")), strict=True))
        for line in output.splitlines() if line.strip()
    ]


def _software_metadata() -> dict[str, str]:
    import httpx
    import pandas
    import pyarrow
    import yaml
    return {"python": sys.version, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__, "httpx": httpx.__version__, "pyyaml": yaml.__version__}


def _git_metadata() -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        try:
            return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    status = git("status", "--porcelain")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


if __name__ == "__main__":
    main()
