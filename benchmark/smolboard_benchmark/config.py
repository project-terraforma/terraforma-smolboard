"""Configuration loading and stable run fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "benchmark_config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = (path or DEFAULT_CONFIG_PATH).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Benchmark configuration must be a mapping: {config_path}")
    for key in ("benchmark", "datasets", "prompts", "models", "adapters", "generation"):
        if key not in config or not isinstance(config[key], dict):
            raise ValueError(f"Missing mapping '{key}' in {config_path}")
    config["_config_path"] = str(config_path)
    return config


def get_named(config: Mapping[str, Any], collection: str, name: str) -> dict[str, Any]:
    values = config.get(collection, {})
    if name not in values:
        choices = ", ".join(sorted(values)) or "none"
        raise ValueError(f"Unknown {collection[:-1]} '{name}'. Available: {choices}")
    value = values[name]
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {collection[:-1]} configuration for '{name}'")
    return dict(value)


def resolve_workspace_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()


def fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
