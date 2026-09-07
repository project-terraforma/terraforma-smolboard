"""Structured, resumable file persistence for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Protocol

from .types import ExampleResult


class ResultWriter(Protocol):
    def append(self, result: ExampleResult) -> None: ...
    def iter_checkpoint(self) -> Iterator[ExampleResult]: ...
    def complete(self, metrics: Mapping[str, Any], run_result: Mapping[str, Any]) -> None: ...
    def interrupt(self, reason: str) -> None: ...


class FileResultWriter:
    """Writes ordered JSONL checkpoints and normalized aggregate JSON outputs."""

    def __init__(self, run_dir: Path, manifest: Mapping[str, Any]) -> None:
        self.run_dir = run_dir
        self.manifest_path = run_dir / "run_manifest.json"
        self.checkpoint_path = run_dir / "predictions.checkpoint.jsonl"
        self.predictions_path = run_dir / "predictions.jsonl"
        self._manifest = dict(manifest)
        self._handle: Any = None
        self._open_or_validate()

    @property
    def log_path(self) -> Path:
        return self.run_dir / "model.log"

    def _open_or_validate(self) -> None:
        if self.manifest_path.exists():
            existing = _read_json(self.manifest_path)
            if existing.get("run_fingerprint") != self._manifest.get("run_fingerprint"):
                raise ValueError("Existing run directory has a different configuration fingerprint")
            if existing.get("status") == "completed":
                raise FileExistsError(f"Run is already complete: {self.run_dir}")
            self._manifest = existing
        else:
            self.run_dir.mkdir(parents=True, exist_ok=False)
            self._manifest["status"] = "running"
            _write_json(self.manifest_path, self._manifest)
        if self.predictions_path.exists():
            raise RuntimeError("Completed predictions file exists for an incomplete run; refusing to overwrite it")
        self._handle = self.checkpoint_path.open("a", encoding="utf-8", buffering=1)

    def append(self, result: ExampleResult) -> None:
        if self._handle is None:
            raise RuntimeError("Result writer is closed")
        self._handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
        self._handle.flush()

    def iter_checkpoint(self) -> Iterator[ExampleResult]:
        if not self.checkpoint_path.exists():
            return
        with self.checkpoint_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    yield ExampleResult.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Malformed checkpoint record at {self.checkpoint_path}:{line_number}") from exc

    def complete(self, metrics: Mapping[str, Any], run_result: Mapping[str, Any]) -> None:
        self.close()
        if not self.checkpoint_path.exists():
            raise RuntimeError("No checkpoint exists to finalize")
        self.checkpoint_path.replace(self.predictions_path)
        _write_json(self.run_dir / "metrics.json", dict(metrics))
        _write_json(self.run_dir / "run_result.json", dict(run_result))
        self._manifest["status"] = "completed"
        self._manifest["completed_at"] = run_result.get("completed_at")
        _write_json(self.manifest_path, self._manifest)

    def interrupt(self, reason: str) -> None:
        self.close()
        self._manifest["status"] = "interrupted"
        self._manifest["interruption_reason"] = reason
        _write_json(self.manifest_path, self._manifest)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
