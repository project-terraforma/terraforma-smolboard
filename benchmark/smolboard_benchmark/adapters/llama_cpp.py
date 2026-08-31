"""llama.cpp OpenAI-compatible server adapter reused from the Gemma experiment."""

from __future__ import annotations

import concurrent.futures
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx
from huggingface_hub import hf_hub_download

from ..config import resolve_workspace_path
from ..datasets import sha256_file
from ..monitoring import ResourceMonitor
from ..types import ModelResponse
from .base import ModelAdapter


class LlamaCppAdapter(ModelAdapter):
    def __init__(
        self,
        model_name: str,
        model_config: Mapping[str, Any],
        adapter_config: Mapping[str, Any],
        generation_config: Mapping[str, Any],
        *,
        allow_download: bool,
    ) -> None:
        self.model_name = model_name
        self.model_config = dict(model_config)
        self.adapter_config = dict(adapter_config)
        self.generation_config = dict(generation_config)
        self.allow_download = allow_download
        self.process: subprocess.Popen[str] | None = None
        self.log_handle: Any = None
        self.monitor: ResourceMonitor | None = None
        self.client: httpx.Client | None = None
        self.model_path: Path | None = None
        self.server_command: list[str] = []
        self.startup_seconds: float | None = None

    def start(self, log_path: str) -> None:
        self.model_path = self._ensure_model_file()
        executable = resolve_workspace_path(str(self.adapter_config["server_executable"]))
        if not executable.exists():
            raise FileNotFoundError(f"Missing llama.cpp server: {executable}")
        host, port = str(self.adapter_config["host"]), int(self.adapter_config["port"])
        self._ensure_port_available(host, port)
        self.server_command = [
            str(executable), "-m", str(self.model_path), "--host", host, "--port", str(port),
            "--ctx-size", str(self.adapter_config["context_size"]),
            "--parallel", str(self.adapter_config["parallel_slots"]),
            "--batch-size", str(self.adapter_config["batch_size"]),
            "--ubatch-size", str(self.adapter_config["micro_batch_size"]),
            "--gpu-layers", str(self.adapter_config["gpu_layers"]),
            "--flash-attn", str(self.adapter_config["flash_attention"]),
            "--reasoning", str(self.adapter_config["reasoning"]), "--reasoning-format", "auto",
            "--chat-template-kwargs", '{"enable_thinking":false}', "--metrics", "--no-webui",
        ]
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = path.open("a", encoding="utf-8")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self.server_command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            creationflags=creationflags,
        )
        self.monitor = ResourceMonitor(self.process)
        self.monitor.start()
        try:
            self.startup_seconds = self._wait_for_server(host, port)
        except Exception:
            self.stop()
            tail = ""
            try:
                tail = "\n".join(path.read_text(encoding="utf-8").splitlines()[-30:])
            except OSError:
                pass
            raise RuntimeError(f"Failed to start llama.cpp server. Log tail:\n{tail}")
        self.client = httpx.Client(base_url=f"http://{host}:{port}")

    def generate_batch(self, prompts: Sequence[str]) -> list[ModelResponse]:
        if not self.client:
            raise RuntimeError("Adapter has not been started")
        slots = int(self.adapter_config["parallel_slots"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=slots) as executor:
            return list(executor.map(self._generate_one, prompts))

    def _generate_one(self, prompt: str) -> ModelResponse:
        assert self.client is not None
        payload = self._request_payload(prompt)
        started = time.perf_counter()
        error: str | None = None
        raw_output: str | None = None
        usage: Mapping[str, Any] = {}
        for attempt in range(1, 4):
            try:
                response = self.client.post(
                    "/v1/chat/completions",
                    json=payload,
                    timeout=float(self.adapter_config["request_timeout_sec"]),
                )
                response.raise_for_status()
                body = response.json()
                raw_output = body["choices"][0]["message"].get("content")
                usage = body.get("usage") or {}
                error = None
                break
            except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < 3:
                    time.sleep(attempt)
        return ModelResponse(
            raw_output=raw_output,
            input_tokens=_optional_int(usage.get("prompt_tokens")),
            output_tokens=_optional_int(usage.get("completion_tokens")),
            latency_seconds=time.perf_counter() - started,
            request_error=error,
        )

    def _request_payload(self, prompt: str) -> dict[str, Any]:
        generation = self.generation_config
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": generation["temperature"],
            "top_p": generation["top_p"],
            "max_tokens": generation["max_tokens"],
            "seed": generation["seed"],
            "stop": generation["stop"],
            "stream": False,
            "cache_prompt": True,
        }

    def stop(self) -> dict[str, Any]:
        resource_metrics: dict[str, Any] = {}
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.monitor is not None:
            resource_metrics = self.monitor.stop()
            self.monitor = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        metadata = {
            "adapter": "llama_cpp",
            "backend": self.adapter_config["backend"],
            "backend_version": self.adapter_config["backend_version"],
            "server_command": self.server_command,
            "server_startup_seconds": self.startup_seconds,
            "quantization": self.model_config.get("quantization", "Google QAT Q4_0 GGUF"),
            "gpu_layers_requested": self.adapter_config["gpu_layers"],
            **resource_metrics,
        }
        if self.model_path and self.model_path.exists():
            metadata.update({
                "model_path": str(self.model_path),
                "model_file_size_bytes": self.model_path.stat().st_size,
                "model_file_sha256": sha256_file(self.model_path),
            })
        return metadata

    def _ensure_model_file(self) -> Path:
        path = resolve_workspace_path(str(self.model_config["path"]))
        if not path.exists() and self.allow_download:
            path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = hf_hub_download(
                repo_id=self.model_config["quantized_repo"],
                revision=self.model_config["revision"],
                filename=self.model_config["filename"],
                local_dir=path.parent,
            )
            path = Path(downloaded).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing model file {path}. Re-run without --no-download.")
        expected = int(self.model_config["expected_size_bytes"])
        if path.stat().st_size != expected:
            raise ValueError(f"Model size mismatch for {path}: expected {expected}, got {path.stat().st_size}")
        return path

    @staticmethod
    def _ensure_port_available(host: str, port: int) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex((host, port)) == 0:
                raise RuntimeError(f"Refusing to reuse occupied server address {host}:{port}")

    def _wait_for_server(self, host: str, port: int) -> float:
        started = time.perf_counter()
        deadline = started + float(self.adapter_config["startup_timeout_sec"])
        last_error = "server did not answer"
        while time.perf_counter() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"llama.cpp server exited during startup with code {self.process.returncode}")
            try:
                response = httpx.get(f"http://{host}:{port}/health", timeout=3)
                if response.status_code == 200:
                    return time.perf_counter() - started
                last_error = f"health returned HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(1)
        raise TimeoutError(f"llama.cpp server startup timed out: {last_error}")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
