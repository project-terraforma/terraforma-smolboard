"""Registered model backends."""

from __future__ import annotations

from typing import Any, Mapping

from .base import ModelAdapter
from .llama_cpp import LlamaCppAdapter
from .qwen3 import Qwen3LlamaCppAdapter


def create_adapter(
    model_name: str,
    model_config: Mapping[str, Any],
    adapter_config: Mapping[str, Any],
    generation_config: Mapping[str, Any],
    *,
    allow_download: bool,
) -> ModelAdapter:
    adapter_kind = model_config.get("adapter")
    if adapter_kind == "llama_cpp":
        return LlamaCppAdapter(
            model_name,
            model_config,
            adapter_config,
            generation_config,
            allow_download=allow_download,
        )
    if adapter_kind == "qwen3_llama_cpp":
        return Qwen3LlamaCppAdapter(
            model_name,
            model_config,
            adapter_config,
            generation_config,
            allow_download=allow_download,
        )
    raise ValueError(f"Unsupported model adapter '{adapter_kind}' for model '{model_name}'")


__all__ = ["ModelAdapter", "create_adapter"]
