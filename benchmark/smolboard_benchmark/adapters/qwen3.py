"""Qwen3 llama.cpp adapter with deterministic non-thinking chat behavior."""

from __future__ import annotations

from typing import Any

from .llama_cpp import LlamaCppAdapter


class Qwen3LlamaCppAdapter(LlamaCppAdapter):
    """Use Qwen3's template toggle while retaining the common llama.cpp lifecycle."""

    def _request_payload(self, prompt: str) -> dict[str, Any]:
        payload = super()._request_payload(prompt)
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return payload
