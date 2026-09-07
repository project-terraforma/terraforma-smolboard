"""Backend contract for normalized model inference."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from ..types import ModelResponse


class ModelAdapter(ABC):
    @abstractmethod
    def start(self, log_path: str) -> None:
        """Load or start the backend and make it ready for requests."""

    @abstractmethod
    def generate_batch(self, prompts: Sequence[str]) -> list[ModelResponse]:
        """Return responses in the same order as prompts."""

    @abstractmethod
    def stop(self) -> dict[str, Any]:
        """Release resources and return backend/runtime metadata."""
