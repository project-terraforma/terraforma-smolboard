"""Core, backend-neutral benchmark data structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Mapping


class Label(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"


class Prediction(StrEnum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"
    INVALID = "INVALID"


@dataclass(frozen=True)
class PlaceRecord:
    fields: Mapping[str, Any]


@dataclass(frozen=True)
class ConflationExample:
    example_id: str
    record_a: PlaceRecord
    record_b: PlaceRecord
    label: Label
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    template: str
    exposed_fields: tuple[str, ...]
    sha256: str


@dataclass(frozen=True)
class ModelResponse:
    raw_output: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    request_error: str | None = None
    backend_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    ground_truth: Label
    prediction: Prediction
    raw_model_output: str | None
    valid_prediction: bool
    latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    request_error: str | None
    model: str
    prompt_name: str
    prompt_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ground_truth"] = self.ground_truth.value
        payload["prediction"] = self.prediction.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExampleResult":
        return cls(
            example_id=str(payload["example_id"]),
            ground_truth=Label(payload["ground_truth"]),
            prediction=Prediction(payload["prediction"]),
            raw_model_output=payload.get("raw_model_output"),
            valid_prediction=bool(payload["valid_prediction"]),
            latency_seconds=float(payload["latency_seconds"]),
            input_tokens=_optional_int(payload.get("input_tokens")),
            output_tokens=_optional_int(payload.get("output_tokens")),
            request_error=payload.get("request_error") or None,
            model=str(payload["model"]),
            prompt_name=str(payload["prompt_name"]),
            prompt_version=str(payload["prompt_version"]),
            metadata=dict(payload.get("metadata") or {}),
        )


def _optional_int(value: Any) -> int | None:
    return None if value is None or value == "" else int(value)
