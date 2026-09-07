"""One metric implementation for every standardized benchmark adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .types import ExampleResult, Label, Prediction


@dataclass
class MetricsAccumulator:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    invalid_count: int = 0
    request_error_count: int = 0
    latencies: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    missing_token_counts: bool = False

    def add(self, result: ExampleResult) -> None:
        self.latencies.append(result.latency_seconds)
        if result.request_error:
            self.request_error_count += 1
        if result.input_tokens is None or result.output_tokens is None:
            self.missing_token_counts = True
        else:
            self.input_tokens += result.input_tokens
            self.output_tokens += result.output_tokens

        effective = result.prediction
        if effective is Prediction.INVALID:
            self.invalid_count += 1
            effective = Prediction.NO_MATCH if result.ground_truth is Label.MATCH else Prediction.MATCH
        if result.ground_truth is Label.MATCH and effective is Prediction.MATCH:
            self.tp += 1
        elif result.ground_truth is Label.NO_MATCH and effective is Prediction.NO_MATCH:
            self.tn += 1
        elif result.ground_truth is Label.NO_MATCH:
            self.fp += 1
        else:
            self.fn += 1

    def as_dict(self, *, inference_runtime_seconds: float, total_runtime_seconds: float, resource_metrics: Mapping[str, Any]) -> dict[str, Any]:
        count = self.tp + self.tn + self.fp + self.fn
        if not count:
            raise ValueError("Cannot calculate metrics for zero examples")
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        latency = np.asarray(self.latencies, dtype=float)
        token_total = self.input_tokens + self.output_tokens
        return {
            "total_examples": count,
            "accuracy": (self.tp + self.tn) / count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": self.tp,
            "true_negatives": self.tn,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "invalid_count": self.invalid_count,
            "invalid_rate": self.invalid_count / count,
            "request_error_count": self.request_error_count,
            "inference_runtime_seconds": inference_runtime_seconds,
            "total_runtime_seconds": total_runtime_seconds,
            "average_latency_seconds": float(latency.mean()),
            "median_latency_seconds": float(np.median(latency)),
            "p95_latency_seconds": float(np.quantile(latency, 0.95)),
            "examples_per_second": count / inference_runtime_seconds if inference_runtime_seconds else None,
            "total_input_tokens": None if self.missing_token_counts else self.input_tokens,
            "total_output_tokens": None if self.missing_token_counts else self.output_tokens,
            "total_tokens": None if self.missing_token_counts else token_total,
            "tokens_per_second": (
                token_total / inference_runtime_seconds
                if not self.missing_token_counts and inference_runtime_seconds
                else None
            ),
            "malformed_scoring_policy": "INVALID predictions count as errors: MATCH truth -> FN; NO_MATCH truth -> FP.",
            "estimated_compute_cost_usd": None,
            "compute_cost_note": "Unsupported for local inference without a configured, auditable cost model.",
            "subset_metrics": {},
            **dict(resource_metrics),
        }
