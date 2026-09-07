"""Bounded, resumable evaluation over normalized examples."""

from __future__ import annotations

import itertools
import time
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Sequence

from .adapters.base import ModelAdapter
from .datasets import ParquetConflationDataset
from .metrics import MetricsAccumulator
from .parsing import parse_prediction
from .prompts import build_prompt
from .results import FileResultWriter
from .types import ConflationExample, ExampleResult, PromptSpec
from .visualization import make_run_plots


def run_evaluation(
    *,
    dataset: ParquetConflationDataset,
    prompt: PromptSpec,
    model_name: str,
    adapter: ModelAdapter,
    writer: FileResultWriter,
    dataset_batch_size: int,
    evaluator_batch_size: int,
    limit: int | None,
    expected_examples: int,
    run_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate batches in deterministic order, safely continuing a validated checkpoint."""
    total_started = time.perf_counter()
    accumulator = MetricsAccumulator()
    examples = dataset.iter_examples(batch_size=dataset_batch_size, limit=limit)
    resumed = _consume_checkpoint(writer, examples, accumulator)
    completed = resumed
    inference_started: float | None = None
    adapter_metadata: dict[str, Any] = {}
    try:
        adapter.start(str(writer.log_path))
        inference_started = time.perf_counter()
        for batch in _batched(examples, evaluator_batch_size):
            prompts = [build_prompt(example, prompt) for example in batch]
            responses = adapter.generate_batch(prompts)
            if len(responses) != len(batch):
                raise RuntimeError("Model adapter returned a response count different from the prompt count")
            for example, response in zip(batch, responses, strict=True):
                prediction = parse_prediction(response.raw_output)
                result = ExampleResult(
                    example_id=example.example_id,
                    ground_truth=example.label,
                    prediction=prediction,
                    raw_model_output=response.raw_output,
                    valid_prediction=prediction.value != "INVALID",
                    latency_seconds=response.latency_seconds,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    request_error=response.request_error,
                    model=model_name,
                    prompt_name=prompt.name,
                    prompt_version=prompt.version,
                    metadata=example.metadata,
                )
                writer.append(result)
                accumulator.add(result)
                completed += 1
                if completed % 100 == 0 or completed == expected_examples:
                    print(f"Completed {completed}/{expected_examples} examples", flush=True)
        inference_runtime = time.perf_counter() - inference_started
    except KeyboardInterrupt:
        writer.interrupt("KeyboardInterrupt")
        raise
    except Exception as exc:
        writer.interrupt(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        adapter_metadata = adapter.stop()

    total_runtime = time.perf_counter() - total_started
    metrics = accumulator.as_dict(
        inference_runtime_seconds=inference_runtime,
        total_runtime_seconds=total_runtime,
        resource_metrics=adapter_metadata,
    )
    result = {
        **dict(run_metadata),
        "completed_at": _utc_now(),
        "resumed_examples": resumed,
        "partial_run": limit is not None,
        "leaderboard_eligible": limit is None,
        "metrics": metrics,
    }
    make_run_plots(metrics, writer.run_dir / "plots")
    writer.complete(metrics, result)
    return result


def _consume_checkpoint(
    writer: FileResultWriter,
    examples: Iterator[ConflationExample],
    accumulator: MetricsAccumulator,
) -> int:
    resumed = 0
    for result in writer.iter_checkpoint():
        try:
            expected = next(examples)
        except StopIteration as exc:
            raise ValueError("Checkpoint contains more rows than the current dataset selection") from exc
        if result.example_id != expected.example_id or result.ground_truth is not expected.label:
            raise ValueError("Checkpoint does not match the intended dataset order and labels")
        accumulator.add(result)
        resumed += 1
    return resumed


def _batched(values: Iterator[ConflationExample], size: int) -> Iterator[list[ConflationExample]]:
    if size < 1:
        raise ValueError("Evaluator batch size must be positive")
    while batch := list(itertools.islice(values, size)):
        yield batch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
