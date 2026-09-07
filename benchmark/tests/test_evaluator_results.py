from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from smolboard_benchmark.adapters.base import ModelAdapter
from smolboard_benchmark.datasets import ParquetConflationDataset
from smolboard_benchmark.evaluator import run_evaluation
from smolboard_benchmark.prompts import load_prompt
from smolboard_benchmark.results import FileResultWriter
from smolboard_benchmark.types import ExampleResult, Label, ModelResponse, Prediction


class FakeAdapter(ModelAdapter):
    def __init__(self):
        self.started = False

    def start(self, log_path: str) -> None:
        self.started = True
        Path(log_path).write_text("fake\n", encoding="utf-8")

    def generate_batch(self, prompts):
        return [ModelResponse("MATCH", 10, 2, 0.01) for _ in prompts]

    def stop(self):
        return {"backend": "fake", "gpu_memory_peak_total_mib": None}


def test_evaluator_persists_completed_ordered_run(tmp_path: Path):
    parquet_path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({
        "label": [1, 0], "names": ["A", "B"], "base_names": ["AA", "BB"], "id": ["a", "b"],
    }), parquet_path)
    dataset = ParquetConflationDataset("test", {
        "path": str(parquet_path), "label_column": "label", "positive_label": 1, "example_id": "generated_row_index",
        "record_a_columns": {"names": "names"}, "record_b_columns": {"names": "base_names"}, "metadata_columns": {"source_id": "id"},
    })
    prompt = load_prompt("baseline", {"prompts": {"baseline": {"version": 1, "exposed_fields": ["names"], "template": "{record_a} {record_b}"}}})
    writer = FileResultWriter(tmp_path / "run", {"run_fingerprint": "same"})
    result = run_evaluation(
        dataset=dataset, prompt=prompt, model_name="fake", adapter=FakeAdapter(), writer=writer,
        dataset_batch_size=2, evaluator_batch_size=2, limit=None, expected_examples=2,
        run_metadata={"run_id": "test"},
    )
    assert result["metrics"]["total_examples"] == 2
    assert (tmp_path / "run" / "predictions.jsonl").exists()
    assert not (tmp_path / "run" / "predictions.checkpoint.jsonl").exists()
    assert (tmp_path / "run" / "plots" / "metrics.png").exists()
    assert (tmp_path / "run" / "plots" / "confusion_matrix.png").exists()


def test_interrupted_checkpoint_resumes_without_duplicate_rows(tmp_path: Path):
    parquet_path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({
        "label": [1, 0], "names": ["A", "B"], "base_names": ["AA", "BB"], "id": ["a", "b"],
    }), parquet_path)
    dataset = ParquetConflationDataset("test", {
        "path": str(parquet_path), "label_column": "label", "positive_label": 1, "example_id": "generated_row_index",
        "record_a_columns": {"names": "names"}, "record_b_columns": {"names": "base_names"}, "metadata_columns": {"source_id": "id"},
    })
    prompt = load_prompt("baseline", {"prompts": {"baseline": {"version": 1, "exposed_fields": ["names"], "template": "{record_a} {record_b}"}}})
    run_dir = tmp_path / "resumed-run"
    initial = FileResultWriter(run_dir, {"run_fingerprint": "same"})
    initial.append(ExampleResult(
        example_id="row_000000", ground_truth=Label.MATCH, prediction=Prediction.MATCH,
        raw_model_output="MATCH", valid_prediction=True, latency_seconds=0.01, input_tokens=10, output_tokens=2,
        request_error=None, model="fake", prompt_name="baseline", prompt_version="1", metadata={"row_index": 0},
    ))
    initial.interrupt("test interruption")
    resumed = FileResultWriter(run_dir, {"run_fingerprint": "same"})
    result = run_evaluation(
        dataset=dataset, prompt=prompt, model_name="fake", adapter=FakeAdapter(), writer=resumed,
        dataset_batch_size=2, evaluator_batch_size=2, limit=None, expected_examples=2, run_metadata={"run_id": "resumed"},
    )
    rows = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert result["resumed_examples"] == 1
    assert len(rows) == 2
