from smolboard_benchmark.metrics import MetricsAccumulator
from smolboard_benchmark.types import ExampleResult, Label, Prediction


def result(truth: Label, prediction: Prediction, *, tokens: bool = True) -> ExampleResult:
    return ExampleResult(
        example_id=f"{truth}-{prediction}", ground_truth=truth, prediction=prediction,
        raw_model_output=prediction.value, valid_prediction=prediction is not Prediction.INVALID,
        latency_seconds=0.2, input_tokens=10 if tokens else None, output_tokens=2 if tokens else None,
        request_error=None, model="fake", prompt_name="baseline", prompt_version="1",
    )


def test_confusion_matrix_metrics_are_manually_verifiable():
    accumulator = MetricsAccumulator()
    for item in (
        result(Label.MATCH, Prediction.MATCH), result(Label.MATCH, Prediction.NO_MATCH),
        result(Label.NO_MATCH, Prediction.MATCH), result(Label.NO_MATCH, Prediction.NO_MATCH),
    ):
        accumulator.add(item)
    metrics = accumulator.as_dict(inference_runtime_seconds=2.0, total_runtime_seconds=2.5, resource_metrics={})
    assert (metrics["true_positives"], metrics["true_negatives"], metrics["false_positives"], metrics["false_negatives"]) == (1, 1, 1, 1)
    assert metrics["accuracy"] == metrics["precision"] == metrics["recall"] == metrics["f1"] == 0.5
    assert metrics["tokens_per_second"] == 24.0


def test_invalid_predictions_are_explicit_and_penalized():
    accumulator = MetricsAccumulator()
    accumulator.add(result(Label.MATCH, Prediction.INVALID))
    accumulator.add(result(Label.NO_MATCH, Prediction.INVALID, tokens=False))
    metrics = accumulator.as_dict(inference_runtime_seconds=1.0, total_runtime_seconds=1.0, resource_metrics={})
    assert metrics["invalid_count"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["tokens_per_second"] is None
