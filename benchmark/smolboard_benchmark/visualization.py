"""Small, model-independent plots generated from normalized metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def make_run_plots(metrics: Mapping[str, Any], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    _metric_plot(metrics, plot_dir / "metrics.png")
    _confusion_matrix(metrics, plot_dir / "confusion_matrix.png")


def _metric_plot(metrics: Mapping[str, Any], path: Path) -> None:
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [float(metrics[key]) for key in ("accuracy", "precision", "recall", "f1")]
    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    bars = axis.bar(names, values, color="#3b82f6", edgecolor="#1e3a8a")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Benchmark classification metrics")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.3f", padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _confusion_matrix(metrics: Mapping[str, Any], path: Path) -> None:
    matrix = np.asarray(
        [
            [metrics["true_positives"], metrics["false_negatives"]],
            [metrics["false_positives"], metrics["true_negatives"]],
        ]
    )
    fig, axis = plt.subplots(figsize=(4.8, 4.2))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["MATCH", "NO_MATCH"])
    axis.set_yticks([0, 1], ["MATCH", "NO_MATCH"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
