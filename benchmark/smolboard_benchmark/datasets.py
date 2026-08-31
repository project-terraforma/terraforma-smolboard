"""Streaming dataset normalization for place-conflation benchmarks."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow.parquet as pq

from .config import resolve_workspace_path
from .types import ConflationExample, Label, PlaceRecord


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ParquetConflationDataset:
    """Maps a configured Parquet schema into streaming normalized examples."""

    def __init__(self, name: str, config: Mapping[str, Any]) -> None:
        self.name = name
        self.config = dict(config)
        self.path = resolve_workspace_path(str(self.config["path"]))
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        self.label_column = str(self.config["label_column"])
        self.positive_label = int(self.config.get("positive_label", 1))
        self.record_a_columns = dict(self.config["record_a_columns"])
        self.record_b_columns = dict(self.config["record_b_columns"])
        self.metadata_columns = dict(self.config.get("metadata_columns") or {})
        self.example_id_kind = str(self.config.get("example_id", "generated_row_index"))
        self._validate_schema()

    def _validate_schema(self) -> None:
        schema_names = set(pq.ParquetFile(self.path).schema_arrow.names)
        required = {self.label_column, *self.record_a_columns.values(), *self.record_b_columns.values()}
        required.update(self.metadata_columns.values())
        missing = sorted(required - schema_names)
        if missing:
            raise ValueError(f"Dataset '{self.name}' is missing configured columns: {', '.join(missing)}")
        if self.example_id_kind != "generated_row_index":
            raise ValueError(f"Unsupported example_id strategy: {self.example_id_kind}")

    @property
    def columns(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    self.label_column,
                    *self.record_a_columns.values(),
                    *self.record_b_columns.values(),
                    *self.metadata_columns.values(),
                ]
            )
        )

    def iter_examples(
        self, *, batch_size: int, limit: int | None = None
    ) -> Iterator[ConflationExample]:
        if limit is not None and limit < 2:
            raise ValueError("--limit must be at least 2 so both labels are checked")
        parquet_file = pq.ParquetFile(self.path)
        quotas = None
        selected = Counter()
        if limit is not None:
            quotas = {Label.MATCH: (limit + 1) // 2, Label.NO_MATCH: limit // 2}

        row_index = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=self.columns):
            columns = batch.to_pydict()
            for position in range(batch.num_rows):
                example = self._normalize_row(columns, position, row_index)
                row_index += 1
                if quotas is not None:
                    if selected[example.label] >= quotas[example.label]:
                        continue
                    selected[example.label] += 1
                yield example
                if quotas is not None and all(selected[label] >= quotas[label] for label in quotas):
                    return
        if quotas is not None:
            raise ValueError(f"Dataset cannot provide the requested balanced limit of {limit} examples")

    def _normalize_row(
        self, columns: Mapping[str, list[Any]], position: int, row_index: int
    ) -> ConflationExample:
        raw_label = columns[self.label_column][position]
        if raw_label is None:
            raise ValueError(f"Null label at dataset row {row_index}")
        label_value = int(raw_label)
        if label_value not in {0, 1}:
            raise ValueError(f"Expected binary label 0/1 at row {row_index}, got {raw_label!r}")
        label = Label.MATCH if label_value == self.positive_label else Label.NO_MATCH
        record_a = {field: columns[column][position] for field, column in self.record_a_columns.items()}
        record_b = {field: columns[column][position] for field, column in self.record_b_columns.items()}
        metadata = {name: columns[column][position] for name, column in self.metadata_columns.items()}
        metadata["row_index"] = row_index
        return ConflationExample(
            example_id=f"row_{row_index:06d}",
            record_a=PlaceRecord(record_a),
            record_b=PlaceRecord(record_b),
            label=label,
            metadata=metadata,
        )

    def manifest(self) -> dict[str, Any]:
        counts: Counter[Label] = Counter()
        total = 0
        for example in self.iter_examples(batch_size=4096):
            counts[example.label] += 1
            total += 1
        if not counts[Label.MATCH] or not counts[Label.NO_MATCH]:
            raise ValueError("Dataset must contain both MATCH and NO_MATCH rows")
        file_hash = sha256_file(self.path)
        return {
            "dataset_name": self.name,
            "dataset_path": str(self.path),
            "dataset_sha256": file_hash,
            "dataset_version": f"sha256:{file_hash[:12]}",
            "dataset_split": None,
            "total_examples": total,
            "class_counts": {Label.MATCH.value: counts[Label.MATCH], Label.NO_MATCH.value: counts[Label.NO_MATCH]},
            "label_mapping": {str(self.positive_label): Label.MATCH.value, str(1 - self.positive_label): Label.NO_MATCH.value},
            "example_id_definition": "zero-padded original Parquet row index",
            "metadata_columns": self.metadata_columns,
        }
