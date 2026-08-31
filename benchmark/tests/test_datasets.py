from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from smolboard_benchmark.datasets import ParquetConflationDataset
from smolboard_benchmark.prompts import build_prompt, load_prompt


def dataset_config(path: Path) -> dict:
    return {
        "path": str(path), "label_column": "label", "positive_label": 1,
        "example_id": "generated_row_index",
        "record_a_columns": {"names": "names", "phones": "phones"},
        "record_b_columns": {"names": "base_names", "phones": "base_phones"},
        "metadata_columns": {"source_record_id": "id"},
    }


def test_parquet_rows_normalize_and_balanced_limit(tmp_path: Path):
    path = tmp_path / "pairs.parquet"
    pq.write_table(pa.table({
        "label": [1, 0, 1, 0], "names": ['{"primary":"A"}', '{"primary":"B"}', '{"primary":"C"}', '{"primary":"D"}'],
        "phones": [None, '["2"]', None, None], "base_names": ['{"primary":"AA"}', '{"primary":"BB"}', '{"primary":"CC"}', '{"primary":"DD"}'],
        "base_phones": [None, None, None, None], "id": ["a", "b", "c", "d"],
    }), path)
    dataset = ParquetConflationDataset("test", dataset_config(path))
    examples = list(dataset.iter_examples(batch_size=2, limit=4))
    assert [example.example_id for example in examples] == ["row_000000", "row_000001", "row_000002", "row_000003"]
    assert examples[0].metadata["source_record_id"] == "a"
    prompt = load_prompt("baseline", {"prompts": {"baseline": {"version": 1, "exposed_fields": ["names", "phones"], "template": "A={record_a};B={record_b}"}}})
    rendered = build_prompt(examples[0], prompt)
    assert '"names": {"primary": "A"}' in rendered
    assert "source_record_id" not in rendered
