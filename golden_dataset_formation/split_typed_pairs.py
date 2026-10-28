"""Split medium_pairs_typed.parquet into four shuffled parquet parts for manual review.

    .venv\\Scripts\\python golden_dataset_formation\\split_typed_pairs.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets/geo/medium_pairs_typed.parquet"
PARTS = 4

df = pd.read_parquet(SRC).sample(frac=1, random_state=42).reset_index(drop=True)
df["original_llm_pair_type"] = df["llm_pair_type"]
df["review_status"] = "unreviewed"
size = -(-len(df) // PARTS)
for i in range(PARTS):
    out = SRC.with_name(f"medium_pairs_typed_part{i + 1}.parquet")
    part = df.iloc[i * size : (i + 1) * size]
    part.to_parquet(out, index=False)
    print(f"{out.name}: {len(part)} rows")
