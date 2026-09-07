#!/usr/bin/env python3
"""Make naive pair datasets from terraforma_samples_10M.parquet.

A pair is two rows put side by side, plus a label:
    label = 1  -> both rows describe the SAME place (they share an id)
    label = 0  -> the two rows describe DIFFERENT places

"Naive" means no cleaning and no checking. Same id => positive. Different id
=> negative. That is the whole rule.

Rules we agreed on:
  - Half the pairs are positive, half are negative.
  - Each place is used in exactly ONE pair, ever. No reuse.
  - Places are drawn with equal chance, no matter how many rows they have.
  - Which row lands on the left is a coin flip.
  - The finished pairs are shuffled so labels are mixed, not in blocks.
  - Each size is its own independent draw, so the sets may overlap.

NOTE: left_id and right_id are in the output. label is just
      (left_id == right_id), so these columns give the answer away.
      Same for left_provider / right_provider: every positive pair is
      cross-provider, but some negatives are not.
      These are metadata. Strip them before showing a pair to a model.

Run it:
    ./venv/bin/python golden_dataset_formation/make_pairs.py --size micro
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# How many pairs each named size holds.
SIZES = {"micro": 1_000, "mini": 3_000, "small": 10_000, "medium": 50_000}

# Each size gets its own seed so the four draws are independent.
# Without this, micro and mini would start with the very same pairs.
SEED_STEP = {"micro": 0, "mini": 1, "small": 2, "medium": 3}


def build_index(source):
    """Pass 1: read only the id column and group row numbers by place.

    We never load the whole file. The id column alone is small.

    Returns:
        order  - row numbers sorted so that rows of one place sit together
        starts - where each place begins inside `order`
        counts - how many rows each place has
    """
    ids = pq.read_table(source, columns=["id"]).column("id").combine_chunks()

    # Turn each id string into a small integer code. Same id -> same code.
    codes = ids.dictionary_encode()
    n_places = len(codes.dictionary)
    place_of_row = codes.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    del ids, codes

    # Sort row numbers by place code, so one place's rows are next to each other.
    order = np.argsort(place_of_row, kind="stable")
    sorted_codes = place_of_row[order]
    del place_of_row

    all_places = np.arange(n_places)
    starts = np.searchsorted(sorted_codes, all_places, side="left")
    counts = np.searchsorted(sorted_codes, all_places, side="right") - starts

    # Every place must have at least 2 rows, or it cannot make a positive pair.
    assert counts.min() >= 2, "found a place with only one row"
    return order, starts, counts


def pick_pairs(order, starts, counts, n_pairs, seed):
    """Choose the row numbers for every pair. Returns (left, right, label)."""
    rng = np.random.default_rng(seed)
    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos

    # Draw all the places we need in one go, with no repeats.
    # A positive pair eats 1 place. A negative pair eats 2.
    picked = rng.choice(len(counts), size=n_pos + 2 * n_neg, replace=False)
    pos_places = picked[:n_pos]
    neg_places = picked[n_pos:].reshape(n_neg, 2)

    def one_row(places):
        """Pick 1 random row from each place."""
        offset = (rng.random(len(places)) * counts[places]).astype(np.int64)
        return order[starts[places] + offset]

    # Positive pair: two DIFFERENT rows from the same place.
    # Pick the first row, then pick the second from what is left and shift it
    # up if it would collide. That keeps the two rows distinct.
    c = counts[pos_places]
    first = (rng.random(n_pos) * c).astype(np.int64)
    second = (rng.random(n_pos) * (c - 1)).astype(np.int64)
    second += second >= first
    pos_a = order[starts[pos_places] + first]
    pos_b = order[starts[pos_places] + second]

    # Negative pair: one row from each of two different places.
    neg_a = one_row(neg_places[:, 0])
    neg_b = one_row(neg_places[:, 1])

    side_a = np.concatenate([pos_a, neg_a])
    side_b = np.concatenate([pos_b, neg_b])
    label = np.concatenate([np.ones(n_pos, np.int8), np.zeros(n_neg, np.int8)])

    # Coin flip: decide which of the two rows goes on the left.
    flip = rng.random(n_pairs) < 0.5
    left = np.where(flip, side_b, side_a)
    right = np.where(flip, side_a, side_b)

    # Shuffle so positives and negatives are mixed together.
    mix = rng.permutation(n_pairs)
    return left[mix], right[mix], label[mix]


def fetch_rows(source, wanted):
    """Pass 2: read the file one row-group at a time and keep only what we need.

    The file unpacks to ~2.9 GB. One row group is ~250 MB, so this stays small.
    `wanted` must be sorted. The result rows come back in that same order.
    """
    pf = pq.ParquetFile(source)
    sizes = [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)]
    edges = np.concatenate([[0], np.cumsum(sizes)])

    parts = []
    for i in range(len(sizes)):
        lo, hi = edges[i], edges[i + 1]
        here = wanted[(wanted >= lo) & (wanted < hi)]
        if len(here):
            parts.append(pf.read_row_group(i).take(pa.array(here - lo)))
    return pa.concat_tables(parts)


def make(source, size, seed, out_dir, index):
    n_pairs = SIZES[size]
    started = time.time()

    left, right, label = pick_pairs(*index, n_pairs, seed + SEED_STEP[size])

    # Read each needed row once, then point both sides at it.
    wanted = np.unique(np.concatenate([left, right]))
    rows = fetch_rows(source, wanted)
    left_rows = rows.take(pa.array(np.searchsorted(wanted, left)))
    right_rows = rows.take(pa.array(np.searchsorted(wanted, right)))

    cols = {"pair_id": pa.array(np.arange(n_pairs)), "label": pa.array(label)}
    for name in rows.schema.names:
        cols[f"left_{name}"] = left_rows.column(name)
        cols[f"right_{name}"] = right_rows.column(name)
    table = pa.table(cols)

    # --- check our own work before writing anything ---
    lid = np.array(table.column("left_id").to_pylist())
    rid = np.array(table.column("right_id").to_pylist())
    same = lid == rid
    assert table.num_rows == n_pairs, "wrong number of pairs"
    assert label.sum() == n_pairs // 2, "labels are not half and half"
    assert len(wanted) == 2 * n_pairs, "a row got used more than once"
    assert same[label == 1].all(), "a positive pair has two different ids"
    assert not same[label == 0].any(), "a negative pair has the same id twice"

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{size}_pairs.parquet"
    pq.write_table(table, path)

    stats = {
        "size": size,
        "pairs": n_pairs,
        "same_place_pairs": int(label.sum()),
        "different_place_pairs": int(n_pairs - label.sum()),
        "rows_used": len(wanted),
        "seed": seed + SEED_STEP[size],
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / f"{size}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{size}: {n_pairs} pairs -> {path}  ({stats['seconds']}s)")


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", default="micro", choices=[*SIZES, "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", type=Path, default=here.parent / "terraforma_samples_10M.parquet")
    p.add_argument("--out", type=Path, default=here / "output")
    a = p.parse_args()

    print("reading ids...")
    index = build_index(a.source)
    print(f"found {len(index[2]):,} places")

    for size in (SIZES if a.size == "all" else [a.size]):
        make(a.source, size, a.seed, a.out, index)


if __name__ == "__main__":
    main()
