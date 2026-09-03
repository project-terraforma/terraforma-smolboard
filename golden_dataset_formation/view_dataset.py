#!/usr/bin/env python3
"""View pairs from an already-generated dataset file (e.g. micro_pairs.parquet)
in the same readable format preview_pairs.py uses -- the difference is this
reads the real, finished file instead of generating a fresh sample, so it's
what to use for reviewing an actual deliverable rather than a dry run.

Run it:
    python view_dataset.py output_geo/micro_pairs.parquet --n 30
    python view_dataset.py output_geo/micro_pairs.parquet --n 20 --type hard_negative
    python view_dataset.py output_geo/micro_pairs.parquet --n 20 --shuffle --seed 3
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def flatten(value, preferred_keys=(), max_len=70):
    """Same best-effort nested-field flattener as preview_pairs.py."""
    if value is None:
        return "-"
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "-"
        return flatten(value[0], preferred_keys, max_len)
    if isinstance(value, dict):
        for k in preferred_keys:
            if k in value and value[k]:
                return flatten(value[k], (), max_len)
        for v in value.values():
            if v:
                return flatten(v, (), max_len)
        return "-"
    s = str(value)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def print_pair(i, row):
    label = row["label"]
    verdict = "MATCH" if label == 1 else "NO_MATCH"
    hard_source = row.get("hard_neg_source") or ""
    type_str = f"{row['pair_type']} [{hard_source}]" if hard_source else row["pair_type"]
    print(f"{'=' * 20} PAIR {i:03d} (row {row['pair_id']}) {'=' * 20}")
    print(f"label: {label} ({verdict})   pair_type: {type_str}   "
          f"distance_m: {row['distance_m']:.2f}")
    print()
    for side in ("left", "right"):
        provider = row.get(f"{side}_provider", "-")
        rid = row.get(f"{side}_id", "-")
        print(f"{side.upper():<5} [{provider}]  id={rid}")
        print(f"  name:     {flatten(row.get(f'{side}_names'), ('primary', 'common'))}")
        print(f"  address:  {flatten(row.get(f'{side}_addresses'), ('freeform', 'address'))}")
        print(f"  category: {row.get(f'{side}_basic_category', '-')}")
        print(f"  confidence: {row.get(f'{side}_confidence', '-')}")
        print()
    print("=" * 54)
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", type=Path, help="a *_pairs.parquet file to review")
    p.add_argument("--n", type=int, default=30, help="how many pairs to print")
    p.add_argument("--type", choices=["match", "hard_negative", "easy_negative"],
                    help="only show pairs of this pair_type")
    p.add_argument("--hard-source", choices=["geo", "brand"],
                    help="only show hard_negative pairs from this source "
                         "(implies --type hard_negative)")
    p.add_argument("--shuffle", action="store_true", help="shuffle before taking --n")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rows = pq.read_table(a.file).to_pylist()
    print(f"{a.file}: {len(rows)} total pairs")

    want_type = a.type or ("hard_negative" if a.hard_source else None)
    if want_type:
        rows = [r for r in rows if r["pair_type"] == want_type]
        print(f"  {len(rows)} are pair_type={want_type}")

    if a.hard_source:
        rows = [r for r in rows if r.get("hard_neg_source") == a.hard_source]
        print(f"  {len(rows)} are hard_neg_source={a.hard_source}")

    if a.shuffle:
        import random
        random.Random(a.seed).shuffle(rows)

    print()
    for i, row in enumerate(rows[: a.n], 1):
        print_pair(i, row)

    from collections import Counter
    print(f"Showed {min(a.n, len(rows))} of {len(rows)}. "
          f"Full file breakdown: {dict(Counter(r['pair_type'] for r in pq.read_table(a.file).to_pylist()))}")


if __name__ == "__main__":
    main()
