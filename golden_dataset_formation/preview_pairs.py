#!/usr/bin/env python3
"""Print a small, readable batch of MATCH / hard-negative / easy-negative
pairs to the terminal, for quick QA before committing to a full generation
run.

This is a thin QA wrapper, not a separate implementation: it imports and
calls the real candidate-search and pair-assembly functions straight out
of make_pairs_geo.py, so it must live in the same folder as that file. The
one place it's deliberately cheaper is --hard-neg-candidates, which
defaults much lower here (5,000 instead of 300,000) since a small preview
only needs a couple dozen hard negatives, not enough to justify the full
search.

Run it (from the same folder as make_pairs_geo.py):
    python preview_pairs.py --n 60 --source terraforma_samples_10M.parquet
    python preview_pairs.py --n 100 --seed 7 --source terraforma_samples_10M.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pairs_geo as geo


def flatten(value, preferred_keys=(), max_len=70):
    """Best-effort turn an Overture nested field (struct/list/etc.) into one
    readable line. Overture's `names` is a struct with a `primary` field;
    `addresses` is a list of structs with a `freeform` field -- but schemas
    like this can vary, so this degrades gracefully instead of crashing."""
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
        # fall back to the first non-empty value in the struct
        for v in value.values():
            if v:
                return flatten(v, (), max_len)
        return "-"
    s = str(value)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def print_pair(i, label, pair_type, hard_source, dist, left, right):
    verdict = "MATCH" if label == 1 else "NO_MATCH"
    type_str = f"{pair_type} [{hard_source}]" if hard_source else pair_type
    print(f"{'=' * 20} PAIR {i:03d} {'=' * 20}")
    print(f"label: {label} ({verdict})   pair_type: {type_str}   distance_m: {dist:.2f}")
    print()
    for side_name, row in (("LEFT", left), ("RIGHT", right)):
        provider = row.get("provider", "-")
        rid = row.get("id", "-")
        print(f"{side_name:<5} [{provider}]  id={rid}")
        print(f"  name:     {flatten(row.get('names'), ('primary', 'common'))}")
        print(f"  address:  {flatten(row.get('addresses'), ('freeform', 'address'))}")
        print(f"  category: {row.get('basic_category', '-')}")
        print(f"  confidence: {row.get('confidence', '-')}")
        print()
    print("=" * 54)
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    p.add_argument("--n", type=int, default=60, help="total pairs to preview")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", type=Path, default=here.parent / "terraforma_samples_10M.parquet")
    p.add_argument("--match-max-m", type=float, default=geo.MATCH_MAX_M)
    p.add_argument("--hard-neg-max-m", type=float, default=geo.HARD_NEG_MAX_M)
    p.add_argument("--easy-neg-min-m", type=float, default=geo.EASY_NEG_MIN_M)
    p.add_argument("--hard-neg-fraction", type=float, default=geo.HARD_NEG_FRACTION)
    p.add_argument("--hard-neg-candidates", type=int, default=5_000,
                    help="kept small by default so the preview stays fast")
    p.add_argument("--max-pairs-per-cell", type=int, default=5,
                    help="cap on candidates from any single dense cluster")
    p.add_argument("--brand-hard-neg-fraction", type=float, default=geo.BRAND_HARD_NEG_FRACTION)
    p.add_argument("--brand-hard-neg-min-m", type=float, default=geo.BRAND_HARD_NEG_MIN_M)
    p.add_argument("--brand-hard-neg-max-m", type=float, default=geo.BRAND_HARD_NEG_MAX_M)
    p.add_argument("--brand-hard-neg-candidates", type=int, default=5_000,
                    help="kept small by default so the preview stays fast")
    p.add_argument("--max-rows-per-brand-group", type=int, default=500)
    p.add_argument("--max-pairs-per-brand-group", type=int, default=5)
    p.add_argument("--skip-alignment-check", action="store_true")
    a = p.parse_args()

    print("reading ids...")
    ids_column = pq.read_table(a.source, columns=["id"]).column("id")
    index = geo.build_index(ids_column)
    order, starts, counts = index
    print(f"found {len(counts):,} places")

    print("decoding geometry (lon/lat) from WKB...")
    lon, lat = geo.build_geo(a.source)
    if not a.skip_alignment_check:
        geo.verify_alignment(a.source, ids_column)

    categories = geo.encode_categories(a.source)

    print(f"finding MATCH candidates (<{a.match_max_m}m)...")
    pos_cand = geo.find_positive_candidates(order, starts, counts, lon, lat, a.match_max_m)
    print(f"  {len(pos_cand[0]):,} verified MATCH places available")

    print(f"finding geo hard-negative candidates (<{a.hard_neg_max_m}m, preferring same "
          f"category, capped at {a.hard_neg_candidates:,})...")
    hardneg_geo_cand = geo.find_hard_negative_candidates(
        ids_column, lon, lat, max_m=a.hard_neg_max_m, categories=categories,
        max_candidates=a.hard_neg_candidates, max_pairs_per_cell=a.max_pairs_per_cell)
    print(f"  {len(hardneg_geo_cand[0]):,} geo hard-negative candidate pairs available "
          f"({int(hardneg_geo_cand[3].sum()):,} same-category)")

    print("extracting brand/name keys and localities...")
    name_key = geo.extract_brand_or_name_key(a.source)
    locality_key = geo.extract_locality(a.source)
    print(f"finding brand hard-negative candidates (>={a.brand_hard_neg_min_m}m and "
          f"<={a.brand_hard_neg_max_m}m, preferring same locality, "
          f"capped at {a.brand_hard_neg_candidates:,})...")
    hardneg_brand_cand = geo.find_brand_hard_negative_candidates(
        ids_column, name_key, lon, lat, min_m=a.brand_hard_neg_min_m, max_m=a.brand_hard_neg_max_m,
        locality_keys=locality_key, max_candidates=a.brand_hard_neg_candidates,
        max_group_rows=a.max_rows_per_brand_group, max_pairs_per_group=a.max_pairs_per_brand_group)
    print(f"  {len(hardneg_brand_cand[0]):,} brand hard-negative candidate pairs available "
          f"({int(hardneg_brand_cand[3].sum()):,} same-locality)")
    print()

    left, right, label, pair_type, dist, hard_source, n_geo_same_cat, n_brand_same_loc = geo.pick_pairs_geo(
        index, lon, lat, categories, pos_cand, hardneg_geo_cand, hardneg_brand_cand,
        a.n, a.seed, a.hard_neg_fraction, a.brand_hard_neg_fraction, a.easy_neg_min_m)

    wanted = np.unique(np.concatenate([left, right]))
    rows = geo.fetch_rows(a.source, wanted)
    import pyarrow as pa
    left_rows = rows.take(pa.array(np.searchsorted(wanted, left))).to_pylist()
    right_rows = rows.take(pa.array(np.searchsorted(wanted, right))).to_pylist()

    for i in range(a.n):
        print_pair(i + 1, int(label[i]), pair_type[i], hard_source[i], float(dist[i]),
                   left_rows[i], right_rows[i])

    from collections import Counter
    print(f"Summary: {dict(Counter(pair_type.tolist()))}  "
          f"(of the drawn hard negs: {n_geo_same_cat} geo were same-category, "
          f"{n_brand_same_loc} brand were same-locality)")


if __name__ == "__main__":
    main()
