#!/usr/bin/env python3
"""Make LOCATION-AWARE pair datasets from terraforma_samples_10M.parquet.

This is the companion to Mann's make_pairs.py ("naive" datasets, where a
shared Overture id alone means MATCH). We proved with DuckDB earlier that a
shared id is NOT trustworthy on its own -- 2.27M of the 6.95M shared-id
cross-provider pairs are more than 5km apart. So this script adds geometry
into the label:

    MATCH          -> same Overture id, AND the two points are < match-max-m
                      apart (default 5m).
    HARD_NEGATIVE  -> DIFFERENT Overture id, but the metadata looks
                      deceptively similar. Two sources feed this bucket,
                      and BOTH now require metadata plausibility, not just
                      geometric proximity, since geometry is never shown
                      to the model (see the "why metadata, not just
                      distance" note further down):
                        - geo:   close together (< hard-neg-max-m apart,
                                 default 10m -- e.g. two different shops in
                                 the same strip mall), PREFERRING pairs
                                 that also share a basic_category (two
                                 competing dentists in the same plaza) over
                                 ones that don't (a hotel next to a
                                 restaurant is close but not confusable on
                                 text alone, so it's a weak hard negative).
                        - brand: same brand/name text (e.g. two different
                                 McDonald's), >= brand-hard-neg-min-m apart
                                 so they're confidently different physical
                                 locations rather than the same place seen
                                 through two providers, PREFERRING pairs
                                 that also share a locality (same city,
                                 different street) over ones that don't
                                 (different countries is just as easy a
                                 tell as a mismatched name), and bounded
                                 by brand-hard-neg-max-m as a backstop.
                      Which source produced a given pair is in the
                      hard_neg_source column ("geo" / "brand").
    EASY_NEGATIVE  -> different id, far apart (>= easy-neg-min-m, default
                      1000m), and where possible a different top-level
                      category too (McDonald's vs. a ski resort).

Same-id pairs that are >= match-max-m apart are dropped entirely -- not
MATCH, not kept as "uncertain". We have 10M rows and only need 50K pairs
max, so there's no need to keep borderline cases.

Rules carried over from make_pairs.py:
  - Each size is 50% MATCH / 50% NO_MATCH.
  - Of the NO_MATCH half, hard-neg-fraction (default 40%) are hard
    negatives and the rest are easy negatives -- this matches the OKR's
    10,000 hard / 25,000 no-match target for the 50K tier.
  - Each place (Overture id) is used in exactly ONE pair, ever, within a
    given size's draw. No reuse.
  - Which row lands on "left" is a coin flip, and the finished pairs are
    shuffled so labels aren't in blocks.
  - Each size is its own independent draw (same as make_pairs.py), so the
    four output files may overlap with each other.

Output columns match make_pairs.py's convention (pair_id, label, then
left_<col>/right_<col> for every column in the source file), plus three
new ones:
    pair_type       -> "match" / "hard_negative" / "easy_negative"
    hard_neg_source -> "geo" / "brand" for hard_negative rows, "" otherwise
    distance_m      -> haversine distance in meters between the two points

NOTE: left_id/right_id, pair_type, hard_neg_source and distance_m all give
the label away, same caveat as make_pairs.py -- strip them before showing
a pair to a model.

SETUP:
    Nothing beyond what make_pairs.py already uses (numpy, pyarrow). No
    duckdb, no network, no extensions -- geometry is decoded directly from
    the WKB bytes (see build_geo() for why that's safe for this dataset).

RUN IT:
    python make_pairs_geo.py --size micro --source terraforma_samples_10M.parquet
    python make_pairs_geo.py --size all --source terraforma_samples_10M.parquet --out datasets/geo

HOW THE HARD-NEGATIVE SEARCH WORKS (the one genuinely new piece of
machinery here): finding "different id, but within 10m" pairs among 10M
points can't be done as a full pairwise comparison. Instead we bucket every
point into a square grid whose cell size is >= hard-neg-max-m. Any two
points closer than that threshold must land in the same cell or one of its
8 neighbors, so we only ever compare points within a 3x3 block of cells,
never the full 10M x 10M space. We walk the densest cells first (that's
where close-together different-id points actually live) and stop once
we've gathered enough candidates. This means the search isn't perfectly
exhaustive -- a small number of valid pairs that straddle a dense/sparse
cell boundary can be missed if we hit the candidate cap first -- but we
only need a few thousand candidates out of what should be a much larger
true population, so that's fine.

SECOND HARD-NEGATIVE SOURCE -- brand/name matching: the geometric search
above finds pairs that are hard because of *coordinates* (which the model
never sees, since distance_m and geometry are stripped before showing a
pair to a model) -- but their name/category text is usually easy to tell
apart on sight (a TeamLogic IT next to a cleaning company). That leaves a
real gap: a model that pattern-matches on name text alone would sail
through those. The brand/name search closes it from the other direction:
group rows by normalized brand name (falling back to the place's own
primary name when there's no brand), and within a group pick pairs that
are far enough apart (>= brand-hard-neg-min-m) to be confidently different
physical locations -- e.g. two different Starbucks. Same name, same
category, same everything except address/locality text, which is exactly
what the model has to read to get it right. The two sources are additive,
not a replacement for each other -- see hard_neg_source in the output.

WHY METADATA, NOT JUST DISTANCE (added after eyeballing the first batch of
real output): a pair like "Bros Inn Hotel" 3m from "Francesco's" (different
category, unrelated name) LOOKS hard because of geometry, but a model that
never sees geometry can call it NO_MATCH on sight -- distance alone isn't
a difficulty signal once it's hidden from the reader. And the brand search
had a mirror-image gap: with no ceiling on distance, two same-name places
on opposite sides of the planet are just as easy to rule out, since a
wildly different country/city in the address text is as obvious a tell as
a mismatched name. So both searches now PREFER (not strictly require, so
the candidate pool doesn't dry up) pairs where the *other* signal also
looks confusable: geo prefers same basic_category, brand prefers same
locality and is capped by brand-hard-neg-max-m. Distance itself keeps
exactly one job either way -- proving two rows are genuinely different
real-world places -- never a feature the model sees.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# How many pairs each named size holds -- same tiers Mann used.
SIZES = {"micro": 1_000, "mini": 3_000, "small": 10_000, "medium": 50_000}
SEED_STEP = {"micro": 0, "mini": 1, "small": 2, "medium": 3}

# Defaults (all overridable from the CLI -- see main()).
MATCH_MAX_M = 5.0
HARD_NEG_MAX_M = 10.0
EASY_NEG_MIN_M = 1_000.0
HARD_NEG_FRACTION = 0.4

# Of the hard-negative budget, what fraction comes from the brand/name
# search vs. the geometric (grid) search -- split evenly by default since
# the two test different failure modes and neither subsumes the other.
BRAND_HARD_NEG_FRACTION = 0.5
# Floor distance for a same-brand/name pair to count as two different
# locations rather than the same place seen through two providers.
BRAND_HARD_NEG_MIN_M = 200.0
# Ceiling distance for a brand/name pair -- without this, "same brand,
# opposite sides of the planet" pairs are just as easy to rule out from
# address text as a mismatched name would be, which defeats the point.
# ~50km keeps pairs within a plausible metro area even when locality text
# is missing/inconsistent and the locality-match preference (see
# find_brand_hard_negative_candidates) can't be used.
BRAND_HARD_NEG_MAX_M = 50_000.0

EARTH_R_M = 6_371_008.8  # mean earth radius in meters


# --------------------------------------------------------------------------
# distance
# --------------------------------------------------------------------------

def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in meters. Vectorized -- all four args can be
    numpy arrays. Accurate to well under a meter at the 5-10m scale we care
    about here (the earlier ST_Distance_Spheroid work used a slightly more
    exact spheroid model, which matters more at the km scale than here)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------
# place index (same grouping logic as make_pairs.py's build_index)
# --------------------------------------------------------------------------

def build_index(ids_column):
    """Pass 1: group row numbers by place (Overture id), from an id column
    already read from disk (see main() -- we read it once and reuse it for
    the double-read alignment check too).

    Returns:
        order  - row numbers sorted so that rows of one place sit together
        starts - where each place begins inside `order`
        counts - how many rows each place has
    """
    ids = ids_column.combine_chunks()
    codes = ids.dictionary_encode()
    n_places = len(codes.dictionary)
    place_of_row = codes.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    del ids, codes

    order = np.argsort(place_of_row, kind="stable")
    sorted_codes = place_of_row[order]
    del place_of_row

    all_places = np.arange(n_places)
    starts = np.searchsorted(sorted_codes, all_places, side="left")
    counts = np.searchsorted(sorted_codes, all_places, side="right") - starts
    return order, starts, counts


def build_row_to_place(order, counts):
    """Inverse of build_index: row number -> place id. Used so the 'each
    place used once' rule can be enforced from a row index alone."""
    place_of_row = np.empty(len(order), dtype=np.int64)
    place_of_row[order] = np.repeat(np.arange(len(counts)), counts)
    return place_of_row


def encode_categories(source):
    """Compact integer codes for basic_category, for the easy-negative
    'different category' check. basic_category has low cardinality (a
    couple hundred distinct values across 10M rows), so dictionary-coding
    it the same way build_index() codes ids keeps this to tens of MB --
    materializing 10M individual Python strings via to_pylist() instead
    costs well over a gigabyte for no benefit (the check only ever compares
    two codes for equality, never needs the text itself). Nulls (some rows
    really do have no category) become code -1, which never equals another
    -1 in the sense we want -- see the fill_null call below, which routes
    every null to the SAME sentinel so two null-category rows don't get
    treated as "different category" by pure accident of both being -1...
    actually they SHOULD count as equal-enough-to-skip here (a genuine
    "unknown vs unknown" isn't the clean McDonald's-vs-ski-resort signal
    we want), so -1 is deliberately reused for every null.
    """
    try:
        col = pq.read_table(source, columns=["basic_category"]).column("basic_category").combine_chunks()
        codes = col.dictionary_encode().indices
        codes = pc.fill_null(codes, -1)
        return codes.to_numpy(zero_copy_only=False)
    except Exception as e:
        print(f"note: couldn't load basic_category for easy-negative filtering ({e}); skipping that check")
        return None


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

GEOM_POINT_DTYPE = np.dtype([("order", "u1"), ("type", "<u4"), ("lon", "<f8"), ("lat", "<f8")])


def build_geo(source, n_spotcheck=25):
    """Decode lon/lat for every row directly from the WKB geometry column --
    no DuckDB, no network, no external dependency at all.

    Every geometry in this dataset is a standard little-endian WKB Point,
    exactly 21 bytes (1 byte order + 4 byte type + 8 byte X + 8 byte Y),
    confirmed by scanning every row group's byte-length distribution by
    hand (all 21, zero nulls). That fixed size means the raw values buffer
    behind the column can be reinterpreted directly as a numpy structured
    array -- no per-row Python loop, no copy.

    Reading geometry via a plain pq.read_table() call keeps the exact same
    physical row order as build_index()'s read of the id column (pyarrow's
    parquet reader parallelizes decoding but always reassembles rows in
    file order) -- so no extra alignment machinery is needed here, unlike
    the old DuckDB path.

    As a safety net against any of that being subtly wrong on some other
    machine/pyarrow version, a handful of rows are independently re-decoded
    the slow-but-obviously-correct way (struct.unpack per row) and checked
    against the fast path before trusting it for the other ~10M.
    """
    import struct

    col = pq.read_table(source, columns=["geometry"]).column("geometry").combine_chunks()
    n = len(col)
    buf = col.buffers()[2]
    rec = np.frombuffer(buf, dtype=GEOM_POINT_DTYPE, count=n)

    assert np.all(rec["order"] == 1), "found a non-little-endian WKB record -- this parser assumes byte_order=1"
    assert np.all(rec["type"] == 1), "found a non-Point geometry -- this parser only handles WKB Points"

    lon = np.array(rec["lon"])
    lat = np.array(rec["lat"])

    rng = np.random.default_rng(0)
    spot = rng.choice(n, size=min(n_spotcheck, n), replace=False)
    for i in spot:
        b = col[int(i)].as_py()
        _, _, x, y = struct.unpack("<BIdd", b)
        assert x == lon[i] and y == lat[i], (
            f"fast WKB parse disagreed with a slow per-row decode at row {i} "
            f"({x},{y}) vs ({lon[i]},{lat[i]}) -- do not trust this output"
        )

    return lon, lat


def verify_alignment(source, ids, n_spotcheck=50, seed=1):
    """Spot-check that `ids` (already read once, e.g. via build_index's
    caller) lines up with a completely fresh direct read of a handful of
    rows from the file -- so a silent misalignment can never sneak a wrong
    distance onto a pair. Deliberately a spot-check, not a full second
    10M-row comparison: doubling up a full materialization of the id
    column is real memory on a constrained machine, and a handful of
    matches is already strong evidence against any row-order bug (the
    kind of bug this guards against would misalign *every* row, not just
    the unlucky ones)."""
    if isinstance(ids, pa.ChunkedArray):
        ids = ids.combine_chunks()
    n = len(ids)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=min(n_spotcheck, n), replace=False))

    pf = pq.ParquetFile(source)
    found = {}
    offset = 0
    for batch in pf.iter_batches(batch_size=200_000, columns=["id"]):
        bn = batch.num_rows
        here = idx[(idx >= offset) & (idx < offset + bn)]
        if len(here):
            vals = batch.column("id").take(pa.array(here - offset)).to_pylist()
            for j, v in zip(here, vals):
                found[int(j)] = v
        offset += bn
        if len(found) == len(idx):
            break

    expected = ids.take(pa.array(idx)).to_pylist()
    for j, exp in zip(idx, expected):
        got = found[int(j)]
        assert got == exp, (
            f"id mismatch at row {j}: expected {exp!r}, a fresh read of the file has {got!r} -- "
            f"do not trust this output"
        )


# --------------------------------------------------------------------------
# MATCH candidates: same id, close together
# --------------------------------------------------------------------------

def find_positive_candidates(order, starts, counts, lon, lat, max_m):
    """For every place, check the pairwise distances between its rows and
    keep the closest pair if it's under max_m. Places have at most a
    handful of rows (one per provider), so instead of looping over ~4.4M
    places in Python, we loop over group SIZE (2, 3, 4, 5 rows) and handle
    every place of that size at once with vectorized numpy ops."""
    n_places = len(counts)
    best_a = np.full(n_places, -1, dtype=np.int64)
    best_b = np.full(n_places, -1, dtype=np.int64)
    best_d = np.full(n_places, np.inf)

    max_k = int(counts.max()) if n_places else 0
    for k in range(2, max_k + 1):
        places_k = np.where(counts == k)[0]
        if len(places_k) == 0:
            continue
        base = starts[places_k]
        for i in range(k):
            for j in range(i + 1, k):
                a = order[base + i]
                b = order[base + j]
                d = haversine_m(lon[a], lat[a], lon[b], lat[b])
                better = d < best_d[places_k]
                idx = places_k[better]
                best_d[idx] = d[better]
                best_a[idx] = a[better]
                best_b[idx] = b[better]

    keep = best_d < max_m
    place_ids = np.where(keep)[0]
    return place_ids, best_a[keep], best_b[keep], best_d[keep]


# --------------------------------------------------------------------------
# hard-negative candidates: different id, close together (grid-hash search)
# --------------------------------------------------------------------------

def find_hard_negative_candidates(ids, lon, lat, max_m, cell_m=None, categories=None,
                                   max_candidates=300_000, max_cell_rows=4_000,
                                   max_pairs_per_cell=5, rng=None):
    """See the module docstring for how the grid search works. Returns
    (row_a, row_b, distance_m, same_category) arrays of candidate hard-
    negative pairs.

    `ids` should be a pyarrow (Chunked)Array, not a numpy array of Python
    str -- materializing all ~10M ids as individual Python string objects
    costs over a gigabyte on its own. Comparisons here only ever touch a
    small per-cell slice at a time via .take(), so the id data stays in
    its compact arrow buffer for everything except that tiny slice.

    Likewise, cell lookup uses a sorted int64 key + binary search instead
    of a {(cx,cy): slot} dict -- millions of tuple keys in a Python dict
    is another easy gigabyte; a sorted array of int64 is a fraction of that.

    max_pairs_per_cell caps how many candidates come from any one grid
    cell's 3x3 neighborhood, so a single extreme outlier (e.g. a
    company-registration address with 100+ unrelated businesses at the
    same coordinates -- these exist: the real dataset's densest 10m cell
    has 114 different places in it, vs. a median close cluster of just 2-4)
    can't flood the candidate pool and make the final hard-negative set
    repetitive. Real measurement on the full dataset: only ~10-20 cells
    are that extreme; the vast majority (1.3M+) are ordinary 2-4-point
    clusters, so this cap costs almost nothing there and only reins in
    the genuine outliers.

    `categories` (optional, from encode_categories()) is what makes a geo
    candidate actually hard for a model that never sees coordinates: two
    places that are merely close but visibly different on paper (a hotel
    next to a restaurant) aren't a hard negative once distance is hidden --
    they're an easy one that happens to be nearby. Within each cell's
    matches, same-category pairs (two competing dentists in the same
    plaza) are preferred up to max_pairs_per_cell; a different-category
    pair only fills a leftover slot if the cell doesn't have enough
    same-category matches to fill the cap on its own -- a soft preference,
    not a hard filter, so the search doesn't come up short in cells that
    happen to be category-diverse. If categories is None (unavailable),
    every candidate is returned as same_category=False, i.e. no ranking.
    """
    if cell_m is None:
        cell_m = max(max_m, 1.0)
    if rng is None:
        rng = np.random.default_rng(0)
    if isinstance(ids, pa.ChunkedArray):
        ids = ids.combine_chunks()

    lat_rad = np.radians(lat)
    x_m = lon * 111_320.0 * np.cos(lat_rad)
    y_m = lat * 110_540.0
    cx = np.floor(x_m / cell_m).astype(np.int64)
    cy = np.floor(y_m / cell_m).astype(np.int64)

    n = len(cx)
    order_by_cell = np.lexsort((cy, cx))
    cx_sorted, cy_sorted = cx[order_by_cell], cy[order_by_cell]
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = (cx_sorted[1:] != cx_sorted[:-1]) | (cy_sorted[1:] != cy_sorted[:-1])
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], n)
    counts = ends - starts

    # sorted-key binary search in place of a {(cx,cy): slot} dict
    y_span = int(max(abs(int(cy.min())), abs(int(cy.max())))) + 1
    key_mult = 2 * y_span + 1
    slot_key = cx_sorted[starts].astype(np.int64) * key_mult + (cy_sorted[starts].astype(np.int64) + y_span)

    def lookup_slot(qcx, qcy):
        k = qcx * key_mult + (qcy + y_span)
        j = np.searchsorted(slot_key, k)
        if j < len(slot_key) and slot_key[j] == k:
            return int(j)
        return None

    cell_of_slot_cx = cx_sorted[starts]
    cell_of_slot_cy = cy_sorted[starts]
    dense_order = np.argsort(-counts)
    NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]

    cand_a, cand_b, cand_d, cand_same_cat = [], [], [], []
    found = 0
    for slot in dense_order:
        if found >= max_candidates:
            break
        bcx, bcy = int(cell_of_slot_cx[slot]), int(cell_of_slot_cy[slot])

        # A cell with only 1 point of its own can still hold half of a valid
        # pair (its lone point close to a lone point in a neighboring cell).
        # So don't skip on the home cell's own count -- skip only if the
        # WHOLE 3x3 neighborhood (which always includes home, via (0,0))
        # has fewer than 2 points total, i.e. there's truly nothing nearby.
        neighbor_slots = [lookup_slot(bcx + dx, bcy + dy) for dx, dy in NEIGH]
        neighbor_slots = [s for s in neighbor_slots if s is not None]
        if sum(counts[s] for s in neighbor_slots) < 2:
            continue

        home_rows = order_by_cell[starts[slot]:ends[slot]]
        if len(home_rows) > max_cell_rows:
            home_rows = rng.choice(home_rows, size=max_cell_rows, replace=False)

        pool_parts = [order_by_cell[starts[s]:ends[s]] for s in neighbor_slots]
        pool = np.concatenate(pool_parts)
        if len(pool) > max_cell_rows:
            pool = rng.choice(pool, size=max_cell_rows, replace=False)

        a_rep = np.repeat(home_rows, len(pool))
        b_tile = np.tile(pool, len(home_rows))
        keep = a_rep < b_tile  # unordered-pair dedup within this cell's view
        a_rep, b_tile = a_rep[keep], b_tile[keep]
        if len(a_rep) == 0:
            continue

        id_a = ids.take(pa.array(a_rep))
        id_b = ids.take(pa.array(b_tile))
        diff_id = pc.not_equal(id_a, id_b).to_numpy(zero_copy_only=False)
        a_rep, b_tile = a_rep[diff_id], b_tile[diff_id]
        if len(a_rep) == 0:
            continue

        d = haversine_m(lon[a_rep], lat[a_rep], lon[b_tile], lat[b_tile])
        close = d < max_m
        if not close.any():
            continue
        idx = np.flatnonzero(close)

        if categories is not None:
            same_cat = categories[a_rep[idx]] == categories[b_tile[idx]]
            same_cat &= categories[a_rep[idx]] != -1  # unknown-vs-unknown isn't a real match
        else:
            same_cat = np.zeros(len(idx), dtype=bool)

        # fill the per-cell cap with same-category pairs first, then use
        # any leftover room for different-category ones -- a preference,
        # not a requirement, so a category-diverse cell still contributes
        idx_pref, idx_fallback = idx[same_cat], idx[~same_cat]
        chosen_parts, chosen_flags = [], []
        if len(idx_pref):
            take = idx_pref if len(idx_pref) <= max_pairs_per_cell else \
                rng.choice(idx_pref, size=max_pairs_per_cell, replace=False)
            chosen_parts.append(take)
            chosen_flags.append(np.ones(len(take), dtype=bool))
        room = max_pairs_per_cell - sum(len(c) for c in chosen_parts)
        if room > 0 and len(idx_fallback):
            take = idx_fallback if len(idx_fallback) <= room else \
                rng.choice(idx_fallback, size=room, replace=False)
            chosen_parts.append(take)
            chosen_flags.append(np.zeros(len(take), dtype=bool))
        if not chosen_parts:
            continue
        chosen = np.concatenate(chosen_parts)

        cand_a.append(a_rep[chosen])
        cand_b.append(b_tile[chosen])
        cand_d.append(d[chosen])
        cand_same_cat.append(np.concatenate(chosen_flags))
        found += len(chosen)

    if not cand_a:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=float), np.array([], dtype=bool))

    a = np.concatenate(cand_a)
    b = np.concatenate(cand_b)
    d = np.concatenate(cand_d)
    same_cat_out = np.concatenate(cand_same_cat)
    pair_key = a.astype(np.int64) * n + b.astype(np.int64)
    _, uniq_idx = np.unique(pair_key, return_index=True)
    return a[uniq_idx], b[uniq_idx], d[uniq_idx], same_cat_out[uniq_idx]


# --------------------------------------------------------------------------
# hard-negative candidates, source 2: same brand/name, different location
# --------------------------------------------------------------------------

def extract_brand_or_name_key(source):
    """Build one normalized grouping key per row: the place's brand name if
    it has one, else its own primary name. Returns a pyarrow string array
    aligned to the file's physical row order (same guarantee as build_geo).

    Overture's `brand` field is a nested struct (brand.names.primary) --
    the strongest same-chain signal, since every location of a chain
    carries the same brand.names.primary even when each location's own
    `names.primary` varies (store numbers, per-provider formatting). Most
    rows have no brand at all (most places aren't chains), so those fall
    back to their own names.primary -- two unrelated businesses that just
    happen to share a name are still a legitimate hard negative.

    Deliberately defensive: struct field access (pc.struct_field) can
    raise if a column turns out not to be the nested shape we expect (this
    sandbox has never seen the real file's exact arrow schema, only the
    documented Overture Places one), so every extraction step is wrapped
    and falls back to treating the column as an already-flat string --
    which also happens to be what our synthetic smoke-test fixture uses,
    so that fallback path is exercised by the smoke test even though the
    real file should never need it.
    """
    def struct_text(col, path):
        arr = col
        for key in path:
            arr = pc.struct_field(arr, key)
        return pc.cast(arr, pa.string())

    brand_key = None
    try:
        brand_col = pq.read_table(source, columns=["brand"]).column("brand").combine_chunks()
        brand_key = struct_text(brand_col, ["names", "primary"])
    except Exception as e:
        print(f"note: couldn't read brand.names.primary as a nested struct ({e}); "
              f"brand/name hard negatives will use place names only")

    names_col = pq.read_table(source, columns=["names"]).column("names").combine_chunks()
    try:
        name_key = struct_text(names_col, ["primary"])
    except Exception as e:
        print(f"note: couldn't read names.primary as a nested struct ({e}); "
              f"treating the names column as already-flat text")
        name_key = pc.cast(names_col, pa.string())

    if brand_key is not None:
        # pc.or_ (not _kleene) propagates a null from EITHER side even when
        # the other side is definitively True, which would leak nulls into
        # empty_brand for every row with no brand and make key() null too
        # instead of falling back to name_key -- pc.equal(null, "") is null,
        # so plain or_ would turn "definitely empty" into "unknown". Kleene
        # logic is what we actually want: is_null(brand_key)=True already
        # settles it regardless of what the (possibly-null) equality check
        # says.
        empty_brand = pc.or_kleene(pc.is_null(brand_key),
                                    pc.equal(pc.utf8_trim_whitespace(brand_key), ""))
        key = pc.if_else(empty_brand, name_key, brand_key)
    else:
        key = name_key

    key = pc.cast(key, pa.string())
    key = pc.utf8_trim_whitespace(pc.utf8_lower(pc.fill_null(key, "")))
    return key


def extract_locality(source):
    """Normalized city/locality text per row -- lets the brand/name search
    prefer "same city, different street" pairs over "different continent"
    ones. A different country/city in the address text is just as easy a
    tell as a mismatched name would be, so an unbounded-distance brand
    pair doesn't actually test what a hard negative is supposed to test;
    same-locality pairs force the model to read the finer-grained part of
    the address (street/unit) to disambiguate, which is the real target.

    Overture's `addresses` field is list<struct<..., locality, ...>> -- a
    place can have more than one address, so this takes the first one's
    locality. Same defensive-fallback shape as extract_brand_or_name_key:
    if the column isn't the nested shape expected, every row comes back
    with locality "" (never treated as matching another "" -- see
    same_locality in find_brand_hard_negative_candidates) rather than
    crashing or silently pairing every unknown-locality row together.
    """
    try:
        col = pq.read_table(source, columns=["addresses"]).column("addresses").combine_chunks()
        first = pc.list_element(col, 0)
        locality = pc.cast(pc.struct_field(first, "locality"), pa.string())
    except Exception as e:
        print(f"note: couldn't read addresses[0].locality as a nested list<struct> ({e}); "
              f"brand/name hard negatives will fall back to distance-only bounding")
        n = pq.ParquetFile(source).metadata.num_rows
        return pa.array([""] * n, type=pa.string())
    return pc.utf8_trim_whitespace(pc.utf8_lower(pc.fill_null(locality, "")))


def find_brand_hard_negative_candidates(ids, name_keys, lon, lat, min_m, max_m=BRAND_HARD_NEG_MAX_M,
                                         locality_keys=None,
                                         max_candidates=150_000, max_group_rows=500,
                                         max_pairs_per_group=5, rng=None):
    """Second hard-negative source, orthogonal to the grid search above.
    That search finds pairs that are close in space but usually easy to
    tell apart by name/category text -- hard only because of coordinates
    the model never sees. This one finds the opposite failure mode: pairs
    whose brand/name text is IDENTICAL (so a model pattern-matching on
    name alone would call them MATCH on sight) but which are different
    real-world locations -- the "which Starbucks?" case. Only the
    address/locality text can tell them apart, which is exactly what an
    entity-matching model is supposed to use.

    `name_keys` is the normalized per-row key from extract_brand_or_name_key
    (rows with an empty key are skipped -- nothing to group on). `ids`
    distinguishes the same place seen through multiple providers (a MATCH
    candidate, not a hard negative) from genuinely different places that
    happen to share a name/brand.

    min_m is a floor: two same-name rows only a few meters apart are the
    same physical place (or a near-duplicate), not two different chain
    locations, so they're excluded rather than mislabeled as a hard
    negative. max_m is a ceiling (default BRAND_HARD_NEG_MAX_M, ~50km) --
    without one, "same brand, opposite sides of the planet" pairs are just
    as easy to rule out from address text as a mismatched name, which
    defeats the point of this source.

    `locality_keys` (optional, from extract_locality()) sharpens that
    further: within min_m..max_m, pairs that ALSO share a locality (same
    city, different street) are preferred over ones that don't, up to
    max_pairs_per_group -- a different-locality pair only fills a leftover
    slot if a group doesn't have enough same-locality matches on its own.
    Soft preference again, not a hard filter, same reasoning as the
    category preference in find_hard_negative_candidates.

    max_group_rows/max_pairs_per_group guard against mega-chains (a name
    like "Starbucks" can have thousands of rows) blowing up the
    O(group_size^2) pairwise search inside one group and keep the final
    pool from being dominated by a single chain -- same purpose as
    max_pairs_per_cell in the geometric search above.

    Returns (row_a, row_b, distance_m, same_locality) arrays -- same shape
    as find_hard_negative_candidates's (row_a, row_b, distance_m,
    same_category).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if isinstance(ids, pa.ChunkedArray):
        ids = ids.combine_chunks()
    if isinstance(name_keys, pa.ChunkedArray):
        name_keys = name_keys.combine_chunks()

    if isinstance(locality_keys, pa.ChunkedArray):
        locality_keys = locality_keys.combine_chunks()

    nonempty = pc.and_(pc.is_valid(name_keys), pc.not_equal(name_keys, ""))
    nonempty = nonempty.to_numpy(zero_copy_only=False)
    rows = np.flatnonzero(nonempty)
    if len(rows) == 0:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=float), np.array([], dtype=bool))

    keys_sub = name_keys.take(pa.array(rows))
    codes = keys_sub.dictionary_encode()
    n_groups = len(codes.dictionary)
    group_of = codes.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    del codes

    order_local = np.argsort(group_of, kind="stable")
    sorted_groups = group_of[order_local]
    all_groups = np.arange(n_groups)
    g_starts = np.searchsorted(sorted_groups, all_groups, side="left")
    g_counts = np.searchsorted(sorted_groups, all_groups, side="right") - g_starts
    rows_sorted = rows[order_local]  # back to original (file) row numbers

    dense_order = np.argsort(-g_counts)
    cand_a, cand_b, cand_d, cand_same_loc = [], [], [], []
    found = 0
    for gi in dense_order:
        if found >= max_candidates:
            break
        cnt = int(g_counts[gi])
        if cnt < 2:
            break  # dense_order is descending by count, so every later group is smaller too

        grp_rows = rows_sorted[g_starts[gi]: g_starts[gi] + cnt]
        if len(grp_rows) > max_group_rows:
            grp_rows = rng.choice(grp_rows, size=max_group_rows, replace=False)

        m = len(grp_rows)
        ai, bi = np.triu_indices(m, k=1)
        a_rows, b_rows = grp_rows[ai], grp_rows[bi]

        id_a = ids.take(pa.array(a_rows))
        id_b = ids.take(pa.array(b_rows))
        diff_id = pc.not_equal(id_a, id_b).to_numpy(zero_copy_only=False)
        a_rows, b_rows = a_rows[diff_id], b_rows[diff_id]
        if len(a_rows) == 0:
            continue

        d = haversine_m(lon[a_rows], lat[a_rows], lon[b_rows], lat[b_rows])
        ok = d >= min_m
        if max_m is not None:
            ok &= d <= max_m
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)

        if locality_keys is not None:
            loc_a = locality_keys.take(pa.array(a_rows[idx]))
            loc_b = locality_keys.take(pa.array(b_rows[idx]))
            # locality_keys is already null-filled to "" by extract_locality,
            # so a plain (non-Kleene) equal/and_ is safe here -- no nulls to
            # propagate, unlike the brand/name fallback in
            # extract_brand_or_name_key which does need the Kleene version.
            same_loc = pc.and_(pc.equal(loc_a, loc_b),
                                pc.not_equal(loc_a, "")).to_numpy(zero_copy_only=False)
        else:
            same_loc = np.zeros(len(idx), dtype=bool)

        # same preference-then-fallback pattern as the geo search: fill the
        # per-group cap with same-locality pairs first, spill into
        # different-locality only if the group doesn't have enough
        idx_pref, idx_fallback = idx[same_loc], idx[~same_loc]
        chosen_parts, chosen_flags = [], []
        if len(idx_pref):
            take = idx_pref if len(idx_pref) <= max_pairs_per_group else \
                rng.choice(idx_pref, size=max_pairs_per_group, replace=False)
            chosen_parts.append(take)
            chosen_flags.append(np.ones(len(take), dtype=bool))
        room = max_pairs_per_group - sum(len(c) for c in chosen_parts)
        if room > 0 and len(idx_fallback):
            take = idx_fallback if len(idx_fallback) <= room else \
                rng.choice(idx_fallback, size=room, replace=False)
            chosen_parts.append(take)
            chosen_flags.append(np.zeros(len(take), dtype=bool))
        if not chosen_parts:
            continue
        chosen = np.concatenate(chosen_parts)

        cand_a.append(a_rows[chosen])
        cand_b.append(b_rows[chosen])
        cand_d.append(d[chosen])
        cand_same_loc.append(np.concatenate(chosen_flags))
        found += len(chosen)

    if not cand_a:
        return (np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                np.array([], dtype=float), np.array([], dtype=bool))

    a = np.concatenate(cand_a)
    b = np.concatenate(cand_b)
    d = np.concatenate(cand_d)
    same_loc_out = np.concatenate(cand_same_loc)
    pair_key = a.astype(np.int64) * len(ids) + b.astype(np.int64)
    _, uniq_idx = np.unique(pair_key, return_index=True)
    return a[uniq_idx], b[uniq_idx], d[uniq_idx], same_loc_out[uniq_idx]


# --------------------------------------------------------------------------
# easy negatives: far apart, ideally different category
# --------------------------------------------------------------------------

def sample_easy_negative(rng, counts, order, starts, lon, lat, categories,
                          used_places, min_m, max_tries=20):
    """Pick 2 different, not-yet-used places, far enough apart. Tries first
    to also land on a different top-level category (the 'McDonald's vs. ski
    resort' case); falls back to distance-only if that's taking too long."""
    n_places = len(counts)

    def one_try(require_diff_category):
        p1, p2 = rng.choice(n_places, size=2, replace=False)
        if p1 in used_places or p2 in used_places:
            return None
        r1 = order[starts[p1] + int(rng.random() * counts[p1])]
        r2 = order[starts[p2] + int(rng.random() * counts[p2])]
        d = haversine_m(lon[r1], lat[r1], lon[r2], lat[r2])
        if d < min_m:
            return None
        if require_diff_category and categories is not None and categories[r1] == categories[r2]:
            return None
        return p1, p2, r1, r2, d

    for _ in range(max_tries):
        got = one_try(require_diff_category=True)
        if got:
            return got
    for _ in range(max_tries):
        got = one_try(require_diff_category=False)
        if got:
            return got
    raise RuntimeError("could not find an easy-negative pair after many tries -- "
                        "try lowering --easy-neg-min-m")


# --------------------------------------------------------------------------
# assemble one size's worth of pairs
# --------------------------------------------------------------------------

def pick_pairs_geo(index, lon, lat, categories, pos_cand, hardneg_geo_cand, hardneg_brand_cand,
                    n_pairs, seed, hard_neg_fraction, brand_hard_neg_fraction, easy_neg_min_m):
    order, starts, counts = index
    rng = np.random.default_rng(seed)
    row_to_place = build_row_to_place(order, counts)

    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos
    n_hard = round(n_neg * hard_neg_fraction)
    n_easy = n_neg - n_hard
    n_hard_brand = round(n_hard * brand_hard_neg_fraction)
    n_hard_geo = n_hard - n_hard_brand

    pos_places, pos_a, pos_b, pos_d = pos_cand
    hn_geo_a, hn_geo_b, hn_geo_d, hn_geo_pref = hardneg_geo_cand
    hn_brand_a, hn_brand_b, hn_brand_d, hn_brand_pref = hardneg_brand_cand

    used_places = set()

    def select_hard_negatives(cand_a, cand_b, cand_d, cand_pref, k, source_label, retry_hint):
        """Pick k distinct-place pairs from one hard-negative candidate
        pool, marking both places of each chosen pair used. Shared by the
        geo and brand pools below so the two sources are drawn the same
        way (and can't collide with each other or with MATCH places,
        since used_places is threaded through both calls).

        cand_pref marks candidates that also passed the metadata-
        similarity preference (same category for geo, same locality for
        brand) -- those are drawn first (each tier independently shuffled),
        so the metadata-plausible pairs get first claim on scarce places
        and only spill into the plain geometric/distance pool if the
        preferred tier runs out. Returns how many of the k selected came
        from the preferred tier, for the stats file."""
        order_pref = rng.permutation(np.flatnonzero(cand_pref))
        order_fallback = rng.permutation(np.flatnonzero(~cand_pref))
        idx_order = np.concatenate([order_pref, order_fallback])
        sel = []
        for i in idx_order:
            a, b = cand_a[i], cand_b[i]
            pa_, pb_ = row_to_place[a], row_to_place[b]
            if pa_ in used_places or pb_ in used_places:
                continue
            used_places.add(pa_)
            used_places.add(pb_)
            sel.append(i)
            if len(sel) == k:
                break
        if len(sel) < k:
            raise RuntimeError(
                f"only {len(sel)} {source_label} hard-negative candidates available, need {k}. "
                f"{retry_hint}"
            )
        sel = np.array(sel, dtype=np.int64)
        n_pref_used = int(cand_pref[sel].sum())
        return cand_a[sel], cand_b[sel], cand_d[sel], n_pref_used

    # --- MATCH: n_pos distinct places from the verified <match-max-m pool ---
    idx_order = rng.permutation(len(pos_places))
    sel = []
    for i in idx_order:
        p = pos_places[i]
        if p in used_places:
            continue
        used_places.add(p)
        sel.append(i)
        if len(sel) == n_pos:
            break
    if len(sel) < n_pos:
        raise RuntimeError(
            f"only {len(sel)} verified MATCH places available, need {n_pos}. "
            f"Try a slightly larger --match-max-m."
        )
    sel = np.array(sel)
    left_pos, right_pos, dist_pos = pos_a[sel], pos_b[sel], pos_d[sel]

    # --- hard negatives: n_hard_geo from the geometric pool, n_hard_brand from
    # the brand/name pool -- drawn against the SAME used_places set, so the
    # two sources never collide with each other or with MATCH places ---
    left_geo, right_geo, dist_geo, n_geo_same_cat = select_hard_negatives(
        hn_geo_a, hn_geo_b, hn_geo_d, hn_geo_pref, n_hard_geo, "geo",
        "Re-run with --hard-neg-candidates set higher.")
    left_brand, right_brand, dist_brand, n_brand_same_loc = select_hard_negatives(
        hn_brand_a, hn_brand_b, hn_brand_d, hn_brand_pref, n_hard_brand, "brand",
        "Re-run with --brand-hard-neg-candidates set higher, or lower "
        "--brand-hard-neg-fraction.")

    left_hn = np.concatenate([left_geo, left_brand])
    right_hn = np.concatenate([right_geo, right_brand])
    dist_hn = np.concatenate([dist_geo, dist_brand])
    hard_source_hn = np.array(["geo"] * n_hard_geo + ["brand"] * n_hard_brand)

    # --- easy negatives ---
    left_easy, right_easy, dist_easy = [], [], []
    for _ in range(n_easy):
        p1, p2, r1, r2, d = sample_easy_negative(
            rng, counts, order, starts, lon, lat, categories, used_places, easy_neg_min_m)
        used_places.add(p1)
        used_places.add(p2)
        left_easy.append(r1)
        right_easy.append(r2)
        dist_easy.append(d)
    left_easy = np.array(left_easy, dtype=np.int64)
    right_easy = np.array(right_easy, dtype=np.int64)
    dist_easy = np.array(dist_easy, dtype=float)

    left = np.concatenate([left_pos, left_hn, left_easy])
    right = np.concatenate([right_pos, right_hn, right_easy])
    dist = np.concatenate([dist_pos, dist_hn, dist_easy])
    label = np.concatenate([np.ones(n_pos, np.int8), np.zeros(n_hard + n_easy, np.int8)])
    pair_type = np.array(["match"] * n_pos + ["hard_negative"] * n_hard + ["easy_negative"] * n_easy)
    hard_source = np.concatenate([np.array([""] * n_pos), hard_source_hn, np.array([""] * n_easy)])

    # coin flip which row lands on the left -- same convention as make_pairs.py
    flip = rng.random(n_pairs) < 0.5
    L = np.where(flip, right, left)
    R = np.where(flip, left, right)

    mix = rng.permutation(n_pairs)
    return (L[mix], R[mix], label[mix], pair_type[mix], dist[mix], hard_source[mix],
            n_geo_same_cat, n_brand_same_loc)


# --------------------------------------------------------------------------
# row fetch (identical to make_pairs.py's fetch_rows)
# --------------------------------------------------------------------------

def fetch_rows(source, wanted, batch_size=50_000):
    """Pass 2: stream the file in small batches and keep only what we need.
    `wanted` must be sorted. The result rows come back in that order.

    Streams in batch_size-row chunks rather than whole row groups (which
    can be 800K+ rows in this file) so peak memory stays bounded -- nested
    columns like names/addresses get expensive to materialize per row, and
    a full row group's worth of them can blow past a constrained machine's
    memory even though the final selection is tiny."""
    pf = pq.ParquetFile(source)
    parts = []
    offset = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        n = batch.num_rows
        here = wanted[(wanted >= offset) & (wanted < offset + n)]
        if len(here):
            parts.append(pa.Table.from_batches([batch]).take(pa.array(here - offset)))
        offset += n
    return pa.concat_tables(parts)


# --------------------------------------------------------------------------
# build + write one size
# --------------------------------------------------------------------------

def make(source, size, seed, out_dir, index, lon, lat, categories,
         pos_cand, hardneg_geo_cand, hardneg_brand_cand, match_max_m, hard_neg_max_m,
         easy_neg_min_m, hard_neg_fraction, brand_hard_neg_fraction,
         brand_hard_neg_min_m, brand_hard_neg_max_m):
    n_pairs = SIZES[size]
    started = time.time()

    left, right, label, pair_type, dist, hard_source, n_geo_same_cat, n_brand_same_loc = pick_pairs_geo(
        index, lon, lat, categories, pos_cand, hardneg_geo_cand, hardneg_brand_cand,
        n_pairs, seed + SEED_STEP[size], hard_neg_fraction, brand_hard_neg_fraction, easy_neg_min_m)

    wanted = np.unique(np.concatenate([left, right]))
    rows = fetch_rows(source, wanted)
    left_rows = rows.take(pa.array(np.searchsorted(wanted, left)))
    right_rows = rows.take(pa.array(np.searchsorted(wanted, right)))

    cols = {
        "pair_id": pa.array(np.arange(n_pairs)),
        "label": pa.array(label),
        "pair_type": pa.array(pair_type),
        "hard_neg_source": pa.array(hard_source),
        "distance_m": pa.array(dist),
    }
    for name in rows.schema.names:
        cols[f"left_{name}"] = left_rows.column(name)
        cols[f"right_{name}"] = right_rows.column(name)
    table = pa.table(cols)

    # --- check our own work before writing anything ---
    lid = np.array(table.column("left_id").to_pylist())
    rid = np.array(table.column("right_id").to_pylist())
    same = lid == rid

    n_pos = n_pairs // 2
    n_neg = n_pairs - n_pos
    n_hard = round(n_neg * hard_neg_fraction)
    n_easy = n_neg - n_hard
    n_hard_brand = round(n_hard * brand_hard_neg_fraction)
    n_hard_geo = n_hard - n_hard_brand
    hard_mask = pair_type == "hard_negative"
    easy_mask = pair_type == "easy_negative"
    geo_hard_mask = hard_mask & (hard_source == "geo")
    brand_hard_mask = hard_mask & (hard_source == "brand")

    assert table.num_rows == n_pairs, "wrong number of pairs"
    assert label.sum() == n_pos, "labels are not half and half"
    assert len(wanted) == 2 * n_pairs, "a row got used more than once"
    assert same[label == 1].all(), "a MATCH pair has two different ids"
    assert not same[label == 0].any(), "a NO_MATCH pair has the same id twice"
    assert (dist[label == 1] < match_max_m).all(), "a MATCH pair is >= match-max-m apart"
    assert hard_mask.sum() == n_hard, "wrong hard-negative count"
    assert easy_mask.sum() == n_easy, "wrong easy-negative count"
    assert geo_hard_mask.sum() == n_hard_geo, "wrong geo hard-negative count"
    assert brand_hard_mask.sum() == n_hard_brand, "wrong brand hard-negative count"
    assert (hard_source[~hard_mask] == "").all(), "a non-hard-negative pair has a hard_neg_source set"
    assert (dist[geo_hard_mask] < hard_neg_max_m).all(), "a geo hard negative is >= hard-neg-max-m apart"
    assert (dist[brand_hard_mask] >= brand_hard_neg_min_m).all(), \
        "a brand hard negative is closer than brand-hard-neg-min-m (looks like the same place)"
    if brand_hard_neg_max_m is not None:
        assert (dist[brand_hard_mask] <= brand_hard_neg_max_m).all(), \
            "a brand hard negative is farther than brand-hard-neg-max-m"
    assert (dist[easy_mask] >= easy_neg_min_m).all(), "an easy negative is closer than easy-neg-min-m"

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{size}_pairs.parquet"
    pq.write_table(table, path)

    stats = {
        "size": size,
        "pairs": n_pairs,
        "match_pairs": int(n_pos),
        "hard_negative_pairs": int(n_hard),
        "hard_negative_geo_pairs": int(n_hard_geo),
        "hard_negative_geo_same_category_pairs": int(n_geo_same_cat),
        "hard_negative_brand_pairs": int(n_hard_brand),
        "hard_negative_brand_same_locality_pairs": int(n_brand_same_loc),
        "easy_negative_pairs": int(n_easy),
        "match_max_m": match_max_m,
        "hard_neg_max_m": hard_neg_max_m,
        "brand_hard_neg_fraction": brand_hard_neg_fraction,
        "brand_hard_neg_min_m": brand_hard_neg_min_m,
        "brand_hard_neg_max_m": brand_hard_neg_max_m,
        "easy_neg_min_m": easy_neg_min_m,
        "rows_used": len(wanted),
        "seed": seed + SEED_STEP[size],
        "seconds": round(time.time() - started, 1),
    }
    (out_dir / f"{size}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"{size}: {n_pairs} pairs ({n_pos} match / {n_hard} hard-neg "
          f"[{n_hard_geo} geo ({n_geo_same_cat} same-category) + "
          f"{n_hard_brand} brand ({n_brand_same_loc} same-locality)] / "
          f"{n_easy} easy-neg) -> {path}  ({stats['seconds']}s)")


# --------------------------------------------------------------------------

def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="micro", choices=[*SIZES, "all"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", type=Path, default=here.parent / "terraforma_samples_10M.parquet")
    p.add_argument("--out", type=Path, default=here / "output_geo")
    p.add_argument("--match-max-m", type=float, default=MATCH_MAX_M)
    p.add_argument("--hard-neg-max-m", type=float, default=HARD_NEG_MAX_M)
    p.add_argument("--easy-neg-min-m", type=float, default=EASY_NEG_MIN_M)
    p.add_argument("--hard-neg-fraction", type=float, default=HARD_NEG_FRACTION)
    p.add_argument("--hard-neg-candidates", type=int, default=300_000,
                    help="how many geometric hard-negative candidate pairs to search for up front")
    p.add_argument("--max-pairs-per-cell", type=int, default=5,
                    help="cap on candidates from any single dense cluster (guards against "
                         "outliers like a company-registration address with 100+ businesses)")
    p.add_argument("--brand-hard-neg-fraction", type=float, default=BRAND_HARD_NEG_FRACTION,
                    help="of the hard-negative budget, what fraction comes from the brand/name "
                         "search vs. the geometric search (default: split evenly)")
    p.add_argument("--brand-hard-neg-min-m", type=float, default=BRAND_HARD_NEG_MIN_M,
                    help="minimum distance for a same-brand/name pair to count as two different "
                         "locations rather than the same place seen through two providers")
    p.add_argument("--brand-hard-neg-max-m", type=float, default=BRAND_HARD_NEG_MAX_M,
                    help="maximum distance for brand/name hard negatives -- without a "
                         "ceiling, same-brand pairs on opposite sides of the planet are just "
                         "as easy to rule out from address text as a mismatched name")
    p.add_argument("--brand-hard-neg-candidates", type=int, default=150_000,
                    help="how many brand/name hard-negative candidate pairs to search for up front")
    p.add_argument("--max-rows-per-brand-group", type=int, default=500,
                    help="cap on how many rows of one mega-chain (e.g. every McDonald's) are "
                         "considered at once")
    p.add_argument("--max-pairs-per-brand-group", type=int, default=5,
                    help="cap on candidate pairs kept from any single brand/name group")
    p.add_argument("--skip-alignment-check", action="store_true",
                    help="skip the double-read id cross-check (saves time once you trust it)")
    a = p.parse_args()

    print("reading ids...")
    ids_column = pq.read_table(a.source, columns=["id"]).column("id")
    index = build_index(ids_column)
    order, starts, counts = index
    print(f"found {len(counts):,} places")

    print("decoding geometry (lon/lat) from WKB...")
    lon, lat = build_geo(a.source)

    if not a.skip_alignment_check:
        print("spot-checking id alignment against a fresh read...")
        verify_alignment(a.source, ids_column)

    categories = encode_categories(a.source)

    print(f"finding MATCH candidates (same id, <{a.match_max_m}m apart)...")
    pos_cand = find_positive_candidates(order, starts, counts, lon, lat, a.match_max_m)
    print(f"  {len(pos_cand[0]):,} verified MATCH places available")

    print(f"finding geo hard-negative candidates (different id, <{a.hard_neg_max_m}m apart, "
          f"preferring same basic_category)...")
    hardneg_geo_cand = find_hard_negative_candidates(
        ids_column, lon, lat, max_m=a.hard_neg_max_m, categories=categories,
        max_candidates=a.hard_neg_candidates, max_pairs_per_cell=a.max_pairs_per_cell)
    n_geo_pref = int(hardneg_geo_cand[3].sum())
    print(f"  {len(hardneg_geo_cand[0]):,} geo hard-negative candidate pairs available "
          f"({n_geo_pref:,} same-category)")

    print("extracting brand/name keys and localities for the brand hard-negative search...")
    name_key = extract_brand_or_name_key(a.source)
    locality_key = extract_locality(a.source)
    print(f"finding brand hard-negative candidates (same brand/name, "
          f">={a.brand_hard_neg_min_m}m and <={a.brand_hard_neg_max_m}m apart, "
          f"preferring same locality, capped at {a.brand_hard_neg_candidates:,})...")
    hardneg_brand_cand = find_brand_hard_negative_candidates(
        ids_column, name_key, lon, lat, min_m=a.brand_hard_neg_min_m, max_m=a.brand_hard_neg_max_m,
        locality_keys=locality_key, max_candidates=a.brand_hard_neg_candidates,
        max_group_rows=a.max_rows_per_brand_group, max_pairs_per_group=a.max_pairs_per_brand_group)
    n_brand_pref = int(hardneg_brand_cand[3].sum())
    print(f"  {len(hardneg_brand_cand[0]):,} brand hard-negative candidate pairs available "
          f"({n_brand_pref:,} same-locality)")

    for size in (SIZES if a.size == "all" else [a.size]):
        make(a.source, size, a.seed, a.out, index, lon, lat, categories,
             pos_cand, hardneg_geo_cand, hardneg_brand_cand, a.match_max_m, a.hard_neg_max_m,
             a.easy_neg_min_m, a.hard_neg_fraction, a.brand_hard_neg_fraction,
             a.brand_hard_neg_min_m, a.brand_hard_neg_max_m)


if __name__ == "__main__":
    main()
