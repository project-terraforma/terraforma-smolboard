"""Write a FAKE results CSV in the same format as run_benchmark.py's checkpoint.

Only for testing build_leaderboard.py and the site before real results arrive.
Every fake model name starts with "FAKE-" so it can't be mistaken for a real run.

Usage: python site_tools/make_fake_results.py datasets/geo/micro_pairs.parquet fake_results.csv

The fake models are chosen so we know the right answers ahead of time:
  FAKE-perfect        always right                 -> score 100
  FAKE-always-match   answers MATCH to everything  -> score 50
  FAKE-always-no      answers NOT_MATCH always     -> score 50
  FAKE-coin-flip      random answers               -> score about 50
  FAKE-weak-on-hard   right except on hard negatives, where it says MATCH
                      -> score 65 (50 + 15 + 0)
"""
import csv
import random
import sys

import pyarrow.parquet as pq

COLUMNS = ["model", "prompt", "example_id", "pair_type", "hard_neg_source",
           "true_label", "raw_output", "prediction", "status", "correct"]


def answer(model, truth, pair_type, rng):
    if model == "FAKE-perfect":
        return truth
    if model == "FAKE-always-match":
        return "MATCH"
    if model == "FAKE-always-no":
        return "NOT_MATCH"
    if model == "FAKE-coin-flip":
        return rng.choice(["MATCH", "NOT_MATCH"])
    if model == "FAKE-weak-on-hard":
        return "MATCH" if pair_type == "hard_negative" else truth
    raise ValueError(model)


def main():
    pairs_path, out_path = sys.argv[1], sys.argv[2]
    rows = pq.read_table(pairs_path, columns=["pair_id", "label", "pair_type", "hard_neg_source"]).to_pylist()
    rng = random.Random(0)
    models = ["FAKE-perfect", "FAKE-always-match", "FAKE-always-no", "FAKE-coin-flip", "FAKE-weak-on-hard"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for model in models:
            for r in rows:
                truth = "MATCH" if r["label"] == 1 else "NOT_MATCH"
                pred = answer(model, truth, r["pair_type"], rng)
                w.writerow({
                    "model": model, "prompt": "text", "example_id": r["pair_id"],
                    "pair_type": r["pair_type"], "hard_neg_source": r["hard_neg_source"] or "",
                    "true_label": truth, "raw_output": pred, "prediction": pred,
                    "status": "ok", "correct": pred == truth,
                })
    print(f"wrote {len(models) * len(rows)} fake rows to {out_path}")


if __name__ == "__main__":
    main()
