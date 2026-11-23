"""Turn benchmark results into the rows the SMOLBoard site reads.

Reads a per-pair results CSV (the checkpoint written by run_benchmark.py) and,
optionally, a case-tags file from golden_dataset_formation/tag_cases.py.
Writes one row per model to docs/data/leaderboard.json.

Usage:
  python site_tools/build_leaderboard.py results/kr3_v2_checkpoint.csv \
      --tags datasets/geo/micro_case_tags.parquet --prompt text

What it computes for each model (one prompt at a time):
  accuracy, precision, recall, f1   same definitions as analysis.py: accuracy
                                    counts parse failures as wrong; precision,
                                    recall and f1 use parsed answers only, with
                                    MATCH as the positive label
  acc_match / acc_easy_negative / acc_hard_negative
                                    accuracy on each kind of pair
  smolboard_score                   0-100, using SCORE_WEIGHTS below:
                                    50% match accuracy, 15% easy-negative
                                    accuracy, 35% hard-negative accuracy.
                                    Matches and negatives each count for half,
                                    so always answering MATCH, always answering
                                    NO_MATCH, or random guessing all score 50
                                    ("no better than a coin flip"). Hard
                                    negatives weigh more than easy ones because
                                    that is where models actually differ.
  cases                             accuracy and pair count for each case tag
"""
import argparse
import csv
import datetime as dt
import json
import re
from collections import defaultdict
from pathlib import Path

POSITIVE = "MATCH"
PAIR_TYPES = ["match", "easy_negative", "hard_negative"]

# How much each kind of pair counts toward the SMOLBoard score. Must add to 1.
# Keep match at 0.5 and easy + hard at 0.5 so a biased model scores 50.
SCORE_WEIGHTS = {"match": 0.50, "easy_negative": 0.15, "hard_negative": 0.35}

# Parameter counts for models whose names don't carry a size like "7b".
KNOWN_PARAMS_B = {"phi-3-mini": 3.8}


def params_billions(model):
    name = model.lower()
    for key, value in KNOWN_PARAMS_B.items():
        if key in name:
            return value
    match = re.search(r"(\d+(?:\.\d+)?)b(?![a-z])", name.split("/")[-1])
    return float(match.group(1)) if match else None


def bucket(params_b):
    if params_b is None:
        return None
    if params_b < 5:
        return "<5B"
    if params_b < 10:
        return "<10B"
    return "<20B"


def is_true(value):
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_div(a, b):
    return a / b if b else None


def prf(rows):
    ok = [r for r in rows if r["status"] == "ok"]
    tp = sum(1 for r in ok if r["prediction"] == POSITIVE and r["true_label"] == POSITIVE)
    fp = sum(1 for r in ok if r["prediction"] == POSITIVE and r["true_label"] != POSITIVE)
    fn = sum(1 for r in ok if r["prediction"] != POSITIVE and r["true_label"] == POSITIVE)
    if not ok:
        return None, None, None
    precision = safe_div(tp, tp + fp) or 0.0
    recall = safe_div(tp, tp + fn) or 0.0
    f1 = safe_div(2 * precision * recall, precision + recall) or 0.0
    return precision, recall, f1


def accuracy(rows):
    return safe_div(sum(is_true(r["correct"]) for r in rows), len(rows))


def load_tags(path):
    if not path:
        return {}, []
    import pyarrow.parquet as pq
    table = pq.read_table(path).to_pylist()
    names = [k for k in table[0] if k != "pair_id"] if table else []
    return {int(r["pair_id"]): [k for k in names if r[k]] for r in table}, names


def summarize(model, rows, tags_by_pair, tag_names, prompt, date):
    precision, recall, f1 = prf(rows)
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["pair_type"]].append(r)
    split = {t: accuracy(by_type[t]) for t in PAIR_TYPES}
    score = None
    if all(split[t] is not None for t in PAIR_TYPES):
        score = 100 * sum(SCORE_WEIGHTS[t] * split[t] for t in PAIR_TYPES)
    params_b = params_billions(model)

    cases = {}
    if tags_by_pair:
        by_tag = defaultdict(list)
        for r in rows:
            for tag in tags_by_pair.get(int(r["example_id"]), []):
                by_tag[tag].append(r)
        cases = {t: {"accuracy": accuracy(by_tag[t]), "n": len(by_tag[t])}
                 for t in tag_names if by_tag[t]}

    def rnd(v):
        return None if v is None else round(v, 4)

    return {
        "model": model,
        "model_name": model,
        "dataset": "micro_geo",
        "prompt": prompt,
        "n_pairs": len(rows),
        "smolboard_score": None if score is None else round(score, 1),
        "accuracy": rnd(accuracy(rows)),
        "precision": rnd(precision),
        "recall": rnd(recall),
        "f1": rnd(f1),
        "acc_match": rnd(split["match"]),
        "acc_easy_negative": rnd(split["easy_negative"]),
        "acc_hard_negative": rnd(split["hard_negative"]),
        "parse_failure_rate": rnd(safe_div(sum(r["status"] != "ok" for r in rows), len(rows))),
        "average_latency": None,
        "tokens_per_second": None,
        "benchmark_date": date,
        "parameter_count": int(params_b * 1e9) if params_b else None,
        "parameter_bucket": bucket(params_b),
        "cases": {k: {"accuracy": rnd(v["accuracy"]), "n": v["n"]} for k, v in cases.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="per-pair results CSV from run_benchmark.py")
    ap.add_argument("--tags", help="case tags parquet from tag_cases.py (same dataset as the run)")
    ap.add_argument("--prompt", default="text", help="which prompt condition to publish (default: text)")
    ap.add_argument("--date", default=dt.date.today().isoformat(), help="benchmark date to show")
    ap.add_argument("--out", default="docs/data/leaderboard.json")
    args = ap.parse_args()

    with open(args.checkpoint, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["prompt"] == args.prompt]
    if not rows:
        raise SystemExit(f"no rows for prompt '{args.prompt}' in {args.checkpoint}")

    tags_by_pair, tag_names = load_tags(args.tags)
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    out = []
    for model, model_rows in by_model.items():
        if all(r["status"] != "ok" for r in model_rows):
            print(f"skipping {model}: no successful answers")
            continue
        out.append(summarize(model, model_rows, tags_by_pair, tag_names, args.prompt, args.date))
    out.sort(key=lambda r: -(r["smolboard_score"] or 0))

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(out)} models to {args.out}")
    for r in out:
        print(f"  {r['model']:28s} score {r['smolboard_score']}  acc {r['accuracy']}  "
              f"match {r['acc_match']}  easy {r['acc_easy_negative']}  hard {r['acc_hard_negative']}")


if __name__ == "__main__":
    main()
