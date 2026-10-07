#!/usr/bin/env python

import argparse
import os
from collections import defaultdict

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from checkpoint import load_results, read_runtime_records

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def compute_metrics(g, pos_label="MATCH"):
    total = len(g)
    ok = g[g["status"] == "ok"]
    n_ok = len(ok)
    accuracy = g["correct"].mean() if total else float("nan")
    parse_failure_rate = (total - n_ok) / total if total else float("nan")
    if n_ok == 0:
        precision = recall = f1 = float("nan")
    else:
        precision = precision_score(ok["true_label"], ok["prediction"], pos_label=pos_label, zero_division=0)
        recall = recall_score(ok["true_label"], ok["prediction"], pos_label=pos_label, zero_division=0)
        f1 = f1_score(ok["true_label"], ok["prediction"], pos_label=pos_label, zero_division=0)
    return pd.Series({
        "n": total, "n_ok": n_ok, "accuracy": accuracy, "precision": precision,
        "recall": recall, "f1": f1, "parse_failure_rate": parse_failure_rate,
    })


def model_prompt_summary(results_df):
    return (
        results_df.groupby(["model", "prompt"])
        .apply(compute_metrics, include_groups=False)
        .reset_index()
        .sort_values(["prompt", "accuracy"], ascending=[True, False])
    )


def prompt_summary(results_df):
    return (
        results_df.groupby("prompt")
        .apply(compute_metrics, include_groups=False)
        .sort_values("accuracy", ascending=False)
    )


def pairtype_summary(results_df):
    return (
        results_df.assign(hard_neg_source=results_df["hard_neg_source"].fillna("n/a"))
        .groupby(["pair_type", "prompt"])
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
        .reset_index()
        .pivot(index="pair_type", columns="prompt", values="accuracy")
    )


def plot_accuracy_by_prompt(prompt_summary_df, path="accuracy_by_prompt.png", show=False):
    import matplotlib.pyplot as plt
    prompt_summary_df["accuracy"].plot(kind="bar", title="Accuracy by prompt format (pooled across models)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path)
    if show:
        plt.show()
    plt.close()


def runtime_provenance(checkpoint_path):
    """Which backend/precision produced each model's rows (from <checkpoint>.runtime.jsonl).
    Models with no entry were produced before the log existed (e.g. the Colab run)."""
    by_model = defaultdict(set)
    for r in read_runtime_records(checkpoint_path):
        by_model[r["model"]].add(f'{r["label"]} @ {r.get("host", "?")}')
    return {m: sorted(v) for m, v in by_model.items()}


def main():
    import matplotlib
    matplotlib.use("Agg")

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=os.path.join(PROJECT_DIR, "results", "kr3_v2_checkpoint.csv"))
    args = p.parse_args()
    args.checkpoint = os.path.expanduser(args.checkpoint)

    results_df = load_results(args.checkpoint)
    if results_df.empty:
        raise SystemExit(f"no results in {args.checkpoint}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 20)

    print(f"{len(results_df)} rows from {args.checkpoint}\n")
    print(model_prompt_summary(results_df).to_string(index=False))

    print("\nRows by status. unparseable_truncated: no MATCH/NOT_MATCH, and the reply was cut off by",
          "the 20-token cap; unparseable: no verdict in a reply that ended on its own. Both count",
          "as parse failures above.")
    print(results_df.groupby(["model", "prompt"])["status"].value_counts().unstack(fill_value=0))

    ps = prompt_summary(results_df)
    print("\nAccuracy / precision / recall / F1 by prompt format (pooled across all models):")
    print(ps)

    print("\nAccuracy by pair_type x prompt (pooled across models). hard_negative is the",
          "diagnostic case: high match/easy_negative accuracy with low hard_negative",
          "accuracy means a model (or prompt) is pattern-matching on name text rather",
          "than actually comparing address/locality.")
    print(pairtype_summary(results_df))

    provenance = runtime_provenance(args.checkpoint)
    print("\nRuntime that produced each model's rows:")
    for m in results_df["model"].unique():
        print(f"  {m:26s} {', '.join(provenance.get(m, ['(not recorded -- predates the runtime log)']))}")

    png = os.path.splitext(args.checkpoint)[0] + "_accuracy_by_prompt.png"
    plot_accuracy_by_prompt(ps, png)
    print(f"\nplot: {png}")


if __name__ == "__main__":
    main()
