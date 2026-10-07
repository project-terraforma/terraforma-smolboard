#!/usr/bin/env python
"""Run (or resume) the KR3 location-matching benchmark locally.

    python run_benchmark.py                      # full benchmark: 10 models x 1000 pairs x 3 conditions
    python run_benchmark.py --dry-run            # smoke test: 3 examples per model, separate checkpoint
    python run_benchmark.py --models qwen3-4b-instruct,smollm2-1.7b-instruct
    python run_benchmark.py --check              # preflight only: install, backend, data, prompts, HF access, disk

Stop it at any time (Ctrl-C, or SIGTERM from a job scheduler); re-run the same command to
resume. One model is resident at a time. See README.md for the options.
"""

import argparse
import os
import platform
import signal
import sys
import time
import traceback
from datetime import datetime, timezone

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import pandas as pd

import benchmark
import checkpoint
import preflight
import runtime
from benchmark import CONDITIONS, MODELS, parse_match

RESULTS_DIR = os.path.join(benchmark.PROJECT_DIR, "results")


CHECKPOINT_EVERY = 25
PROGRESS_EVERY = 10
MAX_CONSECUTIVE_GENERATION_ERRORS = 10


def format_eta(seconds):
    if seconds is None or seconds != seconds or seconds < 0:
        return "unknown"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def default_checkpoint_path(profile, dry_run, sample):
    """kr3_<prompt version>_checkpoint.csv is the benchmark of record. The prompt version is
    in the name because the Colab run (kr3_checkpoint.csv, prompt v1) is a different
    experiment from the current prompts. Other runs that are a different experiment get
    their own file and never mix into it:
      * --sample N re-numbers example ids 0..4N-1, so its ids would collide with the
        full run's -> kr3_v2_sample<N>_...
      * --dry-run -> kr3_v2_dryrun_... (as in the notebook)
      * --quantization none uses different weights than NF4 -> ....<dtype>.csv
    """
    name = f"kr3_{benchmark.PROMPT_VERSION}"
    if sample is not None:
        name += f"_sample{sample}"
    if dry_run:
        name += "_dryrun"
    name += "_checkpoint"
    if profile.precision != "nf4":
        name += f".{profile.precision}"
    return os.path.join(RESULTS_DIR, name + ".csv")


def run_benchmark(active_models, dataset, checkpoint_path, profile, *, dry_run=False,
                  delete_weights=False, hub=None):
    print(("=== DRY RUN " if dry_run else "=== FULL PRODUCTION RUN ") +
          f"-- {len(active_models)} models x {len(dataset)} pairs x {len(CONDITIONS)} conditions ===")
    checkpoint_df, attempted = checkpoint.load_checkpoint(checkpoint_path)
    planned = {(m, p, e["id"]) for m in active_models for e in dataset for p in CONDITIONS}

    total_planned = len(planned)
    total_already_done = len(attempted & planned)
    n_done_this_session = 0
    session_start = time.time()
    buffer = []
    model = tokenizer = None

    def flush():
        if not buffer:
            return
        rows = buffer[:]
        buffer.clear()
        checkpoint.append_rows(checkpoint_path, rows)

    def record(model_name, prompt_name, example, raw, prediction, status):
        correct = bool(status == "ok" and prediction == example["label"])
        buffer.append({
            "model": model_name, "prompt": prompt_name, "example_id": example["id"],
            "pair_type": example["pair_type"], "hard_neg_source": example["hard_neg_source"],
            "true_label": example["label"], "raw_output": raw, "prediction": prediction,
            "status": status, "correct": correct,
        })
        attempted.add((model_name, prompt_name, example["id"]))
        if len(buffer) >= CHECKPOINT_EVERY:
            flush()

    def sync_checkpoint_now(reason):
        if hub is not None:
            hub.sync_now(reason)

    try:
        for short_name, repo_id in active_models.items():
            remaining = [
                (prompt_name, example)
                for example in dataset
                for prompt_name in CONDITIONS
                if (short_name, prompt_name, example["id"]) not in attempted
            ]
            if not remaining:
                print(f"\n[{short_name}] already complete -- skipping model load")
                continue

            print(f"\n=== {short_name} ({repo_id}) -- {len(remaining)} generations remaining ===")
            t_load0 = time.time()
            load_failed = False
            try:
                model, tokenizer = runtime.load_model(repo_id, profile, benchmark.MODEL_REVISIONS.get(short_name))
            except Exception as e:
                print(f"  FAILED to load: {type(e).__name__}: {e}")
                for prompt_name, example in remaining:
                    record(short_name, prompt_name, example,
                           f"[MODEL LOAD ERROR: {type(e).__name__}: {e}]", None, "model_load_error")
                flush()
                load_failed = True
            if load_failed:
                runtime.free_model()
                continue
            load_seconds = time.time() - t_load0
            print(f"  loaded in {load_seconds:.0f}s on {model.device} "
                  f"({model.get_memory_footprint() / 1e9:.1f} GB, {profile.label})")
            checkpoint.append_runtime_record(checkpoint_path, {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "host": platform.node(),
                "model": short_name,
                "repo_id": repo_id,
                "revision": benchmark.MODEL_REVISIONS.get(short_name) or "main (unpinned)",
                "generations_remaining": len(remaining),
                "dataset_commit": benchmark.TERRAFORMA_COMMIT,
                "prompt_version": benchmark.PROMPT_VERSION,
                "prompt_sha256": benchmark.PROMPT_SHA256,
                "chat_template_date": (benchmark.CHAT_TEMPLATE_DATE.date().isoformat()
                                       if benchmark.CHAT_TEMPLATE_DATE else "wall clock"),
                "load_seconds": round(load_seconds, 1),
                "memory_footprint_gb": round(model.get_memory_footprint() / 1e9, 2),
                **profile.describe(),
            })

            model_start = time.time()
            consecutive_errors = 0
            abandoned = False
            for i, (prompt_name, example) in enumerate(remaining, 1):
                run_fn = CONDITIONS[prompt_name]
                try:
                    raw = run_fn(model, tokenizer, short_name, example)
                    pred = parse_match(raw)
                    if pred in ("MATCH", "NOT_MATCH"):
                        status = "ok"
                    elif benchmark.last_generation_hit_token_cap():
                        status = "unparseable_truncated"
                    else:
                        status = "unparseable"
                    consecutive_errors = 0
                except Exception as e:
                    raw = f"[GENERATION ERROR: {type(e).__name__}: {e}]"
                    pred = None
                    status = "generation_error"
                    consecutive_errors += 1

                record(short_name, prompt_name, example, raw, pred, status)
                n_done_this_session += 1

                if consecutive_errors >= MAX_CONSECUTIVE_GENERATION_ERRORS:
                    print(
                        f"  !! {short_name}: {consecutive_errors} consecutive generation errors -- "
                        f"abandoning this model for now, {len(remaining) - i} generations left "
                        f"un-attempted (they will be retried on the next resume). Last error:\n"
                        f"     {raw[:300]}"
                    )
                    abandoned = True
                    break

                if i % PROGRESS_EVERY == 0 or i == len(remaining):
                    elapsed = time.time() - model_start
                    rate = i / elapsed if elapsed > 0 else 0.0
                    session_elapsed = time.time() - session_start
                    session_rate = n_done_this_session / session_elapsed if session_elapsed > 0 else 0.0
                    remaining_total = total_planned - (total_already_done + n_done_this_session)
                    eta = remaining_total / session_rate if session_rate > 0 else None
                    print(
                        f"  [{short_name}/{prompt_name}] {i}/{len(remaining)} this model | "
                        f"{total_already_done + n_done_this_session}/{total_planned} overall | "
                        f"{rate:.2f} gen/s | elapsed {format_eta(session_elapsed)} | ETA {format_eta(eta)}"
                    )

            flush()
            sync_checkpoint_now(f"{short_name} completing")
            print(f"  [{short_name}] model done in {time.time() - model_start:.0f}s")
            model = tokenizer = None
            runtime.free_model()
            if delete_weights and not abandoned:
                runtime.delete_cached_weights(repo_id)
    except BaseException as interrupted:
        _exc, _seen = interrupted, set()
        while _exc is not None and id(_exc) not in _seen:
            _seen.add(id(_exc))
            traceback.clear_frames(_exc.__traceback__)
            _exc = _exc.__cause__ or _exc.__context__
        raise
    finally:
        flush()
        sync_checkpoint_now("an interrupt or the run ending")
        model = tokenizer = None
        runtime.free_model()

    return checkpoint.load_results(checkpoint_path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="KR3 location-matching benchmark (local / portable runner).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sel = p.add_argument_group("what to run")
    sel.add_argument("--models", help="comma-separated short names (default: all 10, in benchmark order); see --list-models")
    sel.add_argument("--dry-run", action="store_true", help="smoke test: a few examples per model, written to a separate checkpoint")
    sel.add_argument("--n-examples", type=int, help=f"examples per model in a dry run (default {benchmark.DRY_RUN_N_EXAMPLES})")
    sel.add_argument("--sample", type=int, metavar="N",
                     help="the notebook's FULL_RUN=False diagnostic set: N per pair type, stratified (own checkpoint)")
    hw = p.add_argument_group("hardware")
    hw.add_argument("--backend", default="auto", choices=("auto",) + runtime.BACKENDS)
    hw.add_argument("--quantization", default="auto", choices=("auto",) + runtime.QUANTIZATIONS,
                    help="auto = nf4 (the reference config) on every backend")
    hw.add_argument("--dtype", default="auto", choices=("auto",) + tuple(runtime.DTYPES),
                    help="nf4 compute dtype / unquantized weight dtype. auto: CUDA bf16 on sm_80+, else fp16 "
                         "(the notebook's rule); bf16 on MPS/XPU/CPU")
    out = p.add_argument_group("storage")
    out.add_argument("--checkpoint", help="checkpoint CSV path (default: results/kr3_v2_checkpoint.csv, see README)")
    out.add_argument("--delete-weights", action="store_true",
                     help="remove each model's weights from the HF cache once all its rows are recorded (small disks)")
    out.add_argument("--hub-sync", action="store_true", help="mirror the checkpoint to a private HF Hub dataset repo")
    out.add_argument("--hub-repo", help="repo for --hub-sync (default: <your HF username>/kr3-benchmark-checkpoints)")
    p.add_argument("--check", action="store_true",
                   help="preflight only: check the install, accelerator, NF4 kernels, dataset, prompts, HF login, "
                        "model access and disk space, then exit (non-zero on any problem)")
    p.add_argument("--list-models", action="store_true", help="print the model list and exit")
    args = p.parse_args(argv)
    if args.n_examples is not None and not args.dry_run:
        p.error("--n-examples only applies to --dry-run")
    for flag in ("n_examples", "sample"):
        if getattr(args, flag) is not None and getattr(args, flag) < 1:
            p.error(f"--{flag.replace('_', '-')} must be >= 1")
    return args


def select_models(spec):
    if not spec:
        return dict(MODELS)
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}. Known: {list(MODELS)}")
    return {n: MODELS[n] for n in MODELS if n in names}


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, errors="backslashreplace")
    args = parse_args(argv)
    if args.list_models:
        for short_name, repo_id in MODELS.items():
            print(f"{short_name:26s} {repo_id}")
        return 0

    stop = lambda signum, frame: sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)

    profile = runtime.resolve_profile(args.backend, args.quantization, args.dtype)
    active_models = select_models(args.models)
    checkpoint_path = os.path.abspath(os.path.expanduser(args.checkpoint) if args.checkpoint
                                      else default_checkpoint_path(profile, args.dry_run, args.sample))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"runtime: {profile.label} on {profile.device_name}")
    for note in profile.notes:
        print(f"  note: {note}")
    print(f"checkpoint: {checkpoint_path}")
    prior = {r["label"] for r in checkpoint.read_runtime_records(checkpoint_path)} - {profile.label}
    if prior:
        print(f"  note: this checkpoint also holds rows produced under {sorted(prior)}; "
              f"per-model provenance is in {os.path.basename(checkpoint.runtime_log_path(checkpoint_path))}")

    if args.check:
        report = preflight.run(profile, active_models, checkpoint_path, full=True, delete_weights=args.delete_weights)
        if report.failures or report.warnings:
            print(f"preflight: {len(report.failures)} failure(s), {len(report.warnings)} warning(s)")
            return 1
        print("preflight: everything OK")
        return 0

    try:
        lock = checkpoint.CheckpointLock(checkpoint_path)
    except RuntimeError as e:
        raise SystemExit(str(e))
    report = preflight.run(profile, active_models, checkpoint_path, full=False)
    if report.failures:
        raise SystemExit("preflight failed, nothing was run:\n  " + "\n  ".join(report.failures))
    hub = checkpoint.HubMirror(checkpoint_path, args.hub_repo) if args.hub_sync else None

    rows = benchmark.load_rows()
    print(f"{len(rows)} total pairs in {benchmark.GOLDEN_DATASET_FILE}")
    full_run = args.sample is None
    dataset = benchmark.build_dataset(rows, full_run, args.sample or benchmark.N_PER_PAIR_TYPE, benchmark.SAMPLE_SEED)
    if args.dry_run:
        dataset = benchmark.dry_run_slice(dataset, args.n_examples or benchmark.DRY_RUN_N_EXAMPLES)
    print(f"dataset ready: {len(dataset)} pairs" + (" [DRY RUN slice]" if args.dry_run else "")
          + ("" if full_run else f" [stratified sample, {args.sample} per pair type]"))
    print(f"active models this run: {list(active_models)}")
    print(f"  -> {len(active_models) * len(dataset) * len(CONDITIONS)} total generations planned")

    try:
        results_df = run_benchmark(active_models, dataset, checkpoint_path, profile, dry_run=args.dry_run,
                                   delete_weights=args.delete_weights, hub=hub)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Every finished generation is in {checkpoint_path}; "
              f"re-run the same command to resume.")
        return 130

    print(f"\n{len(results_df)} total rows in {checkpoint_path}")
    if len(results_df):
        import analysis
        with pd.option_context("display.width", 200, "display.max_rows", 100):
            print(analysis.model_prompt_summary(results_df).to_string(index=False))
        shown = os.path.relpath(checkpoint_path) if checkpoint_path.startswith(os.getcwd() + os.sep) else checkpoint_path
        print(f"\nfull analysis: python analysis.py --checkpoint {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
