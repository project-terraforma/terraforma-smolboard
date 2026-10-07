# KR3 location-matching benchmark — local runner

A portable version of `kr3_location_matching_prompt_eval.ipynb`: 10 Hugging Face
models × 1,000 Terraforma golden pairs × 3 prompt conditions (`json`, `text`, `dspy`),
with greedy 20-token generation, the notebook's MATCH/NOT_MATCH parser, resumable CSV
checkpoint and metrics. It runs as a plain Python script on macOS, Linux and Windows,
on NVIDIA/AMD GPUs, Apple Silicon, Intel GPUs or CPU.

**Prompt version v2.** The `json` and `text` prompts follow the canonical spec
("Terraforma SMOLBoard Running Notes", LLM Prompts 1 and 2), not the notebook. The
Colab run used the notebook's prompts (v1), so v2 is a new experiment and its results
are kept apart from Colab's (see [Prompt versions](#prompt-versions)). `dspy` is the
same in both versions.

**Precision.** Run every model with NF4 + float16 compute, as on the Colab T4, except
Gemma-3, which overflows in float16 and runs in float32:

```bash
python run_benchmark.py --dtype float16                          # all models; Gemma is skipped after 10 errors
python run_benchmark.py --models gemma-3-4b-it --dtype float32   # Gemma, into the same checkpoint
```

The default (`--dtype auto`) is bfloat16 on Apple Silicon and on sm_80+ GPUs, which is
*not* the Colab precision. Each model's dtype is recorded in the runtime log.

```
benchmark.py      the experiment: models, dataset, prompts, DSPy signature + local-LM routing,
                  generate(), parser            (carried over verbatim from the notebook)
runtime.py        portable runtime: backend detection, precision, model load/free
checkpoint.py     results + checkpoint: resume, torn-tail repair, retries (verbatim),
                  per-model runtime log, optional HF Hub mirror
preflight.py      environment checks (--check, and the local ones before every run)
run_benchmark.py  CLI + the notebook's run loop
analysis.py       the notebook's analysis cells (metrics tables + plot)
data/             golden dataset, downloaded on first run and pinned by commit + sha256
results/          checkpoints (created on first run)
```

## Setup

```bash
cd kr3-benchmark
python3.12 -m venv .venv && source .venv/bin/activate     # Python 3.10–3.13
pip install -r requirements.txt
hf auth login              # or: export HF_TOKEN=hf_...
python run_benchmark.py --check
```

On Windows (PowerShell), activate with `.venv\Scripts\Activate.ps1`, and set a token
with `$env:HF_TOKEN = "hf_..."`. **The PyPI torch wheel for Windows is CPU-only**:
on an NVIDIA machine, first install the CUDA build of the pinned version, e.g.
`pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`, then
`pip install -r requirements.txt`. `--check` warns if a GPU is present but torch can't
use it. Git is not needed. torch 2.14.1 publishes wheels for Linux (x86_64, aarch64),
Windows x86_64, and macOS only on Apple Silicon with macOS 14 or newer. Intel Macs and
Windows on ARM can't install it.

`gemma-3-4b-it` and `Llama-3.2-3B-Instruct` are gated, so accept their licenses on
huggingface.co with the account you log in as. `--check` loads no model and exits
non-zero on any problem. It checks:

- the Python version, and that the installed libraries match the pins in `requirements.txt`
- the detected backend, and whether a GPU is present that this torch build can't use
- that bitsandbytes NF4 actually runs on that device and compute dtype (a small layer
  against its dequantized reference)
- the dataset sha256, and that the `json`, `text` and `dspy` messages for all 1,000 pairs
  are byte-identical to the frozen benchmark (`PROMPT_SHA256` in `benchmark.py`)
- the HF login (missing, rejected or unreachable), whether the checkpoint is free or
  already in use by another run, per-model access (offline: whether the pinned revision
  is cached), and free disk space for the models still to download

Every run repeats the local checks before it starts. A failure there (for example
NF4 kernels that don't run, or prompts that differ) stops the run before any model is
downloaded. Warnings are printed and the run continues.

## Running

```bash
python run_benchmark.py --dtype float16                    # the full benchmark (30,000 generations)
python run_benchmark.py --models gemma-3-4b-it --dtype float32
python run_benchmark.py --dry-run --dtype float16          # notebook DRY_RUN: 3 examples × every model
python run_benchmark.py --dry-run --n-examples 10 --models smollm2-1.7b-instruct --dtype float16
python run_benchmark.py --sample 15 --dtype float16        # notebook FULL_RUN=False stratified set
python run_benchmark.py --list-models
python analysis.py                                         # metrics for results/kr3_v2_checkpoint.csv
python analysis.py --checkpoint results/kr3_v2_dryrun_checkpoint.csv
```

A reply that yields no MATCH/NOT_MATCH is recorded with its raw text as
`unparseable_truncated` if generation was cut off by the 20-token cap, or as
`unparseable` if the model ended its reply itself. Both count as parse failures, exactly
as `unparseable` did before; `analysis.py` prints the breakdown.

**Stopping and resuming.** Ctrl-C, `kill` (SIGTERM), closing the terminal
(SIGHUP) or, on Windows, Ctrl-Break stops a run cleanly at any point. Re-run the
same command to continue. Closing a Windows console window kills the process
outright, like a hard kill.
Finished generations are flushed every 25 rows and again on interrupt, each flush
as a single write followed by an fsync. A hard kill or power loss loses at most the
unflushed rows. A half-written last record is dropped and re-run on the next start,
even if it happens to parse. Rows recorded as `model_load_error` or
`generation_error` are retried on resume, and the newest attempt wins. Only one
model is in memory at a time, and it is freed before the next one loads.

Only one run may write a checkpoint at a time. A second run on the same file exits
immediately (`<checkpoint>.lock`, on every OS), because two writers would corrupt
it. `analysis.py` never modifies the checkpoint, so it is safe to run while a
benchmark is in progress. Rows are written with `\n` line endings on every OS, so a
checkpoint can move between machines.

For long runs over SSH, start the run inside `tmux`/`screen`, or use
`nohup python run_benchmark.py > run.log 2>&1 &`. Progress lines are line-buffered.

`--models` changes only which models run. Example ids stay the same, so a subset
run writes into the same checkpoint as the full benchmark and a later full run
skips what is already done.

## Hardware backends

The backend is auto-detected in this order: CUDA, then MPS, then XPU, then CPU.
Override it with `--backend`. Every backend defaults to the notebook's reference
precision: **bitsandbytes 4-bit NF4 with double quantization**.
bitsandbytes ≥ 0.49 provides NF4 kernels for all of these backends, so no backend
needs a different quantization scheme.

| backend | hardware | default | 4-bit compute dtype | tested with this code |
|---|---|---|---|---|
| `cuda` | NVIDIA sm_80+ (A100, L4, H100, RTX 30xx+) | NF4 | bfloat16 | no (no NVIDIA GPU was available); the Linux requirements resolve |
| `cuda` | NVIDIA below sm_80 (T4, V100) | NF4 | float16, as in the Colab T4 run | no; same load/generate config as the notebook's Colab run |
| `cuda` | AMD ROCm build of torch | NF4 | same rule | no |
| `mps` | Apple Silicon | NF4 | bfloat16 | **yes**, M4 Pro on macOS 26 (see below) |
| `xpu` | Intel GPUs | NF4 | bfloat16 | not tested |
| `cpu` | anything | NF4 | bfloat16 | loads and generates; slow |

The compute dtype rule on CUDA is the notebook's `pick_compute_dtype()`: bf16 on
sm_80 and newer, fp16 below that. Off CUDA the default is bf16, following the
same intent (use bf16 wherever the hardware runs it natively). Override it with
`--dtype`, or set `--quantization none` to load plain bf16/fp16 weights instead of
NF4. That is a different experiment, so it writes to its own checkpoint file (see
below).

On Apple Silicon, `pip install -r requirements.txt` also installs `kernels`. That
package lets bitsandbytes download its prebuilt Metal NF4 kernels from the HF Hub
(macOS 26+). Without it, or on an older macOS, bitsandbytes falls back to a
pure-PyTorch implementation that gives the same results about 2× slower.

### How close the Mac is to the Colab run (measured, prompt v1)

These comparisons necessarily use the v1 prompts, the ones the Colab run used.
Qwen3-4B-Instruct on an M4 Pro, comparing 60 generations (20 random pairs × 3
conditions) with the same rows of the Colab T4 checkpoint:

| Mac configuration | gen/s | memory | raw text identical to Colab | same verdict as Colab |
|---|---|---|---|---|
| NF4, fp16 compute (`--dtype float16`, the T4's dtype) | 1.12 | 2.6 GB | **100 %** | 100 % |
| NF4, bf16 compute (**default**) | 1.12 | 2.6 GB | 98.3 % | **100 %** |
| NF4, bf16, without `kernels` (PyTorch fallback) | 0.61 | 2.6 GB | 96.7 % | 100 % |
| unquantized bf16 (`--quantization none`) | 1.21 | 8.0 GB | 70 % | 90 % |

Mistral-7B-Instruct-v0.3 with the same comparison (60 rows) uses 4.0 GB and runs
at 0.57 gen/s. Its raw text is 100 % identical to Colab with both bf16 and fp16
compute.

On these samples, NF4 on MPS reproduces the Colab experiment. Dropping the
quantization changes one verdict in ten, which is why `--quantization none` gets a
separate checkpoint. At these speeds the full 30,000 generations take roughly
10 hours on an M4 Pro, about 0.6 gen/s for the 6–7B models and 1.1 for the rest.

**Gemma-3 needs bf16 (or fp32) compute.** Every one of the 3,000 Gemma rows in the
Colab T4 checkpoint is an empty string scored `unparseable`. The cause is numerical,
not the chat template or the parser: with float16 compute, Gemma-3's logits are NaN
from the first decoding step, greedy decoding then picks token 0 (`<pad>`) twenty
times, and `<pad>` decodes to `""`. The same configuration on the Mac reproduces
this exactly (NaN at 20 of 20 steps). With bf16 or fp32 compute, the prompts are
rendered correctly (one `<bos>`), the logits are finite, and Gemma answers normally.
`generate()` now checks the logits. NaN or inf logits raise `NonFiniteLogitsError`,
so the row is recorded as a retryable `generation_error` instead of an `unparseable`
answer, and the model is skipped after 10 in a row. The check reads the logits
without changing them, and costs nothing measurable.

The benchmark therefore runs Gemma with **float32** compute, the one model not at the
Colab precision: `python run_benchmark.py --models gemma-3-4b-it --dtype float32`, into
the same checkpoint as the rest. fp32 is the working precision closest to fp16: on
Qwen3, fp32 reproduced Colab's fp16 outputs 90/90 byte-for-byte (bf16: 88/90), and
Gemma's bf16 and fp32 verdicts agreed 90/90. The 4-bit weights are identical under every
compute dtype; only the arithmetic differs. Label Gemma's precision when reporting results.

Gemma's `dspy` replies are a separate effect. Every one of 40 sampled replies restates
the input fields first (`[[ ## location_a ## ]] {...}`). It reaches `[[ ## match ## ]]`
only after about 230–270 tokens, far past the 20-token cap, so Gemma's `dspy` rows are
recorded as `unparseable_truncated`, with their raw text. (Measured with the v1 prompts
in bf16 and fp32; the `dspy` prompt is the same in v2.) The prompt is delivered correctly.
Gemma has no system role, so its official template folds DSPy's system message into the
user turn. Every other model in the Colab run starts its reply with `[[ ## match ## ]]`.
This is model behavior under the benchmark's fixed settings, and changing the token cap
or the DSPy prompt would change the experiment.

The startup banner names the exact configuration, for example
`runtime: mps-nf4-bfloat16 on Apple M4 Pro`. Each model load is also logged to
`results/<checkpoint>.runtime.jsonl` with the backend, device, precision, library
versions and host. `analysis.py` prints that provenance for every model.

## Checkpoint files

| run | file |
|---|---|
| full benchmark, NF4 (any backend) | `results/kr3_v2_checkpoint.csv` |
| `--dry-run` | `results/kr3_v2_dryrun_checkpoint.csv` |
| `--sample N` | `results/kr3_v2_sample<N>_checkpoint.csv` (sample ids are renumbered, so they must not mix with full-run ids) |
| `--quantization none` | `....<dtype>.csv`, e.g. `kr3_v2_checkpoint.bfloat16.csv` |
| anything | `--checkpoint PATH` overrides the default |

The CSV schema is unchanged from the notebook. The `v2` in the names is the prompt
version; a run refuses a checkpoint that holds rows from another prompt version.

## Moving to another machine (e.g. a remote NVIDIA server)

```bash
# on the Mac
rsync -av --exclude .venv --exclude data kr3-benchmark/ server:kr3-benchmark/   # includes results/
# on the server
cd kr3-benchmark && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Linux torch wheel from PyPI includes CUDA (see requirements.txt)
hf auth login                       # or export HF_TOKEN
python run_benchmark.py --check --dtype float16     # expect: runtime: cuda-nf4-float16 on <GPU>
tmux new -s kr3 'python run_benchmark.py --dtype float16 && python run_benchmark.py --models gemma-3-4b-it --dtype float32'
```

- **Code only, fresh results:** leave `results/` out of the copy.
- **Continue the same checkpoint:** copy `results/`. Already-finished models are
  skipped. The runtime log records which models ran on which machine.
- **No persistent disk** (spot or ephemeral VMs): add `--hub-sync`. The checkpoint is
  mirrored to a private dataset repo `<you>/kr3-benchmark-checkpoints` (branch
  `checkpoint`) every 120 s and after each model, and restored automatically on a
  fresh machine. This is the notebook's `PERSIST_CHECKPOINT_TO_HUB`. It was not
  exercised while porting, because doing so creates a repo on your HF account.
  The default repo is the one the Colab notebook used, but the v2 file names differ
  from Colab's `kr3_checkpoint.csv`, so the Colab checkpoint is never restored into a
  v2 run.
- **Offline compute nodes:** download once, then run with `HF_HUB_OFFLINE=1`. Model
  revisions are pinned, so they resolve from the local cache. On a Mac, bitsandbytes
  can't resolve its Metal kernels offline and uses its PyTorch fallback: about 2×
  slower, verdicts identical in testing, raw text slightly different. The runtime log
  records which kernel path each model used.
- **Small disk:** add `--delete-weights`. Each model is removed from the HF cache
  once all its rows are recorded. The 10 models total about 95 GB. This was always
  on in Colab, but it is off by default here because it deletes from your shared
  `~/.cache/huggingface`.
- Put the HF cache somewhere else with `export HF_HOME=/big/disk/hf`.

## Prompt versions

**v1** is the notebook's prompts, which the Colab run used. Its byte-for-byte
reproduction on this Mac (fp16) confirms that. **v2** (current) is the canonical spec. All
v2 prompts fit every model's context window (longest `json` prompt: 1,489 tokens).

| | v1 (Colab) | v2 (spec) |
|---|---|---|
| `text` layout | no blank lines, `- ` bullets | the spec's blank lines and `●` bullets |
| `text` coordinates | withheld | withheld (the spec's two `Coordinates:` lines are omitted) |
| `json` fields | 12, `names` first, `names`/`brand` reduced to their primary name | the spec's 18 fields in its order, real values, minus `id` and `geometry` |
| `json` instruction | "Determine whether location_a and location_b represent the same physical location. Answer with exactly: MATCH or NOT_MATCH" | "Answer with exactly: MATCH or NOT_MATCH" (the spec's answer line) |
| `dspy` | unchanged | unchanged |

Two deliberate departures from the spec:

- `id` is withheld because MATCH pairs share their Overture id by construction:
  identical for 100 % of MATCH and 0 % of NOT_MATCH pairs.
- Coordinates are withheld because labels were defined by distance: "MATCH if under
  5 m apart" alone scores 93.2 %, and the dataset's design never shows geometry to the
  model.

The spec's own JSON example prints nested values as `"[object Object]"`, a JavaScript
display artifact, so v2 sends the real values.

**The Colab run (v1) is kept apart.** Its 3,000 Gemma rows are discarded: they are
numerical failures, not answers. The v1 record without them is
`results/kr3_v1_colab_checkpoint.csv` (`python analysis.py --checkpoint
results/kr3_v1_colab_checkpoint.csv`). It is made from the Colab download, which itself
stays untouched:

```bash
python -c "import pandas as pd; d = pd.read_csv('PATH/TO/kr3_checkpoint.csv', dtype=str, keep_default_na=False); d[d.model != 'gemma-3-4b-it'].to_csv('results/kr3_v1_colab_checkpoint.csv', index=False, lineterminator='\n')"
```

v1 rows cannot enter a v2 checkpoint: the preflight fails on any checkpoint whose rows
come from another prompt version, including Colab rows, which have no runtime log.

## What differs from the notebook

Unchanged: the model list, dataset construction, the DSPy signature, adapter and
prompt, `generate()` (chat template, BOS de-duplication, EOS set),
`MAX_NEW_TOKENS=20`, greedy decoding, the parser, the checkpoint schema and resume
rules, and the metrics. All of these are carried over verbatim, with three exceptions in
`generate()`, described below: the out-of-memory handler now frees the cache on any
backend instead of only CUDA, the chat-template date is pinned, and non-finite logits
are raised as errors. It also notes whether a reply hit the token cap.

Changed, and why:

- **The `json` and `text` prompts are v2**, the canonical spec, not the notebook's v1
  (see [Prompt versions](#prompt-versions)). The v1 prompts were checked byte-for-byte
  against the notebook's own code before the change.
- **Unparseable replies cut off by the 20-token cap** are recorded as
  `unparseable_truncated` instead of `unparseable`. Scoring is unchanged: both are
  parse failures.

- **Device handling** moved into `runtime.py`. CUDA keeps `device_map="auto"`.
  Other backends pin the whole model to their single device, so a model that
  doesn't fit fails loudly instead of being silently offloaded. OOM handling and
  memory release now cover MPS and XPU as well as CUDA.
- **No Colab dependencies.** There are no `google.colab` secrets or `!pip`.
  Hugging Face auth uses the standard token from `hf auth login` or `HF_TOKEN`.
  Paths are relative to the project, not the CWD.
- **Dataset pinned** to commit `c0e52a70` with a sha256 check. The notebook
  cloned the branch head, which could change under an existing checkpoint. That
  commit reproduces every (example_id, pair_type, label) in the Colab checkpoint.
  Only that one file is downloaded, over HTTPS, so git is not required.
- **Chat-template date pinned** (`CHAT_TEMPLATE_DATE` in `benchmark.py`). The
  Llama-3.2 template writes "Today Date: <today>" into every prompt, and the
  Granite-3.1 template writes "Today's Date: <today>." into its default system
  prompt (`json` and `text`). transformers fills these in from the local clock, so
  the exact input changed from day to day, between time zones, and at midnight
  during a run. Both now render 26 Jul 2024, the date Llama's own template uses when
  no clock is available. The other eight templates render byte-identically either way
  (checked on all three conditions). Set it to `None` to restore the notebook's
  wall-clock behavior. The date used is recorded in the runtime log.
- **Non-finite logits are errors, not answers** (see the Gemma note above): a
  read-only logits check turns NaN/inf output into a retryable `generation_error`.
  Healthy generations are unchanged: Gemma's raw outputs were identical with and
  without it, and so was its speed.
- **Library versions pinned** to the set tested here. The notebook pinned only
  `dspy` and `pandas`, and installed the latest `transformers`/`bitsandbytes` at
  run time, so the exact Colab versions are unknown.
- **Model revisions pinned** (`MODEL_REVISIONS` in `benchmark.py`). The notebook
  loaded each repo's current HEAD, so a later upstream edit to weights, a chat
  template or `generation_config` would silently change the benchmark between
  machines. None of the 10 repos changed after 2025-12-10, so the pinned commits are
  exactly what the Colab run loaded.
- **DSPy call** (the `dspy` condition's prompt is unchanged):
  - It runs inside `dspy.context(...)` instead of calling `dspy.configure(...)` per
    example. `configure` can only be called from the thread that first configured
    DSPy, so the condition failed on every example when run from any other thread
    (a notebook kernel, a job runner). It also left the engine, and through it the
    loaded model, in DSPy's global settings.
  - DSPy's call history is turned off. It kept the last 10,000 requests in memory
    for nothing.
- **All three conditions' messages are fingerprinted** (`PROMPT_SHA256`, all 1,000
  pairs). DSPy's adapter, not this code, renders the `dspy` prompt, and the frozen
  `json`/`text` prompts depend on how the installed pyarrow converts the parquet rows.
  So another machine could silently run a different condition. `run_benchmark.py`
  refuses to start unless this machine builds all three byte-identically.
- **Checkpoint hardening:**
  - A single-write flush with fsync.
  - A torn last record is always dropped.
  - Recovery from an empty file or a torn header, which previously crashed every
    resume.
  - A clear error for a CSV that isn't a checkpoint.
  - A per-checkpoint lock. Two concurrent runs previously wrote a second header
    mid-file, and the checkpoint could never be resumed. It uses `flock` on
    macOS/Linux and `filelock` (already installed with huggingface_hub) on Windows.
  - `\n` line endings on every OS. pandas defaults to `\r\n` on Windows.
  - The notebook's "nothing parses" branch now raises the real parse error instead
    of `RuntimeError: No active exception to reraise`.
- **Weight deletion and the Hub mirror are opt-in** (`--delete-weights`,
  `--hub-sync`). Both were on in Colab to work around its 80 GB disk and VM resets.
  The notebook's Hub cell also used `CommitScheduler` without importing it, so it
  fails on a fresh runtime; that is fixed here.
- **SIGTERM/SIGHUP** (and SIGBREAK on Windows) now go through the same
  flush-and-free path as Ctrl-C, for schedulers, `kill` and closed terminals.
  Memory release never raises, so an error on a faulted GPU can't hide the error
  that actually stopped the run. A half-loaded model from a failed load is released
  before the next model loads.
- **Preflight checks** (`preflight.py`, see Setup) run before every run as well as
  with `--check`.
- **Import-order fix.** `import dspy` before numpy breaks pyarrow with DSPy 3.3.1.
  The notebook avoided this only because it happened to import pandas first.
- **Progress counts** in a `--models` subset run are measured against that subset
  instead of the whole checkpoint. This affects only the progress line.
