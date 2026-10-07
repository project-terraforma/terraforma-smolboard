# KR3 location-matching benchmark — local runner

Portable version of `kr3_location_matching_prompt_eval.ipynb`: 10 Hugging Face models × 1,000 Terraforma golden pairs × 3 prompt conditions (`json`, `text`, `dspy`). Greedy 20-token generation, MATCH/NOT_MATCH parser, resumable CSV checkpoint. Runs on macOS, Linux and Windows, on NVIDIA/AMD GPUs, Apple Silicon, Intel GPUs or CPU.

**Prompt version v2.** The `json` and `text` prompts follow the canonical spec ("Terraforma SMOLBoard Running Notes", LLM Prompts 1 and 2), not the notebook. The Colab run used the notebook's prompts (v1), so v2 is a new experiment and its results are kept separate. The `dspy` prompt is the same in both.

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Python 3.10–3.13
pip install -r requirements.txt
hf auth login                                           # or: export HF_TOKEN=hf_...
python run_benchmark.py --check                         # verify the environment, loads no model

python run_benchmark.py --dtype float16                          # all models, 30,000 generations
python run_benchmark.py --models gemma-3-4b-it --dtype float32   # Gemma needs float32
python analysis.py                                               # metrics tables + plot
```

- `--dtype float16` matches the Colab T4 precision. The default (`auto`) is bfloat16 on Apple Silicon and sm_80+ GPUs, which does not.
- `gemma-3-4b-it` and `Llama-3.2-3B-Instruct` are gated. Accept their licenses with the account you log in as.
- **Windows:** activate with `.venv\Scripts\Activate.ps1` and set the token with `$env:HF_TOKEN = "hf_..."`. The PyPI torch wheel is CPU-only, so on NVIDIA first run `pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126`, then `pip install -r requirements.txt`.
- torch 2.14.1 has wheels for Linux (x86_64, aarch64), Windows x86_64, and Apple Silicon macOS 14+. Intel Macs and Windows on ARM can't install it.

`--check` exits non-zero on any problem. It verifies the Python and library versions, the backend, that NF4 runs on the device, the dataset sha256, that all prompts are byte-identical to the frozen benchmark (`PROMPT_SHA256`), HF login and model access, checkpoint availability, and disk space. The local checks also run before every benchmark and stop it before any download if they fail.

## Other commands

```bash
python run_benchmark.py --dry-run --dtype float16          # 3 examples × every model
python run_benchmark.py --sample 15 --dtype float16        # stratified subset
python run_benchmark.py --list-models
python analysis.py --checkpoint results/kr3_v2_dryrun_checkpoint.csv
```

## Stopping and resuming

- Ctrl-C, `kill` and closing the terminal (and Ctrl-Break on Windows) stop a run cleanly. Re-run the same command to resume.
- Rows are flushed every 25 rows and on interrupt. A hard kill loses at most the unflushed rows.
- `model_load_error` and `generation_error` rows are retried on resume.
- Only one run can write a checkpoint at a time (`<checkpoint>.lock`). `analysis.py` is read-only, so it is safe to run during a benchmark.
- `--models` only selects which models run. A subset run writes to the same checkpoint, and a later full run skips what is done.
- For long SSH runs, use `tmux`, `screen`, or `nohup python run_benchmark.py > run.log 2>&1 &`.

## Hardware backends

The backend is auto-detected (CUDA, then MPS, XPU, CPU); override it with `--backend`. All backends use bitsandbytes 4-bit NF4 with double quantization (bitsandbytes ≥ 0.49).

| backend | hardware | default compute dtype | tested |
|---|---|---|---|
| `cuda` | NVIDIA sm_80+ (A100, L4, H100, RTX 30xx+) | bfloat16 | No |
| `cuda` | NVIDIA below sm_80 (T4, V100) | float16 | No (same config as the Colab run) |
| `cuda` | AMD ROCm | same rule | No |
| `mps` | Apple Silicon | bfloat16 | **Yes**, M4 Pro, macOS 26 |
| `xpu` | Intel GPUs | bfloat16 | No |
| `cpu` | Any | bfloat16 | Loads and generates; slow |

Override the compute dtype with `--dtype`. `--quantization none` loads plain bf16/fp16 weights instead. That is a different experiment, so it writes to its own checkpoint.

**Apple Silicon.** `kernels` (installed by `requirements.txt`) lets bitsandbytes fetch prebuilt Metal NF4 kernels (macOS 26+). Without it, bitsandbytes falls back to PyTorch: same results, about 2× slower.

**Mac vs Colab (measured, v1 prompts).** Qwen3-4B-Instruct on an M4 Pro, 60 generations compared with the same Colab T4 rows:

| configuration | raw text identical | same verdict |
|---|---|---|
| NF4, fp16 compute | **100%** | 100% |
| NF4, bf16 compute (default) | 98.3% | **100%** |
| Unquantized bf16 | 70% | 90% |

NF4 on MPS reproduces the Colab experiment. A full run takes roughly 10 hours on an M4 Pro (about 0.6 gen/s for the 6–7B models, 1.1 for the rest).

## Gemma-3

- **Needs float32.** With float16 compute, Gemma-3's logits are NaN from the first step, greedy decoding emits `<pad>` 20 times, and the reply is an empty string. This is why all 3,000 Gemma rows in the Colab checkpoint are empty `unparseable`. The Mac reproduces it exactly. fp32 is the working precision closest to fp16, so Gemma is the one model not at the Colab precision. Label this when reporting results.
- **Detection.** `generate()` raises `NonFiniteLogitsError` on NaN/inf logits. The row is recorded as a retryable `generation_error`, and the model is skipped after 10 in a row.
- **`dspy` rows are truncated.** Gemma restates the input fields first and reaches `[[ ## match ## ]]` only after about 230–270 tokens, far past the 20-token cap. Its `dspy` rows are recorded as `unparseable_truncated`. This is model behavior under fixed settings, so the cap and prompt are not changed.

## Checkpoints

| run | file |
|---|---|
| Full benchmark | `results/kr3_v2_checkpoint.csv` |
| `--dry-run` | `results/kr3_v2_dryrun_checkpoint.csv` |
| `--sample N` | `results/kr3_v2_sample<N>_checkpoint.csv` (ids are renumbered, so never mix with full-run ids) |
| `--quantization none` | default name with `.<dtype>` before `.csv`, e.g. `kr3_v2_checkpoint.bfloat16.csv` |
| Any | `--checkpoint PATH` overrides the default |

The CSV schema is unchanged from the notebook. A run refuses a checkpoint with rows from another prompt version.

A reply with no MATCH/NOT_MATCH is recorded with its raw text as `unparseable_truncated` (hit the 20-token cap) or `unparseable` (the model ended the reply). Both count as parse failures; `analysis.py` shows the breakdown. Each model load is logged to `results/<checkpoint>.runtime.jsonl` (backend, device, precision, library versions, host).

## Moving to another machine

```bash
rsync -av --exclude .venv --exclude data kr3-benchmark/ server:kr3-benchmark/   # includes results/
# on the server
cd kr3-benchmark && python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login
python run_benchmark.py --check --dtype float16     # expect: runtime: cuda-nf4-float16 on <GPU>
tmux new -s kr3 'python run_benchmark.py --dtype float16 && python run_benchmark.py --models gemma-3-4b-it --dtype float32'
```

- **Fresh results:** leave `results/` out of the copy. **Continue a run:** copy it, and finished models are skipped.
- **No persistent disk:** add `--hub-sync` to mirror the checkpoint to a private dataset repo `<you>/kr3-benchmark-checkpoints` every 120 s and after each model, restored automatically on a fresh machine. Not exercised while porting, since it creates a repo on your account.
- **Offline nodes:** download once, then set `HF_HUB_OFFLINE=1`. Revisions are pinned, so they resolve from cache. On a Mac, NF4 then uses the PyTorch fallback (about 2× slower).
- **Small disk:** add `--delete-weights` to remove each model from the HF cache once its rows are recorded. The 10 models total about 95 GB.
- **Cache location:** `export HF_HOME=/big/disk/hf`.

## Prompt versions

| | v1 (Colab) | v2 (current, spec) |
|---|---|---|
| `text` | No blank lines, `- ` bullets | Spec's blank lines and `●` bullets |
| `json` fields | 12, primary `names`/`brand` only | Spec's 18 fields in order, real values, minus `id` and `geometry` |
| `json` instruction | "Determine whether location_a and location_b represent the same physical location. Answer with exactly: MATCH or NOT_MATCH" | "Answer with exactly: MATCH or NOT_MATCH" |
| `dspy` | Unchanged | Unchanged |

Two deliberate departures from the spec:

- **`id` is withheld.** MATCH pairs share their Overture id by construction (100% of MATCH, 0% of NOT_MATCH).
- **Coordinates are withheld.** Labels were defined by distance ("MATCH if under 5 m apart" alone scores 93.2%), and the dataset's design never shows geometry to the model.

The spec's JSON example prints nested values as `"[object Object]"`, a JavaScript display artifact, so v2 sends the real values. The longest v2 `json` prompt is 1,489 tokens, which fits every model's context window.

**Colab (v1) results** are kept in `results/kr3_v1_colab_checkpoint.csv`, with the 3,000 Gemma rows dropped as numerical failures. Build it from the Colab download, which stays untouched:

```bash
python -c "import pandas as pd; d = pd.read_csv('PATH/TO/kr3_checkpoint.csv', dtype=str, keep_default_na=False); d[d.model != 'gemma-3-4b-it'].to_csv('results/kr3_v1_colab_checkpoint.csv', index=False, lineterminator='\n')"
```

Analyze it with `python analysis.py --checkpoint results/kr3_v1_colab_checkpoint.csv`. v1 rows can't enter a v2 checkpoint.

## What differs from the notebook

Unchanged: model list, dataset construction, DSPy signature and prompt, `generate()` (chat template, BOS de-duplication, EOS set), `MAX_NEW_TOKENS=20`, greedy decoding, the parser, the checkpoint schema, resume rules, and metrics.

Changed:

- **`generate()`:** the OOM handler frees the cache on any backend (not just CUDA), the chat-template date is pinned, non-finite logits raise errors, and it notes whether a reply hit the token cap.
- **Chat-template date:** the Llama-3.2 and Granite-3.1 templates insert today's date, so inputs varied by day and time zone. Both now render 26 Jul 2024 (`CHAT_TEMPLATE_DATE` in `benchmark.py`; set to `None` for the notebook's behavior). The other eight templates are unaffected.
- **Pinned for reproducibility:** dataset (commit `c0e52a70`, sha256-checked, downloaded over HTTPS, no git needed), model revisions (`MODEL_REVISIONS`), and library versions.
- **Prompt fingerprints:** all three conditions' messages are hashed. The run refuses to start unless the machine builds them byte-identically.
- **DSPy:** runs inside `dspy.context(...)` instead of `dspy.configure(...)` per example (which fails from any thread other than the first configurer), and call history is off.
- **Checkpointing:** single-write flush with fsync, torn last record dropped, recovery from empty or torn files, per-checkpoint lock, `\n` line endings on every OS.
- **Signals:** SIGTERM, SIGHUP and SIGBREAK use the same flush-and-free path as Ctrl-C.
- **Device handling** lives in `runtime.py`. CUDA keeps `device_map="auto"`. Other backends pin the whole model to one device, so a model that doesn't fit fails loudly.
- **Opt-in:** weight deletion (`--delete-weights`) and the Hub mirror (`--hub-sync`) were always on in Colab. The notebook's Hub cell also used `CommitScheduler` without importing it; fixed.
- **Colab-only code removed:** no `google.colab` secrets or `!pip`.
- **Import order:** `import dspy` before numpy breaks pyarrow with DSPy 3.3.1, so the order is fixed.
- **Progress counts** in a `--models` run are measured against that subset.

## Layout

```
benchmark.py      experiment: models, dataset, prompts, DSPy, generate(), parser
runtime.py        backend detection, precision, model load/free
checkpoint.py     results, resume, retries, runtime log, optional Hub mirror
preflight.py      environment checks
run_benchmark.py  CLI and run loop
analysis.py       metrics tables + plot
data/             golden dataset (downloaded on first run)
results/          checkpoints (created on first run)
```
