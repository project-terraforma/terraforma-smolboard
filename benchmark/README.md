# Terraforma / SMOLBoard benchmark

This directory is the portable zero-shot place-conflation benchmark. It contains
the benchmark CLI, model adapters, standardized evaluation logic, tests, and
the local llama.cpp setup script. It intentionally excludes legacy notebooks,
old experiments, downloaded model files, local tools, and historical results.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\setup_llama.ps1
```

The benchmark dataset is external to this code bundle. Before running, provide
the Golden Dataset (or the current development Parquet file) and update the
dataset path under `datasets.overture-3k.path` in `benchmark_config.yaml`.

## Run

```powershell
.\.venv\Scripts\python.exe benchmark.py gemma-12b --limit 10
.\.venv\Scripts\python.exe benchmark.py qwen3-4b --limit 10
```

Model artifacts download automatically from their pinned Hugging Face revisions
unless `--no-download` is supplied. Outputs are written below
`results/benchmark_runs/` and are ignored by Git.

## Text input

The CLI now defaults to `--prompt text`. Dataset files remain Parquet; the
benchmark renders the configured fields as readable text before model inference.
Nested names, all addresses, and contact lists are retained. Missing values are
shown as `Not provided`. Only `exposed_fields` enter the prompt; labels, IDs,
coordinates, and pair-type metadata stay out of the model input.

Preview the actual model input without downloading a model, starting a server,
or creating a run directory (commands below run from `benchmark/`):

```bash
python benchmark.py qwen3-4b --dataset medium_pairs_geo --preview
```

For example, a record renders as:

```text
Names:
  Primary: Example Cafe
Websites:
  - https://example.com
Socials: Not provided
Emails: Not provided
Phones: Not provided
Addresses:
  - Freeform: 123 Main St
    Locality: Seattle
    Country: US
```

The text prompt uses the same instructions and six exposed fields as the JSON
baseline, so comparisons can isolate the record representation. The output
contract remains `MATCH` / `NO_MATCH`. Select the original JSON representation
with `--prompt baseline`.

```bash
# Smoke test; partial runs are not eligible for the leaderboard.
python benchmark.py qwen3-4b --dataset medium_pairs_geo --limit 10

# Full text evaluation; repeat for each configured model in the agreed run list.
python benchmark.py qwen3-4b --dataset medium_pairs_geo --run-id qwen3-4b_geo_text_v1

# Resume an interrupted run by repeating its command and run ID.
# Compare JSON using its own run directory.
python benchmark.py qwen3-4b --dataset medium_pairs_geo --prompt baseline
```

Prompt hashes now include the template, exposed fields and their order, record
format, and renderer version. Run manifests also record the format. Runs created
before this change need a new run ID because their fingerprints differ.

The default `overture-3k` dataset requires an external file; `medium_pairs_geo`
uses the existing bundled dataset. For a separately supplied dataset, configure
its path and column mappings under `datasets` in the benchmark YAML. Pair-type
labels can be mapped through `metadata_columns` for per-example output without
showing them to the model. No dataset files need to be rewritten.

The supplied server path and setup script target Windows. On another platform,
set `server_executable` to the local compatible `llama-server` binary in a custom
configuration passed with `--config`. There are currently four configured models;
the model list and run conditions must be specified before launching a larger
benchmark campaign.
