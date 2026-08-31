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
