# Terraforma / SMOLBoard

The portable standardized place-conflation benchmark is in [benchmark](benchmark/README.md).

## Setup

Run these commands from the repository root in PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r agent\requirements.txt
python -m pip install -r benchmark\requirements.txt
```

If PowerShell activation is blocked, use the virtual-environment interpreter
directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -r agent\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r benchmark\requirements.txt
```

On Linux or macOS, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r agent/requirements.txt
python -m pip install -r benchmark/requirements.txt
```

On Windows Command Prompt (`cmd.exe`), run:

```bat
py -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r agent\requirements.txt
python -m pip install -r benchmark\requirements.txt
```

Activation is optional. If activation does not work, use the environment's
Python executable directly. On Linux/macOS use `.venv/bin/python`; on Windows
use `.venv\Scripts\python.exe`.

## Agent 1: Add A Model

`agent\agent.py` is a LangGraph workflow that searches Hugging Face for a
matching GGUF repository, chooses a quantization file, and adds a model entry
under `models:` in `benchmark\benchmark_config.yaml`. It records the pinned
revision, filename, expected file size, local model path, and adapter. It does
not download the model.

The first argument is the Hugging Face model ID. The second is the YAML config
to update:

```powershell
.\.venv\Scripts\python.exe agent\agent.py google/gemma-4-12B-it benchmark\benchmark_config.yaml
```

For example:

```powershell
.\.venv\Scripts\python.exe agent\agent.py Qwen/Qwen3-4B benchmark\benchmark_config.yaml
```

The model ID must use `author/model-name` format. The generated config key is
derived from the model name, so `Qwen/Qwen3-4B` becomes `qwen3-4b`. The agent
prefers `Q4_K_M`, then `Q4_0`, `Q5_K_M`, and `Q8_0`. If the key already exists,
its entry is replaced. A private Hugging Face repository may require:

```powershell
 huggingface-cli login
```

## Agent 2: Run A Benchmark

The benchmark agent CLI wraps the benchmark runner for one configured model.
Use the model key from the YAML file, not the Hugging Face model ID. Run it
from the `benchmark` directory so the `agents` package resolves correctly:

```powershell
Push-Location benchmark
..\.venv\Scripts\python.exe -m agents.cli gemma-12b --dataset micro_pairs --limit 10
Pop-Location
```

Available options include:

- `--dataset`: a dataset key under `datasets:` in `benchmark_config.yaml`.
- `--prompt`: a prompt key under `prompts:`; defaults to `baseline`.
- `--limit N`: run a balanced sample of `N` examples. `N` must be at least 2.
	Omit it to run the complete configured dataset.

The lower-level benchmark command is equivalent and supports additional
options such as `--no-download`, `--batch-size`, `--run-id`, and `--config`:

```powershell
.\.venv\Scripts\python.exe benchmark\benchmark.py gemma-12b `
	--dataset micro_pairs `
	--limit 10 `
	--config benchmark\benchmark_config.yaml
```

Results are written to `benchmark\results\benchmark_runs\`. A run produces
`run_result.json`, `metrics.json`, predictions, and a model log. The model file
is downloaded automatically from the pinned Hugging Face revision unless
`--no-download` is supplied.

## Publish Results To The Website

`benchmark\agents\exporter.py` converts a completed benchmark result into the
JSON row consumed by both `docs\leaderboard.html` and `docs\compare.html`.
It updates `docs\data\leaderboard.json`, removes an existing row for the same
model, and preserves unrelated existing rows.

Publish the latest run for a configured model without looking up its timestamp:

```powershell
.\.venv\Scripts\python.exe benchmark\agents\exporter.py `
	--model gemma-4-12b-it `
	--replace-model GEMMA-12B
```

`--model` searches `benchmark\results\benchmark_runs\` for the latest run
directory containing that model name. To publish a specific result instead:

```powershell
.\.venv\Scripts\python.exe benchmark\agents\exporter.py `
	benchmark\results\benchmark_runs\<run-folder>\run_result.json `
	--replace-model GEMMA-12B
```

Useful exporter options:

- `--replace-model NAME`: remove an old website row before adding the new row;
	repeat the option to remove multiple names.
- `--output PATH`: write to another JSON file instead of
	`docs\data\leaderboard.json`.
- `--results-root PATH`: change where `--model` searches for benchmark runs.

The exporter accepts partial runs as well as full runs. A partial result is
useful for development, but should be labeled as such before treating it as a
final leaderboard result.

## Start The Website

Start the local JSON-backed website from the repository root:

```powershell
.\.venv\Scripts\python.exe website_server.py
```

Open:

- [Leaderboard](http://localhost:8000/leaderboard.html)
- [Compare graphs](http://localhost:8000/compare.html)

Both pages read `docs\data\leaderboard.json`. Restart the server after changing
`website_server.py`; refresh the browser after publishing new JSON.
