# Multi-Model Classification System

The loader and classification pipeline have regression tests, including a
small local model. The four default models have not yet been tested end-to-end
on this dataset. Review a small run before relying on its results.

This system runs medium pair classifications through multiple HuggingFace language models and generates consensus datasets.

## Overview

- **Input**: `datasets/geo/medium_pairs.parquet`
- **Output**:
  - `datasets/geo/multi_model_results/consensus/agreed_pairs.parquet` - All models unanimous
  - `datasets/geo/multi_model_results/consensus/disagreed_pairs.parquet` - At least one model disagrees

## Default Models

The defaults are four instruction-tuned models from different model families.
Their Hugging Face model repositories were ungated when checked; access policies
can change. Suitability for this dataset still needs empirical evaluation.

1. `Qwen/Qwen2.5-7B-Instruct`
2. `mistralai/Mistral-7B-Instruct-v0.3`
3. `allenai/OLMo-2-1124-7B-Instruct`
4. `microsoft/Phi-3.5-mini-instruct`

## Installation

```powershell
python -m pip install -r multi_model_classification\requirements.txt
```

For CUDA int8 inference, also install a compatible `bitsandbytes` build:
`python -m pip install bitsandbytes`. `--no-int8` disables quantization. CPU
uses float32 without bitsandbytes and can require substantial RAM. Check
`python -c "import torch; print(torch.cuda.is_available())"` for CUDA support.

## Usage

### Basic Usage (Default Models)

```bash
python classify_and_merge.py
```

### Custom Models

```bash
python classify_and_merge.py --models Qwen/Qwen2.5-7B-Instruct mistralai/Mistral-7B-Instruct-v0.3
```

### With Limit (for testing)

```bash
python classify_and_merge.py --limit 100
```

### Sequential vs. Parallel

```bash
# Sequential (default, one model at a time)
python classify_and_merge.py --max-workers 1

# Parallel (run 2 models concurrently)
python classify_and_merge.py --max-workers 2
```

### CPU-Only (slow, but works without GPU)

```bash
python classify_and_merge.py --device cpu
```

### Resume vs. Fresh Start

```bash
# Resume from existing checkpoints (default)
python classify_and_merge.py

# Start fresh, clear v2 checkpoints for selected models
python classify_and_merge.py --no-resume
```

### Disable Int8 Quantization

```bash
python classify_and_merge.py --no-int8
```

## Output Structure

```
datasets/geo/multi_model_results/
├── Qwen_Qwen2.5-7B-Instruct/
│   ├── pairtype_checkpoint_v2.jsonl
│   ├── medium_pairs_typed.parquet
│   └── pairtype_stats.json
├── mistralai_Mistral-7B-Instruct-v0.3/
│   ├── pairtype_checkpoint_v2.jsonl
│   ├── medium_pairs_typed.parquet
│   └── pairtype_stats.json
├── ... (other models)
└── consensus/
    ├── agreed_pairs.parquet           ← All 3+ models agree
    ├── disagreed_pairs.parquet        ← At least one disagrees
    ├── agreement_summary.json
    └── category_breakdown.json
```

## Classification Categories

1. **parent_child** - Hierarchical relationships
2. **same_name_different_address** - Chain branches, different locations
3. **typo_differences** - Same place, typo/formatting differences
4. **incomplete_data** - Missing key fields
5. **unsolvable_conflicting** - Ambiguous/conflicting evidence
6. **languages** - Translation/transliteration differences
7. **other** - None of the above

## Output Datasets

### agreed_pairs.parquet
Columns include:
- All original pair fields (names, addresses, etc.)
- `llm_pair_type` - The first selected model's classification
- `{model}_type` - Classification from each model
- `majority_type` - Unanimous classification
- `confidence_score` - Fraction of models agreeing (not calibrated confidence)
- `agreement_count` - Number of models (equal to total models)

### disagreed_pairs.parquet
Columns include:
- All original pair fields
- `{model}_type` - Classification from each model
- `majority_type` - Most common classification, null if tied
- `confidence_score` - Fraction of models agreeing on the most common label
- `agreement_count` - How many models predicted majority
- `disagreement_pattern` - e.g., "Llama-3:typo vs Mixtral:other vs Qwen:typo"

## Statistics

After running, check:
- `agreement_summary.json` - Overall agreement rates
- `category_breakdown.json` - Per-category agreement analysis

## Tips

1. **First run**: Start with `--limit 5` and inspect results.
2. **GPU memory**: Use one worker; use `--no-int8` only if you have sufficient memory.
   The `--batch-size` option is currently reserved and does not change inference.
3. **Checkpoints**: Version 2 checkpoints include prompt and settings hashes.
   Legacy checkpoints are not reused. A changed prompt or settings requires
   `--no-resume`, which clears selected models' version 2 checkpoints.
   Model revisions are not pinned; avoid resuming after model weights change.
4. **Model download**: Each model requires several GB of storage and substantial
   RAM or VRAM. Runtime and memory usage depend on hardware and configuration.
5. Invalid model responses stop the run; they are not classified as `other`.
   Agreement is not ground truth and should be checked by reviewers.
