# Reproduction Notes

These notes describe the current environment and commands for reproducing the
Newcomb/JM stepwise experiment pipeline.

## Environment

- Repository root: `/Users/lindseyma/Documents/GitHub/arborphy`
- Package manager: `uv`
- Observed `uv` version: `uv 0.9.0`
- Project Python requirement: `>=3.12`
- Python recorded by the real smoke-test metadata: `3.12.10`
- System `python3` observed outside `uv`: `3.14.0`

Use `uv run ...` for experiment commands so the recorded Python environment is
the one used by the runner.

## Install / Sync

From the repo root:

```bash
uv sync
```

The project dependencies are declared in `pyproject.toml`:

- `duckdb`
- `pandas`

## Environment Variables

The wrapper `scripts/run_stepwise_trial.py` reads `.env` from the repo root
before starting the model runner.

Required for real OpenRouter calls:

```bash
OPENROUTER_API_KEY=...
```

Optional OpenRouter settings:

```bash
OPENROUTER_FREE_MODEL=...
OPENROUTER_MAX_TOKENS=200
OPENROUTER_TEMPERATURE=0
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_REFERER=https://local.codex
OPENROUTER_TITLE=JM reproduction
```

To load `.env` into the current terminal session:

```bash
set -a
source .env
set +a
```

This is optional for the main trial wrapper because it loads `.env`
automatically.

## Data Paths

Generated JM-style runner inputs:

- `newcomb_wildflower_guide/experiment_repro/output/feature_value_pairs.csv`
- `newcomb_wildflower_guide/experiment_repro/output/references.csv`
- `newcomb_wildflower_guide/experiment_repro/output/newcomb_preprocessed.csv`
- `newcomb_wildflower_guide/experiment_repro/output/sample.csv`
- `newcomb_wildflower_guide/experiment_repro/output/illustration_paths.json`

Source extraction tables:

- `newcomb_wildflower_guide/extracted_vocabulary/parquet/`

Observation source:

- `observations_pound_ridge/ward_pound_ridge_species.csv`

Trial outputs:

- `trials/artifacts/*.csv`
- `trials/artifacts/*.metadata.json`

## Build Inputs

The JM-style inputs can be rebuilt from the Newcomb parquet files and
observation CSV:

```bash
cd newcomb_wildflower_guide/experiment_repro
uv run python build_inputs.py
```

Current generated input summary:

- `feature_value_pairs.csv`: 76 rows
- `references.csv`: 76 rows
- `newcomb_preprocessed.csv`: 781 rows
- `sample.csv`: 288 rows
- matched illustration paths: 12

## Smoke Test Command

Run from the repo root:

```bash
uv run python scripts/run_stepwise_trial.py \
  --model 'openai/gpt-4o-mini' \
  --sample-limit 1 \
  --features key_flower_type
```

The current real API smoke test was run with:

```bash
uv run python scripts/run_stepwise_trial.py --model openai/gpt-4o-mini --sample-limit 1 --features key_flower_type
```

It produced:

- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.csv`
- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.metadata.json`

## Full Trial Command Shape

Do not run this at large sample sizes until OpenRouter access and cost are
approved.

```bash
uv run python scripts/run_stepwise_trial.py \
  --model openrouter/free \
  --sample-limit 20 \
  --features key_flower_type,key_plant_type,key_leaf_type \
  --workers 4 \
  --timeout 120 \
  --mode command \
  --out-file stepwise_trial_result.csv \
  --trial-id trial_001
```

Supported features in the current runner:

- `key_flower_type`
- `key_plant_type`
- `key_leaf_type`

## Model Adapter

Wrapper:

- `scripts/run_stepwise_trial.py`

Local runner:

- `newcomb_wildflower_guide/experiment_repro/run_stepwise_local.py`

Command adapter:

- `scripts/adapters/openrouter_command_adapter.py`

Adapter contract:

- Runner sends JSON on stdin:
  `{"model": "...", "parts": [...]}`
- `parts` can contain text strings and image dictionaries like
  `{"image": "https://..."}`.
- Adapter converts this to OpenRouter chat-completions message content.
- Adapter prints the model's text response to stdout.

OpenRouter request settings:

- `temperature`: defaults to `0`
- `max_tokens`: defaults to `200`
- timeout: controlled by `--timeout` / `OPENROUTER_TIMEOUT`
- seed: not currently set

For each observation-feature pair, the stepwise runner makes one P1 visibility
call. If P1 parses as `YES`, it makes a second P2 multiple-choice call.

## Evaluation Fields

The output CSV includes:

- `true_value`: expected Newcomb key value from `newcomb_preprocessed.csv`
- `p1_raw`, `p1_parsed`: raw and parsed visibility answer
- `p2_raw`, `p2_parsed`: raw and parsed multiple-choice answer
- `feature_correct`: `True` when the parsed answer matches the expected value
  or the model is inconclusive
- `committed`: `True` when the model committed to a concrete P2 value
- `trial_id`: trial identifier used to link CSV rows to metadata

## Common Errors

### Multiline Command Breaks at `--features`

Symptom:

```text
run_stepwise_trial.py: error: unrecognized arguments:
zsh: command not found: --features
```

Cause: a trailing space after a line-continuation backslash.

Fix: make sure each `\` is the final character on its line, or use a one-line
command.

### Missing API Key

Symptom:

```text
OpenRouter adapter error: OPENROUTER_API_KEY is required
```

Fix: add `OPENROUTER_API_KEY=...` to `.env` at the repo root, or export it in
the shell.

### Unsupported Feature

Symptom:

```text
Unsupported EXPERIMENT_FEATURES: ...
```

Fix: use one or more currently supported features:
`key_flower_type,key_plant_type,key_leaf_type`.

