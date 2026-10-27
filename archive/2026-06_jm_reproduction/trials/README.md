# Trials

This folder stores stepwise trial result CSVs and metadata JSON files under
`trials/artifacts/`.

The trial wrapper reads `.env` from the repo root before it starts the model
runner, so `OPENROUTER_API_KEY=...` can live there. To load the same `.env` into
your current terminal session, run:

```bash
set -a
source .env
set +a
```

Run the normal smoke test from the repo root:

```bash
uv run python scripts/run_stepwise_trial.py \
  --model 'openai/gpt-4o-mini' \
  --sample-limit 1 \
  --features key_flower_type
```

Full option version:

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

Supported feature names are currently `key_flower_type`, `key_plant_type`, and
`key_leaf_type`.

If `--out-file` is omitted, the wrapper creates an auto-generated `trial_id`
using the run time, model, sample limit, and feature list, then writes matching
CSV and metadata files:

```text
trials/artifacts/stepwise-20260624-143000-openai-gpt-4o-mini-n1-key_flower_type.csv
trials/artifacts/stepwise-20260624-143000-openai-gpt-4o-mini-n1-key_flower_type.metadata.json
```

If `--out-file` is a relative filename, it is also written under
`trials/artifacts/`. Use an absolute path only when you intentionally want to
write somewhere else. The metadata filename is always based on `trial_id`.

If `--trial-id jm-baseline-001` is provided and `--out-file` is omitted, the
wrapper writes:

```text
trials/artifacts/jm-baseline-001.csv
trials/artifacts/jm-baseline-001.metadata.json
```

Each generated CSV includes a `trial_id` column and does not include a
`model_command` column.

Each metadata JSON records the trial setup:

```json
{
  "trial_id": "jm-baseline-001",
  "command": "uv run python scripts/run_stepwise_trial.py --model openai/gpt-4o-mini --sample-limit 20 --features key_flower_type,key_plant_type,key_leaf_type --trial-id jm-baseline-001",
  "model": "openai/gpt-4o-mini",
  "sample_limit": 20,
  "features": ["key_flower_type", "key_plant_type", "key_leaf_type"],
  "mode": "command",
  "output_file": "trials/artifacts/jm-baseline-001.csv",
  "git_commit": "...",
  "python": "...",
  "package_manager": "uv"
}
```

## Recorded Trials

| Trial | Record | CSV | Metadata |
| --- | --- | --- | --- |
| Real OpenRouter smoke test, `sample-limit 1`, `key_flower_type` | `trials/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.md` | `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.csv` | `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.metadata.json` |
