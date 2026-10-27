# Stepwise OpenRouter Smoke Test

## Summary

This trial confirms that the current Newcomb/JM stepwise pipeline runs end to
end with a real OpenRouter model call:

input data -> prompt construction -> OpenRouter adapter -> raw model response ->
parser -> evaluation fields -> artifact CSV and metadata JSON.

## Trial Metadata

| Field | Value |
| --- | --- |
| Trial ID | `stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type` |
| Model | `openai/gpt-4o-mini` |
| Mode | `command` |
| Sample limit | `1` |
| Features | `key_flower_type` |
| Package manager | `uv` |
| Python | `3.12.10` |
| Git commit recorded in metadata | `8a41cae034dbf49af49868dc0b25231bbccaa650` |

## Command

```bash
uv run python scripts/run_stepwise_trial.py --model openai/gpt-4o-mini --sample-limit 1 --features key_flower_type
```

## Artifacts

- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.csv`
- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.metadata.json`

## Input Row

| Field | Value |
| --- | --- |
| Observation ID | `7089628` |
| Photo URL | `https://inaturalist-open-data.s3.amazonaws.com/photos/9108809/medium.jpeg` |
| Species | `Monarda fistulosa` |
| Feature | `key_flower_type` |
| Expected value | `Irregular Flowers` |

## Result

| Field | Value |
| --- | --- |
| P1 raw | `YES` |
| P1 parsed | `YES` |
| P2 raw | `Cannot determine from this image.` |
| P2 parsed | `INCONCLUSIVE` |
| `feature_correct` | `True` |
| `committed` | `False` |

## Interpretation

The pipeline is running with the real model adapter. The model found the target
feature visible in P1, then declined to commit to a concrete flower type in P2.
The current evaluation rule treats inconclusive P2 answers as correct but not
committed.

This is enough to report that the experiment is running end to end for
`sample-limit 1`. It is not yet evidence about accuracy or model quality.

## Next Step

Do not scale this trial until OpenRouter access and expected cost are approved.
The next low-risk run should still be small, for example one feature with
`sample-limit 3` or the three supported features with `sample-limit 1`.

