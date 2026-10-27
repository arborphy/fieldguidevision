from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "newcomb_wildflower_guide" / "experiment_repro"
ARTIFACT_DIR = ROOT / "trials" / "artifacts"

sys.path.insert(0, str(ROOT / "scripts" / "adapters"))
from openrouter_models import resolve_model_alias  # noqa: E402


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return result[:120] or "none"


def split_features(features: str) -> list[str]:
    return [feature.strip() for feature in features.split(",") if feature.strip()]


def metadata_sample_limit(sample_limit: str) -> int | str:
    try:
        return int(sample_limit)
    except ValueError:
        return sample_limit


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def command_text(raw_args: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in raw_args)
    return f"uv run python scripts/run_stepwise_trial.py {quoted_args}".rstrip()


def default_trial_id(args: argparse.Namespace, model: str) -> str:
    if args.trial_id:
        return args.trial_id

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    sample_slug = f"n{args.sample_limit}" if args.sample_limit else "all"
    feature_slug = slug("-".join(split_features(args.features)))
    return f"stepwise-{timestamp}-{slug(model)}-{sample_slug}-{feature_slug}"


def output_path(args: argparse.Namespace, trial_id: str) -> Path:
    if args.out_file:
        path = Path(args.out_file).expanduser()
        return path if path.is_absolute() else ARTIFACT_DIR / path
    return ARTIFACT_DIR / f"{slug(trial_id)}.csv"


def metadata_path_for(trial_id: str) -> Path:
    return ARTIFACT_DIR / f"{slug(trial_id)}.metadata.json"


def write_metadata(
    *,
    metadata_file: Path,
    trial_id: str,
    raw_args: list[str],
    model: str,
    args: argparse.Namespace,
    output_file: Path,
) -> None:
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trial_id": trial_id,
        "command": command_text(raw_args),
        "model": model,
        "sample_limit": metadata_sample_limit(args.sample_limit),
        "features": split_features(args.features),
        "mode": args.mode,
        "output_file": relative_or_absolute(output_file),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "package_manager": "uv",
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run a JM stepwise trial with concise options.")
    parser.add_argument("--model", default="openrouter/free", help="OpenRouter model id or local alias openrouter/free.")
    parser.add_argument("--sample-limit", default="1", help="Number of sample rows to run.")
    parser.add_argument("--features", default="key_flower_type", help="Comma-separated feature list.")
    parser.add_argument("--workers", default="1", help="Thread worker count.")
    parser.add_argument("--timeout", default="75", help="Seconds before one model call is treated as timed out.")
    parser.add_argument("--mode", choices=["command", "mock"], default="command", help="Model adapter mode.")
    parser.add_argument(
        "--out-file",
        default="",
        help="Optional output CSV filename. Relative paths are written under trials/artifacts.",
    )
    parser.add_argument(
        "--trial-id",
        default="",
        help="Optional trial id. Defaults to an auto-generated id based on time, model, sample limit, and features.",
    )
    raw_args = sys.argv[1:]
    if any(arg.strip() == "" for arg in raw_args):
        parser.error(
            "received a whitespace-only argument. If you used a multiline shell command, "
            "make sure each line-continuation backslash is the final character on its line."
        )
    args = parser.parse_args(raw_args)

    model = args.model
    if args.mode == "command":
        model = resolve_model_alias(model, require_image=True)
    trial_id = default_trial_id(args, model)
    out_file = output_path(args, trial_id)
    metadata_file = metadata_path_for(trial_id)

    env = os.environ.copy()
    env["EXPERIMENT_MODEL_MODE"] = args.mode
    env["EXPERIMENT_MODELS"] = model
    env["EXPERIMENT_SAMPLE_LIMIT"] = args.sample_limit
    env["EXPERIMENT_FEATURES"] = args.features
    env["EXPERIMENT_NUM_WORKERS"] = args.workers
    env["EXPERIMENT_MODEL_COMMAND_TIMEOUT"] = args.timeout
    env["EXPERIMENT_ARTIFACT_DIR"] = str(ARTIFACT_DIR)
    env["EXPERIMENT_OUT_FILE"] = str(out_file)
    env["EXPERIMENT_TRIAL_ID"] = trial_id
    env["OPENROUTER_TIMEOUT"] = args.timeout

    if args.mode == "command":
        env["EXPERIMENT_MODEL_COMMAND"] = "uv run python ../../scripts/adapters/openrouter_command_adapter.py"

    print("Running JM stepwise trial")
    print(f"  trial id: {trial_id}")
    print(f"  model: {model}")
    print(f"  sample limit: {args.sample_limit}")
    print(f"  features: {args.features}")
    print(f"  workers: {args.workers}")
    print(f"  timeout: {args.timeout}s per model call")
    print(f"  output: {relative_or_absolute(out_file)}")
    print(f"  metadata: {relative_or_absolute(metadata_file)}")
    sys.stdout.flush()

    result = subprocess.run(
        ["uv", "run", "python", "run_stepwise_local.py"],
        cwd=EXPERIMENT_DIR,
        env=env,
        check=False,
    )
    write_metadata(
        metadata_file=metadata_file,
        trial_id=trial_id,
        raw_args=raw_args,
        model=model,
        args=args,
        output_file=out_file,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
