from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ModelAdapterError(RuntimeError):
    pass


def call_model(model: str, parts: list[Any]) -> str:
    """Local adapter contract for experiment runners.

    Supported modes:
    - `EXPERIMENT_MODEL_MODE=mock` returns deterministic placeholder answers.
    - `EXPERIMENT_MODEL_MODE=command` shells out to `EXPERIMENT_MODEL_COMMAND`.

    In command mode, the command receives a JSON payload on stdin:
      {"model": "...", "parts": [...]}

    The command must print a plain-text model response to stdout.
    """

    mode = os.environ.get("EXPERIMENT_MODEL_MODE", "mock").strip().lower()
    if mode == "mock":
        return _mock_response(parts)
    if mode == "command":
        return _command_response(model, parts)
    raise ModelAdapterError(f"Unsupported EXPERIMENT_MODEL_MODE: {mode}")


def _mock_response(parts: list[Any]) -> str:
    text = "\n".join(part for part in parts if isinstance(part, str)).lower()
    if "answer only: yes, no, or inconclusive" in text:
        return "INCONCLUSIVE"
    if "reply with exactly one word: yes, no, or inconclusive" in text:
        return "INCONCLUSIVE"
    if "cannot determine from this image" in text:
        return "Cannot determine from this image"
    return "INCONCLUSIVE"


def _command_response(model: str, parts: list[Any]) -> str:
    import subprocess

    command = os.environ.get("EXPERIMENT_MODEL_COMMAND", "").strip()
    if not command:
        raise ModelAdapterError("EXPERIMENT_MODEL_COMMAND is required in command mode")

    payload = json.dumps({"model": model, "parts": parts})
    timeout = float(os.environ.get("EXPERIMENT_MODEL_COMMAND_TIMEOUT", "75"))
    try:
        result = subprocess.run(
            command,
            input=payload,
            text=True,
            shell=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            cwd=str(Path(__file__).resolve().parent),
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelAdapterError(f"Model command timed out after {timeout:g}s: {command}") from exc
    if result.returncode != 0:
        raise ModelAdapterError(result.stderr.strip() or f"Command failed: {command}")
    return result.stdout.strip()
