"""Temporary private Cloud Run GPU entrypoint for the fusion experiment.

This is intentionally a thin adapter around the exact function used by Modal.
It exists because some execution environments cannot reach Modal's API.  The
workflow deploys the service, downloads the result archive, and removes the
service again, so no GPU is left allocated after an experiment.
"""

from __future__ import annotations

import os
import traceback

from flask import Flask, Response, jsonify, request

from dual_encoder_fusion.modal_run_experiment import run_experiment


app = Flask(__name__)


@app.errorhandler(Exception)
def unhandled_exception(error: Exception):
    """Return diagnostics to the already authenticated one-shot caller."""
    app.logger.exception("Dual-encoder experiment failed")
    return jsonify(
        error=type(error).__name__,
        message=str(error),
        traceback=traceback.format_exc(),
    ), 500


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/run")
def run():
    expected = os.environ.get("EXPERIMENT_TOKEN", "")
    supplied = request.headers.get("X-Experiment-Token", "")
    if not expected or supplied != expected:
        return jsonify(error="unauthorized"), 401

    smoke = request.args.get("smoke", "true").lower() in {"1", "true", "yes"}
    rebuild_cache = request.args.get("rebuild_cache", "false").lower() in {
        "1", "true", "yes"
    }
    payload = run_experiment.get_raw_f()(
        smoke=smoke,
        rebuild_cache=rebuild_cache,
        commit_modal_cache=False,
    )
    return Response(
        payload,
        mimetype="application/gzip",
        headers={"Content-Disposition": "attachment; filename=dual-encoder-results.tar.gz"},
    )
