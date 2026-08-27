"""Finalize an agentic-tagging artifact for import into the local arq-api evidence pool.

The batch import (``POST /api/fieldguidevision/batches/import``) requires:

  - ``input_manifest.source_area.instance_id`` == the target area, and
  - ``input_manifest.frames[].{observation_id, image_sha256}``, and
  - every referenced ``observation_id`` to already exist in the local DB under
    ``arborphy_id`` (which uses the ``inat-{id}`` convention).

This utility rewrites an existing ``notice_batch.json`` to the ``inat-`` prefix,
drops observations not present in the DB (recorded as skipped), and injects the
manifest — so a completed run can be imported without re-running the models.

Usage::

    uv run python scripts/finalize_for_import.py \
        --artifact scratch/p17-strict/notice_batch.json \
        --area-instance-id inst-osm-way-60914147 \
        --db-check   # verify observation ids exist in the local DB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _existing_ids(ids: list[str]) -> set[str]:
    """Return the subset of ids present in the local observations table.

    Uses the running ``arq-postgres`` docker container's psql so fieldguidevision
    needs no extra Postgres driver dependency.
    """
    import subprocess

    if not ids:
        return set()
    id_list = ", ".join("'" + i.replace("'", "''") + "'" for i in ids)
    sql = (
        "SELECT arborphy_id FROM observations "
        f"WHERE arborphy_id IN ({id_list}) AND deleted_at IS NULL;"
    )
    result = subprocess.run(
        ["docker", "exec", "arq-postgres", "psql", "-U", "arq", "-d", "arq", "-t", "-A", "-c", sql],
        capture_output=True, text=True, check=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize an agentic-tagging artifact for import.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--area-instance-id", required=True)
    parser.add_argument("--obs-id-prefix", default="inat-")
    parser.add_argument("--db-check", action="store_true", help="Drop obs not in the local DB.")
    parser.add_argument("--output", type=Path, default=None, help="Default: overwrite --artifact.")
    args = parser.parse_args()

    artifact = json.loads(args.artifact.read_text())
    prefix = args.obs_id_prefix

    def prefixed(oid: str) -> str:
        return oid if oid.startswith(prefix) else f"{prefix}{oid}"

    # Rewrite observation ids everywhere they appear.
    observations = artifact.get("observations", [])
    for obs in observations:
        obs["observation_id"] = prefixed(obs["observation_id"])
        for ann in obs.get("annotations", []):
            ann["observation_id"] = obs["observation_id"]

    # Rebuild the manifest with per-frame hashes.
    frames = [
        {"observation_id": obs["observation_id"], "image_sha256": obs.get("image_sha256", "")}
        for obs in observations
        if obs.get("image_sha256")
    ]

    skipped: list[str] = []
    if args.db_check:
        ids = [f["observation_id"] for f in frames]
        present = _existing_ids(ids)
        before = len(observations)
        observations = [o for o in observations if o["observation_id"] in present]
        frames = [f for f in frames if f["observation_id"] in present]
        skipped = sorted(set(ids) - present)
        if skipped:
            print(f"Skipping {len(skipped)} observations not in local DB: {skipped}")
        print(f"Kept {len(observations)}/{before} observations present in local DB.")

    artifact["observations"] = observations
    artifact["input_manifest"] = {
        "source_area": {"instance_id": args.area_instance_id},
        "frames": frames,
    }

    out = args.output or args.artifact
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out} with input_manifest for area {args.area_instance_id} "
          f"({len(frames)} hashed frames).")


if __name__ == "__main__":
    main()
