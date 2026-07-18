#!/usr/bin/env python3
"""Build an immutable Fieldguidevision manifest from local observation media."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "arq.local-photo-manifest/v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build_manifest(catalogue: dict, media_root: Path) -> dict:
    frames = []
    for observation in sorted(catalogue["observations"], key=lambda item: item["arborphy_id"]):
        photo_url = observation.get("photo_url")
        if not photo_url:
            frames.append({"observation_id": observation["arborphy_id"], "status": "no_photo"})
            continue
        filename = photo_url.rsplit("/", 1)[-1]
        local_path = media_root / observation["arborphy_id"] / filename
        if not local_path.is_file():
            frames.append({"observation_id": observation["arborphy_id"], "status": "missing_media", "image_url": photo_url, "local_path": str(local_path)})
            continue
        frames.append({
            "observation_id": observation["arborphy_id"], "status": "ready", "image_url": photo_url,
            "local_path": str(local_path.resolve()), "image_sha256": _sha256(local_path),
            "byte_size": local_path.stat().st_size, "latitude": observation.get("latitude"),
            "longitude": observation.get("longitude"), "captured_at": observation.get("time_observed_at"),
            "session_id": observation.get("session_id"),
        })
    return {
        "schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_area": {"instance_id": catalogue["area"]["instance_id"], "name": catalogue["area"].get("name")},
        "summary": {"observations": len(frames), "ready": sum(item["status"] == "ready" for item in frames), "non_ready": sum(item["status"] != "ready" for item in frames)},
        "frames": frames,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalogue", type=Path)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Manifest already exists: {args.output}")
    manifest = build_manifest(json.loads(args.catalogue.read_text()), args.media_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
