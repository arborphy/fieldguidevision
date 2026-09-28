#!/usr/bin/env python3
"""Audit every BioImages thumbnail URL used by the static browser."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def check(image: dict, timeout: int) -> dict:
    url = image["thumbnail_url"]
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "BioImagesBrowserAudit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "image_id": image["id"], "url": url, "status": response.status,
                "content_type": response.headers.get("Content-Type", ""), "error": "",
            }
    except urllib.error.HTTPError as exc:
        return {"image_id": image["id"], "url": url, "status": exc.code, "content_type": "", "error": str(exc)}
    except Exception as exc:  # network failures are recorded, not swallowed
        return {"image_id": image["id"], "url": url, "status": 0, "content_type": "", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=15)
    args = parser.parse_args()
    raw = (ROOT / "data" / "site-data.js").read_text()
    data = json.loads(raw.removeprefix("window.BIOIMAGES_DATA=").removesuffix(";\n"))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check, image, args.timeout): image["id"] for image in data["images"]}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["image_id"])

    with (ROOT / "data" / "image_url_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "url", "status", "content_type", "error"))
        writer.writeheader(); writer.writerows(rows)
    ok = [row for row in rows if row["status"] == 200 and row["content_type"].startswith("image/")]
    failures = [row for row in rows if row not in ok]
    summary = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "urls_checked": len(rows),
        "reachable_images": len(ok),
        "failures": len(failures),
        "status_counts": {str(status): sum(row["status"] == status for row in rows) for status in sorted({r["status"] for r in rows})},
        "failure_image_ids": [row["image_id"] for row in failures],
    }
    (ROOT / "data" / "image_url_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
