"""Local real-photo QA; originals are read-only and no model calls are made."""
import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageDraw

DEMO_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fgv-demo.py"
spec = importlib.util.spec_from_file_location("fgv_demo", DEMO_PATH)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    photos = sorted(p for p in args.source.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    annotations = json.loads(args.annotations.read_text()) if args.annotations else {}
    release = demo.ROOT.parent / "arq-refdata/gobotany_extract/outputs/api_pilot/gobotany_release.json"
    rows = []
    sheet = Image.new("RGB", (1200, ((len(photos) + 4) // 5) * 205), "#eee9de")
    draw = ImageDraw.Draw(sheet)
    for index, photo in enumerate(photos, 1):
        start = time.perf_counter()
        row = {"index": index, "file": photo.name}
        try:
            with Image.open(photo) as original:
                row.update(size=list(original.size), exif_orientation=original.getexif().get(274, 1))
                oriented = ImageOps.exif_transpose(original).convert("RGB")
                thumb = ImageOps.contain(oriented, (230, 165))
                col, line = (index - 1) % 5, (index - 1) // 5
                sheet.paste(thumb, (col * 240 + 5, line * 205 + 5))
                draw.text((col * 240 + 5, line * 205 + 173), f"{index:02d} {photo.name[4:23]}", fill="black")
                if args.inventory:
                    row["state"] = "inventoried"
                else:
                    supplied = annotations.get(photo.name, {})
                    evidence, image, masks = demo.classify_photo({"image": str(photo), **supplied}, [release])
                    reference = np.asarray(oriented)
                    row["orientation_size_matches"] = reference.shape == image.shape
                    assert row["orientation_size_matches"], "EXIF-oriented dimensions differ"
                    row["orientation_mean_pixel_difference"] = round(float(np.abs(
                        reference.astype(np.int16) - cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.int16)
                    ).mean()), 4)
                    assert row["orientation_mean_pixel_difference"] < 3, "Decoded pixels differ materially from EXIF-normalized reference"
                    for values in evidence["measurements"].values():
                        assert all(sum(hist) == values["pixel_count"] for hist in values["rgb_histogram"].values())
                    assert evidence["taxon_assignment"] is None and evidence["confirmed_assertions"] == []
                    repeat, _, _ = demo.classify_photo({"image": str(photo), **supplied}, [release])
                    serialized = json.dumps(evidence, sort_keys=True, indent=2) + "\n"
                    assert serialized == json.dumps(repeat, sort_keys=True, indent=2) + "\n", "Measurements differ on repeat"
                    destination = args.output / f"{index:02d}"
                    destination.mkdir()
                    (destination / "input.json").write_text(json.dumps({"image": str(photo), **supplied}, indent=2))
                    (destination / "evidence.json").write_text(serialized)
                    proposals = {"state": "not_requested", "notices": []}
                    (destination / "proposals.json").write_text(json.dumps(proposals))
                    (destination / "report.html").write_text(demo.render_report(evidence, image, masks, proposals))
                    row.update(state="passed", evidence_sha256=hashlib.sha256(serialized.encode()).hexdigest(),
                               mean_rgb=evidence["measurements"]["scene"]["mean_rgb"],
                               edge_fraction=evidence["measurements"]["scene"]["canny_edge_fraction"],
                               regions=len(evidence["regions"]), checklist_count=len(evidence["inspection_checklist"]),
                               tags=evidence["organ_tags"], report=f"{index:02d}/report.html")
                    del image, masks, reference, repeat, evidence
        except Exception as exc:
            row.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        row["seconds"] = round(time.perf_counter() - start, 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
    sheet.save(args.output / "contact.jpg")
    skipped = [p.name for p in sorted(args.source.iterdir()) if p.suffix.lower() == ".dng"]
    summary = {"source": str(args.source), "network_calls": 0, "photos": rows, "skipped_dng": skipped,
               "annotation_provenance": "Assistant visual QA labels/regions, provisional and not director field tags" if annotations else "No organ annotations supplied",
               "passed": sum(row["state"] == "passed" for row in rows),
               "failed": sum(row["state"] == "failed" for row in rows)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    links = "".join(f'<li>{row["index"]:02d} <a href="{row.get("report", "contact.jpg")}">{html.escape(row["file"])}</a>: '
                    f'{html.escape(row["state"])}; {row.get("tags", [])}; {row.get("regions", 0)} regions</li>' for row in rows)
    (args.output / "index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>Tuckaway Local QA</title>'
        '<style>body{font:16px/1.6 system-ui;margin:24px;background:#f4f1e9}img{max-width:100%}li{overflow-wrap:anywhere}</style>'
        '<h1>Tuckaway: Local Photo QA</h1><p>No uploads, no taxon assignments. '
        'Any organ labels/regions are provisional assistant QA markup, not confirmed field annotations.</p>'
        '<img src="contact.jpg" alt="Numbered Tuckaway photo contact sheet"><ol>' + links + '</ol></html>')
    print(json.dumps({"passed": summary["passed"], "failed": summary["failed"], "output": str(args.output)}))
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
