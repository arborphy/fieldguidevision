# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""CLI for the anatomy4 conditioned experts.

Examples are in ``experiments/anatomy4_specialists/README.md``. Every command
is offline/read-only with respect to the app services: the only DB access is a
read-only ``COPY ... TO STDOUT`` through ``docker exec
arq-compose-postgres psql``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Sequence

from .contact_sheet import load_sheet_rows, render_contact_sheets
from .contracts import load_target_spec, write_jsonl
from .crops import build_crops, load_proposals, load_reviews
from .dataset import (
    build_classification_dataset,
    load_crop_manifest,
    validate_dataset,
)
from .dirr_join import load_claims, join_weak_labels
from .splits import assign_splits, apply_assignments


def _db_command(value: str | None) -> list[str] | None:
    return shlex.split(value) if value else None


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def _write(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def cmd_build_crops(args: argparse.Namespace) -> int:
    proposals = load_proposals(args.proposals)
    reviews = load_reviews(args.review) if args.review else {}
    report = build_crops(
        proposals,
        args.out,
        reviews=reviews,
        images_dir=args.images_dir,
        padding_frac=args.padding,
        padding_mode=args.padding_mode,
        min_crop_px=args.min_crop_px,
        min_detector_confidence=args.min_confidence,
        allow_unreviewed=args.allow_unreviewed,
        observations=args.observations,
        split_seed=args.split_seed,
        max_crops=args.max_crops,
    )
    payload = report.to_dict()
    payload["manifest"] = str((Path(args.out) / "crop_manifest.jsonl").resolve())
    _print(payload)
    return 0


def cmd_detect_crops(args: argparse.Namespace) -> int:
    from .detect_crops import detect_crops

    payload = detect_crops(
        weights=args.weights,
        dirr_manifest=args.dirr_manifest,
        out_dir=args.out,
        stage1_dataset=args.stage1_dataset,
        exclude_null_license=args.exclude_null_license,
        conf=args.conf,
        iou=args.iou,
        padding_frac=args.padding,
        padding_mode=args.padding_mode,
        min_crop_px=args.min_crop_px,
        max_crops_per_image=args.max_crops_per_image,
        device=args.device,
        max_vram_fraction=args.max_vram_fraction,
        imgsz=args.imgsz,
        split_seed=args.split_seed,
    )
    _print(payload)
    return 0


def cmd_join_labels(args: argparse.Namespace) -> int:
    target = load_target_spec(args.target_config)
    rows = load_crop_manifest(args.crops)
    claims = load_claims(
        claims_json=args.claims_json,
        db_command=_db_command(args.db_command),
        devo_name=args.devo,
        features=target.claim_features if args.db else None,
    )
    joined = join_weak_labels(
        rows,
        claims,
        target,
        require_anatomy_match=not args.no_anatomy_gate,
    )
    write_jsonl(args.out, joined["rows"])
    _print(joined["audit"])
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    target = load_target_spec(args.target_config)
    rows = load_crop_manifest(args.crops)
    report = validate_dataset(
        rows,
        target,
        require_labels=not args.allow_unlabeled,
        check_files=args.check_files,
        require_species_disjoint=args.species_disjoint,
        min_per_class_per_split=args.min_per_class,
        min_detector_confidence=args.min_confidence,
        allow_unlabeled=args.allow_unlabeled,
    )
    _print(report.to_dict())
    return 0 if report.ok else 1


def cmd_split(args: argparse.Namespace) -> int:
    rows = load_crop_manifest(args.crops)
    report = assign_splits(rows, mode=args.mode, seed=args.seed)
    output = apply_assignments(rows, report["assignments"])
    write_jsonl(args.out, output)
    _print({k: v for k, v in report.items() if k != "assignments"})
    return 0 if report["isolation"]["ok"] else 1


def cmd_build_dataset(args: argparse.Namespace) -> int:
    target = load_target_spec(args.target_config)
    rows = load_crop_manifest(args.crops)
    summary = build_classification_dataset(
        rows,
        target,
        args.out,
        split_mode=args.split_mode,
        seed=args.seed,
        link_mode=args.link_mode,
        min_per_class_per_split=args.min_per_class,
        min_detector_confidence=args.min_confidence,
        check_files=not args.skip_file_check,
        use_manifest_splits=args.use_manifest_splits,
        prefer_test_unseen=args.holdout_prefer_stage1_unseen,
    )
    _print(summary)
    return 0


def cmd_contact_sheet(args: argparse.Namespace) -> int:
    target = load_target_spec(args.target_config)
    rows = load_sheet_rows(crop_manifest=args.crops, dataset_root=args.dataset)
    report = render_contact_sheets(
        rows,
        target,
        args.out,
        columns=args.columns,
        rows_per_page=args.rows,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        include_unlabeled=args.include_unlabeled,
    )
    _print(report)
    return 0


def cmd_export_claims(args: argparse.Namespace) -> int:
    claims = load_claims(
        db_command=_db_command(args.db_command),
        devo_name=args.devo,
        features=args.features,
    )
    payload = {
        "schema": "arq.anatomy4_specialists.claims/v1",
        "devo_name": args.devo,
        "count": len(claims),
        "claims": [
            {
                "taxon_name": claim.taxon_name,
                "feature_name": claim.feature_name,
                "value_label": claim.value_label,
                "devo_name": claim.devo_name,
                "provenance": claim.provenance,
            }
            for claim in claims
        ],
    }
    _write(args.out, payload)
    _print({"count": len(claims), "out": str(Path(args.out).resolve())})
    return 0


def cmd_support(args: argparse.Namespace) -> int:
    """Measured label support from actual crops + claims (not a metric claim)."""

    target = load_target_spec(args.target_config)
    rows = load_crop_manifest(args.crops)
    claims = load_claims(
        claims_json=args.claims_json,
        db_command=_db_command(args.db_command),
        devo_name=args.devo,
        features=target.claim_features,
    )
    joined = join_weak_labels(rows, claims, target)
    payload = {
        "target": target.to_dict(),
        "rows": len(joined["rows"]),
        "audit": joined["audit"],
        "by_split_label": _split_label_counts(joined["rows"]),
        "by_anatomy": _count(joined["rows"], "anatomy_class"),
    }
    _write(args.out, payload)
    _print(payload)
    return 0


def _count(rows: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key))] = counts.get(str(row.get(key)), 0) + 1
    return dict(sorted(counts.items()))


def _split_label_counts(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if row.get("target_label") is None:
            continue
        split = str(row.get("observation_split"))
        label = str(row.get("target_label"))
        counts.setdefault(split, {})
        counts[split][label] = counts[split].get(label, 0) + 1
    return {split: dict(sorted(values.items())) for split, values in sorted(counts.items())}


def cmd_train(args: argparse.Namespace) -> int:
    from .trainer import dry_run, train

    overrides = {
        "device": args.device,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "seed": args.seed,
        "max_vram_fraction": args.max_vram_fraction,
        "cache": args.cache,
        "amp": args.amp,
    }
    if args.dry_run:
        _print(dry_run(args.config, overrides=overrides))
        return 0
    _print(train(args.config, overrides=overrides, output_dir=args.output_dir, resume=args.resume))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .contracts import load_run_config
    from .evaluate import evaluate_run

    config = load_run_config(args.config)
    raw = config.get("raw", config)
    training = raw.get("training", {})
    root = Path(raw.get("dataset_root") or raw.get("dataset", {}).get("root") or "")
    checkpoint = args.checkpoint or str(Path(raw.get("output_dir") or root.parent / "runs" / config["target"].target_id) / "best.pt")
    _print(
        evaluate_run(
            checkpoint,
            root,
            config["target"],
            args.out,
            device=args.device or training.get("device", "auto"),
            imgsz=args.imgsz or int(training.get("imgsz", 224)),
            batch_size=args.batch or int(training.get("batch", 32)),
            target_precision=args.target_precision
            if args.target_precision is not None
            else float(training.get("target_precision", 0.95)),
            min_coverage=args.min_coverage
            if args.min_coverage is not None
            else float(training.get("min_coverage", 0.30)),
            max_vram_fraction=float(training.get("max_vram_fraction", 0.30)),
            min_free_vram_gb=float(training.get("min_free_vram_gb", 0.5)),
            split=args.split,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anatomy4-specialists",
        description="Conditioned leaf/flower expert dataset + training tooling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect-crops", help="Stage-1 anatomy4 YOLO -> padded Stage-2 crops.")
    p.add_argument("--weights", required=True, help="Trained Stage-1 four-class anatomy4 best.pt.")
    p.add_argument("--dirr-manifest", required=True)
    p.add_argument("--stage1-dataset", help="Stage-1 dataset JSON (records in_stage1_train_val).")
    p.add_argument("--exclude-null-license", action="store_true")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.50)
    p.add_argument("--padding", type=float, default=0.18)
    p.add_argument("--padding-mode", choices=("relative", "absolute"), default="relative")
    p.add_argument("--min-crop-px", type=int, default=48)
    p.add_argument("--max-crops-per-image", type=int, default=6)
    p.add_argument("--device", default="0")
    p.add_argument("--max-vram-fraction", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split-seed", default="anatomy4-specialists-v1")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_detect_crops)

    p = sub.add_parser("build-crops", help="Materialize padded crops from reviewed detector boxes.")
    p.add_argument("--proposals", required=True, help="anatomy4 proposals JSON (images + proposals).")
    p.add_argument("--observations", help="anatomy4 manifest or JSONL supplying taxon_name/photo URLs.")
    p.add_argument("--review", help="review.csv with proposal_id,verdict.")
    p.add_argument("--out", required=True)
    p.add_argument("--images-dir", help="Fallback directory holding downloaded source images.")
    p.add_argument("--padding", type=float, default=0.18, help="Pad fraction per side (default 0.18).")
    p.add_argument("--padding-mode", choices=("relative", "absolute"), default="relative")
    p.add_argument("--min-crop-px", type=int, default=48)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--allow-unreviewed", action="store_true")
    p.add_argument("--split-seed", default="anatomy4-specialists-v1")
    p.add_argument("--max-crops", type=int)
    p.set_defaults(func=cmd_build_crops)

    p = sub.add_parser("join-labels", help="Join Dirr weak labels with the anatomy gate.")
    p.add_argument("--crops", required=True)
    p.add_argument("--target-config", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--claims-json", help="Pre-exported claims JSON.")
    source.add_argument("--db", action="store_true", help="Read claims from the local app DB.")
    p.add_argument("--devo", default="dirr")
    p.add_argument("--db-command", default=None, help="Shell string, e.g. 'docker exec arq-compose-postgres psql -U arq -d arq'.")
    p.add_argument("--no-anatomy-gate", action="store_true", help="DEBUG ONLY; violates the contract.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_join_labels)

    p = sub.add_parser("validate", help="Validate crops/labels/splits for a target.")
    p.add_argument("--crops", required=True)
    p.add_argument("--target-config", required=True)
    p.add_argument("--check-files", action="store_true")
    p.add_argument("--allow-unlabeled", action="store_true")
    p.add_argument("--species-disjoint", action="store_true")
    p.add_argument("--min-per-class", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("split", help="Assign observation- or species-disjoint splits.")
    p.add_argument("--crops", required=True)
    p.add_argument("--mode", choices=("observation", "species"), default="species")
    p.add_argument("--seed", default="anatomy4-specialists-v1")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("build-dataset", help="Write the Ultralytics classification tree.")
    p.add_argument("--crops", required=True)
    p.add_argument("--target-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split-mode", choices=("observation", "species", "species_then_observation"), default="species")
    p.add_argument("--seed", default="anatomy4-specialists-v1")
    p.add_argument("--min-per-class", type=int, default=2)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    p.add_argument("--use-manifest-splits", action="store_true")
    p.add_argument("--holdout-prefer-stage1-unseen", action="store_true",
                   help="Fill the species/observation test split from Stage-1-unseen groups first.")
    p.add_argument("--skip-file-check", action="store_true")
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("contact-sheet", help="Render GT crop contact sheets.")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--crops", help="Crop manifest JSONL.")
    source.add_argument("--dataset", help="Built dataset root (uses its manifest.jsonl split).")
    p.add_argument("--target-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--columns", type=int, default=6)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--tile-width", type=int, default=220)
    p.add_argument("--tile-height", type=int, default=200)
    p.add_argument("--include-unlabeled", action="store_true")
    p.set_defaults(func=cmd_contact_sheet)

    p = sub.add_parser("export-claims", help="Export Dirr claims (read-only DB) to JSON.")
    p.add_argument("--out", required=True)
    p.add_argument("--devo", default="dirr")
    p.add_argument("--features", nargs="+", default=None)
    p.add_argument("--db-command", default=None, help="Shell string, e.g. 'docker exec arq-compose-postgres psql -U arq -d arq'.")
    p.set_defaults(func=cmd_export_claims)

    p = sub.add_parser("support", help="Measure actual crop label support.")
    p.add_argument("--crops", required=True)
    p.add_argument("--target-config", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--claims-json")
    source.add_argument("--db", action="store_true")
    p.add_argument("--devo", default="dirr")
    p.add_argument("--db-command", default=None, help="Shell string, e.g. 'docker exec arq-compose-postgres psql -U arq -d arq'.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_support)

    p = sub.add_parser("train", help="Fine-tune one conditioned expert (YOLO26n-cls).")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir")
    p.add_argument("--device", default=None, help="auto | cuda:0 | mps | cpu")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int)
    p.add_argument("--imgsz", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-vram-fraction", type=float)
    p.add_argument("--cache", default=None, help="false | disk | ram")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="Evaluate a checkpoint on val+test.")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--target-precision", type=float, default=None)
    p.add_argument("--min-coverage", type=float, default=None)
    p.add_argument("--split", choices=("val", "test", "both"), default="both")
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
