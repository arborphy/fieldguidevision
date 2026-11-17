# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""CLI: python -m experiments.anatomy4_detector.cli <command> ...

Commands
--------
validate   audit a materialized normalized-xyxy dataset
materialize  audit proposals.json + review.csv -> contract dataset
import-blind frozen blind benchmark -> test-only eval dataset
combine    merge mixed reviewed sources -> train/val dataset
coarse-multi cheap cross-organ tags -> coarse-multi/auto-plantae.jsonl
prepare    write the Ultralytics cx-cy-w-h training view
train      train yolo26n with a VRAM ceiling (``--dry-run`` resolves only)
evaluate   held-out predictions -> metrics, JSONL, overlays, contact sheets
export     ONNX with the arq-api contract, then verify it on onnxruntime
pipeline   validate -> prepare -> train -> evaluate -> export -> manifest
publish    register the run with arq-api (``--dry-run`` prints planned calls)
smoke      tiny synthetic end-to-end (``--dry-run`` validates/loads only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=1, default=str))


def _configure() -> None:
    # Must precede any torch/CUDA initialization.
    from experiments.anatomy4_detector.train import configure_process_env

    configure_process_env()


def _audit_or_exit(dataset: str, require_test: bool):
    from experiments.anatomy4_detector.dataset import validate_dataset

    audit = validate_dataset(dataset, require_test=require_test)
    return audit


def command_validate(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.dataset import (
        render_audit_markdown,
        write_audit,
    )

    audit = _audit_or_exit(args.dataset, not args.no_require_test)
    if args.json_out:
        write_audit(audit, args.json_out)
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(render_audit_markdown(audit))
    _print(
        {
            "status": audit["status"],
            "blocking": audit["blocking"],
            "split_isolation": audit["split_isolation"],
            "totals": audit["totals"],
            "splits": audit["splits"],
            "support": audit["support"],
            "errors": audit["errors"],
            "warnings": audit["warnings"],
            "json_out": args.json_out,
        }
    )
    return 0 if not audit["blocking"] else 2


def command_materialize(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.materialize import materialize

    result = materialize(
        proposals_path=args.proposals,
        review_path=args.review,
        out_dir=args.out,
        manifest_path=args.manifest,
        images_dir=args.images_dir,
        classes=args.classes.split(",") if args.classes else None,
        seed=args.seed,
        negative_policy=args.negative_policy,
        negative_images=args.negative_image,
        negative_file=args.negative_file,
        background_review_path=args.background_review,
        require_fully_reviewed=args.require_fully_reviewed,
        allow_missing_images=args.allow_missing_images,
        copy_mode=args.copy_mode,
        dedupe_exact=not args.keep_duplicate_boxes,
        verify_sha256=not args.no_verify_sha256,
        require_test=not args.no_require_test,
        require_val=not args.no_require_val,
    )
    _print(
        {
            "status": result["status"],
            "blocking": result["blocking"],
            "split_isolation": result["split_isolation"],
            "counts": result["counts"],
            "partial_labeling": result["partial_labeling"],
            "review": result["review"],
            "dataset": result["output"]["dataset"],
            "data_yaml": result["output"]["data_yaml"],
            "audit": result["output"]["audit"],
        }
    )
    return 0


def command_combine(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.combine import combine_sources

    report = combine_sources(
        out_dir=args.out,
        assisted=None if args.no_assisted else args.assisted,
        curated_needle_glob=args.curated_needle
        or [
            "/srv/fast/anatomy4-shenandoah/train-candidates/curated-needle/batch-*/annotations.jsonl"
        ],
        curated_globs=args.curated or [],
        negative_glob=args.negative_review,
        review_queue=args.review_queue,
        extra_sources=args.source,
        class_sources=args.class_source,
        candidate_manifest=args.candidate_manifest,
        images_root=args.images_root,
        blind_manifest=args.blind_manifest,
        blind_annotations=args.blind_annotations,
        seed=args.seed,
        train_ratio=args.train_ratio,
        include_negatives=not args.no_negatives,
        copy_mode=args.copy_mode,
        overlay_count=args.overlays,
        allow_missing_sources=args.allow_missing_sources,
        require_assisted=not args.no_assisted,
        require_blind_denylist=not args.allow_missing_blind_denylist,
    )
    _print(
        {
            "status": report["status"],
            "blocking": report["blocking"],
            "counts": report["counts"],
            "per_source": report["per_source"],
            "dedupe": report["dedupe"]["merged_boxes"],
            "conflicts": report["conflicts"]["quarantined"],
            "excluded": report["excluded"],
            "blind_check": report["blind_check"],
            "audit": report["audit"],
            "report": str(Path(args.out) / "merge-report.json"),
        }
    )
    return 0


def command_coarse_multi(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.coarse_multi import build_coarse_multi

    report = build_coarse_multi(
        proposals_path=args.proposals,
        manifest_path=args.manifest,
        curated_multi_globs=args.curated_multi,
        curated_needle_globs=args.curated_needle,
        blind_manifest_path=args.blind_manifest,
        blind_annotations_path=args.blind_annotations,
        out_path=args.out,
        report_path=args.report,
        leaf_threshold=args.leaf_threshold,
        flower_threshold=args.flower_threshold,
        cluster_threshold=args.cluster_threshold,
    )
    _print(report)
    return 0


def command_import_blind(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.blindtest import import_blind_test

    result = import_blind_test(
        annotations_path=args.annotations,
        manifest_path=args.manifest,
        out_dir=args.out,
        images_root=args.images_root,
        frozen_audit_path=args.frozen_audit,
        verify_frozen=not args.no_verify_frozen,
        verify_sha256=not args.no_verify_sha256,
        include_needs_review=args.include_needs_review,
        copy_mode=args.copy_mode,
        overlay_count=args.overlays,
    )
    _print(
        {
            "status": result["status"],
            "blocking": result["blocking"],
            "test_only": result["test_only"],
            "source": result["source_counts"],
            "included": result["included"],
            "excluded_needs_review": result["excluded_needs_review"],
            "overlays": [item["overlay"] for item in result["overlays"]],
            "audit": result["audit"],
            "marker": result["marker"],
        }
    )
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.dataset import (
        prepare_ultralytics,
        write_audit,
    )

    audit = _audit_or_exit(args.dataset, not args.no_require_test)
    if args.audit_out:
        write_audit(audit, args.audit_out)
    if audit["blocking"]:
        _print({"status": "invalid", "errors": audit["errors"]})
        return 2
    try:
        prepared = prepare_ultralytics(
            audit,
            args.out,
            link_mode=args.link_mode,
            allow_test_only=args.allow_test_only,
        )
    except Exception as exc:
        _print({"status": "refused", "error": str(exc)})
        return 2
    _print(prepared)
    return 0


def command_train(args: argparse.Namespace) -> int:
    _configure()
    from experiments.anatomy4_detector.dataset import (
        prepare_ultralytics,
        validate_dataset,
        write_audit,
    )
    from experiments.anatomy4_detector.train import train

    data_yaml = Path(args.data)
    prepared = None
    if data_yaml.is_dir():
        dataset_root = data_yaml
        audit = validate_dataset(dataset_root, require_test=False)
        audit_path = dataset_root / "dataset-audit.json"
        write_audit(audit, audit_path)
        if audit["blocking"]:
            _print({"status": "invalid", "errors": audit["errors"]})
            return 2
        if audit.get("test_only"):
            _print(
                {
                    "status": "refused",
                    "error": (
                        "frozen test-only dataset cannot be trained; use "
                        "`evaluate` with a trained checkpoint"
                    ),
                }
            )
            return 2
        prepared = prepare_ultralytics(
            audit,
            args.prepare_out or (dataset_root / ".ultralytics"),
            oversample_class=args.oversample_class,
            oversample_factor=args.oversample_factor,
        )
        data_yaml = Path(prepared["data_yaml"])

    report = train(
        data=data_yaml,
        project=args.project,
        name=args.name,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        amp=not args.no_amp,
        gpu_budget_gb=args.gpu_budget_gb,
        allow_low_vram=args.allow_low_vram,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
        cls=args.cls,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        dry_run=args.dry_run,
    )
    report["prepared"] = prepared
    _print(report)
    return 0 if report.get("ok") else 1


def command_evaluate(args: argparse.Namespace) -> int:
    _configure()
    from experiments.anatomy4_detector.dataset import (
        prepare_ultralytics,
        validate_dataset,
    )
    from experiments.anatomy4_detector.evaluate import evaluate_split

    audit = validate_dataset(
        args.dataset, require_test=args.split == "test"
    )
    prepared_root = args.prepared
    if prepared_root is None:
        prepared = prepare_ultralytics(
            audit, Path(args.out) / "prepared", allow_test_only=True
        )
        prepared_root = prepared["root"]
    if not args.weights and not args.eval_onnx:
        _print({"status": "invalid", "error": "pass --weights or --eval-onnx"})
        return 2
    result = evaluate_split(
        audit=audit,
        split=args.split,
        prepared_root=prepared_root,
        weights=args.weights,
        onnx_path=args.eval_onnx,
        out_dir=args.out,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        conf_threshold=args.conf,
        conf_thresholds=_parse_thresholds(args.conf_thresholds, args.conf),
        iou_threshold=args.iou,
        conf_floor=args.conf_floor,
        nms_iou=args.nms_iou,
        max_det=args.max_det,
        max_overlays=args.max_overlays,
        max_contact_images=args.max_contact_images,
        dry_run=args.dry_run,
    )
    _print(
        {
            "split": result["split"],
            "dry_run": result.get("dry_run", False),
            "images": result.get("images"),
            "gt_boxes": result.get("gt_boxes"),
            "macro": result.get("macro"),
            "micro": result.get("micro"),
            "per_class": result.get("per_class"),
            "counts": result.get("counts"),
            "metrics_json": result.get("metrics_json"),
            "predictions_jsonl": result.get("predictions_jsonl"),
            "overlays": len(result.get("overlays") or []),
            "contact_sheets": result.get("contact_sheets"),
        }
    )
    return 0


def command_export(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.export import export_onnx, verify_onnx

    export = export_onnx(args.weights, args.out)
    verification = None if args.no_verify else verify_onnx(export["onnx"])
    _print({"export": export, "verification": verification})
    if verification is not None and not verification["ok"]:
        failed = [item for item in verification["checks"] if not item["ok"]]
        print(f"ONNX verification failed: {failed}", file=sys.stderr)
        return 3
    return 0


def _parse_thresholds(spec: str | None, default: float) -> list[float]:
    values = [float(default)]
    for part in str(spec or "").split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return sorted({round(value, 4) for value in values if 0.0 < value <= 1.0})


def command_pipeline(args: argparse.Namespace) -> int:
    _configure()
    from experiments.anatomy4_detector.contracts import ContractError
    from experiments.anatomy4_detector.dataset import (
        assert_disjoint_datasets,
        prepare_ultralytics,
        render_audit_markdown,
        validate_dataset,
        write_audit,
    )
    from experiments.anatomy4_detector.evaluate import evaluate_split
    from experiments.anatomy4_detector.export import export_onnx, verify_onnx
    from experiments.anatomy4_detector.manifest import (
        build_run_manifest,
        write_manifest,
    )
    from experiments.anatomy4_detector.metrics import select_threshold_by_macro_f1
    from experiments.anatomy4_detector.train import train

    output = Path(args.out).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    test_dataset = Path(args.test_dataset).expanduser().resolve() if args.test_dataset else None
    test_audit = None
    isolation = None
    try:
        if test_dataset is not None:
            audit = validate_dataset(
                args.dataset, require_test=False, require_val=True
            )
            test_audit = validate_dataset(
                test_dataset, require_test=True, require_val=False
            )
        else:
            audit = validate_dataset(args.dataset, require_test=True, require_val=True)
    except ContractError as exc:
        _print({"status": "invalid", "error": str(exc)})
        return 2
    audit_path = write_audit(audit, output / "dataset-audit.json")
    (output / "dataset-audit.md").write_text(render_audit_markdown(audit))
    if test_audit is not None:
        write_audit(test_audit, output / "test-dataset-audit.json")
        (output / "test-dataset-audit.md").write_text(
            render_audit_markdown(test_audit)
        )
        if test_audit["blocking"]:
            _print({"status": "invalid", "test_dataset_errors": test_audit["errors"]})
            return 2
        non_test = (
            test_audit["splits"]["train"]["images"]
            + test_audit["splits"]["val"]["images"]
        )
        if non_test:
            _print(
                {
                    "status": "invalid",
                    "error": (
                        "--test-dataset must be test-only; found "
                        f"{non_test} train/val image(s)"
                    ),
                }
            )
            return 2
        try:
            isolation = assert_disjoint_datasets(args.dataset, test_dataset)
        except ContractError as exc:
            _print({"status": "isolation_violation", "error": str(exc)})
            return 2

    _print(
        {
            "dataset": {
                "status": audit["status"],
                "blocking": audit["blocking"],
                "split_isolation": audit["split_isolation"],
                "totals": audit["totals"],
                "support": audit["support"],
                "errors": audit["errors"],
                "warnings": [item["code"] for item in audit["warnings"]],
            },
            "test_dataset": (
                {
                    "status": test_audit["status"],
                    "test_only": test_audit.get("test_only"),
                    "totals": test_audit["totals"],
                    "support": test_audit["support"],
                    "isolation": isolation,
                }
                if test_audit is not None
                else None
            ),
        }
    )
    if audit["blocking"]:
        return 2
    if audit.get("test_only"):
        _print(
            {
                "status": "refused",
                "error": (
                    "frozen test-only dataset cannot enter the training pipeline; "
                    "evaluate a checkpoint instead"
                ),
            }
        )
        return 2

    prepared = prepare_ultralytics(
        audit,
        output / "prepared",
        link_mode=args.link_mode,
        oversample_class=args.oversample_class,
        oversample_factor=args.oversample_factor,
    )
    training = train(
        data=prepared["data_yaml"],
        project=output / "runs",
        name=args.name,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        seed=args.seed,
        patience=args.patience,
        amp=not args.no_amp,
        gpu_budget_gb=args.gpu_budget_gb,
        allow_low_vram=args.allow_low_vram,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        cos_lr=args.cos_lr,
        warmup_epochs=args.warmup_epochs,
        cls=args.cls,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        dry_run=args.dry_run,
    )
    evaluation = None
    evaluation_val = None
    evaluation_onnx = None
    tuned_threshold = float(args.conf)
    export = None
    verification = None
    if not args.dry_run and training.get("ok"):
        tune_thresholds = _parse_thresholds(
            args.tune_thresholds, args.conf
        ) if args.tune_conf_on_val else _parse_thresholds(
            args.conf_thresholds, args.conf
        )
        evaluation_val = evaluate_split(
            audit=audit,
            split="val",
            prepared_root=prepared["root"],
            weights=training["best_weights"],
            out_dir=output / "eval-val",
            imgsz=args.imgsz,
            device=args.device,
            batch=args.eval_batch,
            conf_threshold=args.conf,
            conf_thresholds=tune_thresholds,
            iou_threshold=args.iou,
            max_det=args.max_det,
            max_overlays=args.max_overlays,
            max_contact_images=args.max_contact_images,
        )
        if args.tune_conf_on_val:
            tuned_threshold = select_threshold_by_macro_f1(
                evaluation_val["threshold_sweep"], default=args.conf
            )
            evaluation_val["selected_threshold"] = tuned_threshold
            evaluation_val["tuning"] = {
                "method": "val_macro_f1",
                "candidates": tune_thresholds,
                "selected": tuned_threshold,
                "default": args.conf,
            }
        else:
            tuned_threshold = float(args.conf)
            evaluation_val["selected_threshold"] = tuned_threshold
            evaluation_val["tuning"] = {
                "method": "fixed_default",
                "candidates": tune_thresholds,
                "selected": tuned_threshold,
            }
        Path(evaluation_val["metrics_json"]).write_text(
            json.dumps(evaluation_val, indent=1) + "\n"
        )

        test_thresholds = (
            sorted({float(args.conf), tuned_threshold})
            if args.tune_conf_on_val
            else _parse_thresholds(args.conf_thresholds, args.conf)
        )
        if test_audit is not None:
            test_prepared = prepare_ultralytics(
                test_audit,
                output / "prepared-test",
                link_mode=args.link_mode,
                allow_test_only=True,
            )
            test_prepared_root = test_prepared["root"]
            test_audit_for_eval = test_audit
        else:
            test_prepared_root = prepared["root"]
            test_audit_for_eval = audit
        evaluation = evaluate_split(
            audit=test_audit_for_eval,
            split="test",
            prepared_root=test_prepared_root,
            weights=training["best_weights"],
            out_dir=output / "eval-test",
            imgsz=args.imgsz,
            device=args.device,
            batch=args.eval_batch,
            conf_threshold=args.conf,
            conf_thresholds=test_thresholds,
            iou_threshold=args.iou,
            max_det=args.max_det,
            max_overlays=args.max_overlays,
            max_contact_images=args.max_contact_images,
        )
        evaluation["selected_threshold"] = tuned_threshold
        evaluation["tuning"] = {
            "method": "val_macro_f1" if args.tune_conf_on_val else "fixed_default",
            "selected": tuned_threshold,
            "default": args.conf,
            "test_thresholds": test_thresholds,
            "val_candidates": (
                evaluation_val["tuning"]["candidates"]
                if evaluation_val.get("tuning")
                else []
            ),
        }
        Path(evaluation["metrics_json"]).write_text(
            json.dumps(evaluation, indent=1) + "\n"
        )
        export = export_onnx(
            training["best_weights"], output / "export" / "anatomy4.onnx"
        )
        verification = verify_onnx(export["onnx"])
        evaluation_onnx = evaluate_split(
            audit=test_audit_for_eval,
            split="test",
            prepared_root=test_prepared_root,
            weights=None,
            onnx_path=export["onnx"],
            out_dir=output / "eval-test-onnx",
            imgsz=args.imgsz,
            batch=args.eval_batch,
            conf_threshold=args.conf,
            conf_thresholds=test_thresholds,
            iou_threshold=args.iou,
            max_det=args.max_det,
            max_overlays=args.max_overlays,
            max_contact_images=args.max_contact_images,
        )
        evaluation_onnx["selected_threshold"] = tuned_threshold
        evaluation_onnx["tuning"] = evaluation["tuning"]
        Path(evaluation_onnx["metrics_json"]).write_text(
            json.dumps(evaluation_onnx, indent=1) + "\n"
        )

    manifest = build_run_manifest(
        code_root=Path(__file__).resolve().parents[2],
        dataset_audit=audit,
        dataset_audit_path=audit_path,
        training=training,
        evaluation=evaluation,
        evaluation_val=evaluation_val,
        evaluation_onnx=evaluation_onnx,
        export=export,
        onnx_verification=verification,
        prepared=prepared,
        synthetic=False,
        status="dry_run" if args.dry_run else None,
    )
    if test_audit is not None:
        manifest["test_dataset"] = {
            "root": test_audit["dataset_root"],
            "test_only": test_audit.get("test_only"),
            "totals": test_audit["totals"],
            "support": test_audit["support"],
            "isolation": isolation,
        }
    manifest_path = write_manifest(manifest, output / "run-manifest.json")
    _print(
        {
            "status": manifest["status"],
            "manifest": str(manifest_path),
            "training": {
                "ok": training.get("ok"),
                "best_weights": training.get("best_weights"),
                "elapsed_seconds": training.get("elapsed_seconds"),
            },
            "eval_val": {
                "macro": (evaluation_val or {}).get("macro"),
                "selected_threshold": (evaluation_val or {}).get("selected_threshold"),
                "leaf_tagged_bbox": (evaluation_val or {}).get("leaf_tagged_bbox"),
            },
            "eval_test_onnx": {
                "macro": (evaluation_onnx or {}).get("macro"),
                "micro": (evaluation_onnx or {}).get("micro"),
                "backend": (evaluation_onnx or {}).get("backend"),
                "leaf_tagged_bbox": (evaluation_onnx or {}).get("leaf_tagged_bbox"),
            },
            "eval_test": {
                "macro": (evaluation or {}).get("macro"),
                "micro": (evaluation or {}).get("micro"),
                "selected_threshold": (evaluation or {}).get("selected_threshold"),
                "thresholds": (evaluation or {}).get("threshold_sweep", {}).get(
                    "thresholds"
                ),
                "leaf_tagged_bbox": (evaluation or {}).get("leaf_tagged_bbox"),
            },
            "export": {
                "onnx": (export or {}).get("onnx"),
                "verified": (verification or {}).get("ok"),
            },
        }
    )
    if verification is not None and not verification["ok"]:
        return 3
    return 0


def command_publish(args: argparse.Namespace) -> int:
    from experiments.anatomy4_detector.publish import (
        PublishOptions,
        auth_from_env,
        publish,
        resolve_base,
    )

    options = PublishOptions(
        manifest_path=Path(args.manifest),
        dataset_root=Path(args.dataset) if args.dataset else None,
        test_dataset_root=Path(args.test_dataset) if args.test_dataset else None,
        materialization_path=Path(args.materialization) if args.materialization else None,
        dataset_id=args.dataset_id,
        model_id=args.model_id,
        name=args.name,
        batch_size=args.batch_size,
        skip_items=args.skip_items,
        include_pending=not args.no_pending,
        max_previews=args.max_previews,
        allow_no_previews=args.no_previews,
        allow_synthetic=args.allow_synthetic,
        allow_zero_support=args.allow_zero_support,
        test_image=Path(args.test_image) if args.test_image else None,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        dry_run=args.dry_run,
        report_out=Path(args.report_out) if args.report_out else None,
    )
    auth = auth_from_env()
    base = args.base or __import__("os").environ.get("ARQ_BASE") or "http://localhost:8000"
    result = publish(
        options,
        base=base,
        auth=auth,
        timeout=args.timeout,
        retries=args.retries,
    )
    _print(result)
    return 0


def command_smoke(args: argparse.Namespace) -> int:
    _configure()
    from experiments.anatomy4_detector.smoke import run_smoke

    manifest = run_smoke(
        args.out,
        dataset_dir=args.dataset,
        device=args.device,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        gpu_budget_gb=args.gpu_budget_gb,
        dry_run=args.dry_run,
    )
    _print(
        {
            "status": manifest.get("status"),
            "synthetic": manifest.get("synthetic"),
            "manifest": str(Path(args.out) / "run-manifest.json"),
            "dataset": (manifest.get("dataset") or {}).get("totals"),
            "eval": ((manifest.get("evaluation") or {}).get("metrics") or {}).get(
                "macro"
            ),
            "export": (manifest.get("export") or {}).get("onnx"),
            "export_verified": (
                (manifest.get("export") or {}).get("verification") or {}
            ).get("ok"),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anatomy4-detector",
        description="anatomy4-v1 four-class detector pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="audit a materialized dataset")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--json-out", default=None)
    validate.add_argument("--markdown-out", default=None)
    validate.add_argument("--no-require-test", action="store_true")
    validate.set_defaults(func=command_validate)

    combine = sub.add_parser(
        "combine", help="merge mixed reviewed sources -> train/val dataset"
    )
    combine.add_argument("--out", required=True)
    combine.add_argument(
        "--assisted",
        default="/srv/fast/anatomy4-shenandoah/train-candidates/verified/assisted-labels.jsonl",
    )
    combine.add_argument(
        "--curated-needle",
        action="append",
        default=None,
        help="curated needle JSONL/glob (repeatable or comma-separated)",
    )
    combine.add_argument(
        "--curated",
        action="append",
        default=[],
        help="additional curated JSONL/glob, any class (repeatable)",
    )
    combine.add_argument(
        "--negative-review",
        default="/srv/fast/anatomy4-shenandoah/train-candidates/negative-review/batch-*.jsonl",
    )
    combine.add_argument("--review-queue", action="append", default=[])
    combine.add_argument("--source", action="append", default=[], help="name=path")
    combine.add_argument(
        "--class-source", action="append", default=[], help="CLASS=source[,source]"
    )
    combine.add_argument(
        "--candidate-manifest",
        default="/srv/fast/anatomy4-shenandoah/train-candidates/manifest.jsonl",
    )
    combine.add_argument(
        "--images-root",
        default="/srv/fast/anatomy4-shenandoah/train-candidates",
    )
    combine.add_argument("--blind-manifest", default=None)
    combine.add_argument("--blind-annotations", default=None)
    combine.add_argument("--seed", default="anatomy4-train-v1")
    combine.add_argument("--train-ratio", type=float, default=0.85)
    combine.add_argument("--no-negatives", action="store_true")
    combine.add_argument(
        "--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy"
    )
    combine.add_argument("--overlays", type=int, default=6)
    combine.add_argument("--no-assisted", action="store_true")
    combine.add_argument("--allow-missing-sources", action="store_true")
    combine.add_argument("--allow-missing-blind-denylist", action="store_true")
    combine.set_defaults(func=command_combine)

    coarse = sub.add_parser(
        "coarse-multi", help="cheap cross-organ coarse tags for remaining Plantae"
    )
    coarse.add_argument(
        "--proposals",
        default="/srv/fast/anatomy4-shenandoah/train-candidates/hybrid-proposals.json",
    )
    coarse.add_argument(
        "--manifest",
        default="/srv/fast/anatomy4-shenandoah/train-candidates/manifest.jsonl",
    )
    coarse.add_argument(
        "--curated-multi",
        action="append",
        default=[
            "/srv/fast/anatomy4-shenandoah/train-candidates/curated-multi/*/annotations.jsonl"
        ],
    )
    coarse.add_argument(
        "--curated-needle",
        action="append",
        default=[
            "/srv/fast/anatomy4-shenandoah/train-candidates/curated-needle/batch-*/annotations.jsonl"
        ],
    )
    coarse.add_argument(
        "--blind-manifest",
        default="/srv/fast/anatomy4-shenandoah/blind-test/merged/manifest.jsonl",
    )
    coarse.add_argument(
        "--blind-annotations",
        default="/srv/fast/anatomy4-shenandoah/blind-test/merged/annotations.jsonl",
    )
    coarse.add_argument(
        "--out",
        default="/srv/fast/anatomy4-shenandoah/train-candidates/coarse-multi/auto-plantae.jsonl",
    )
    coarse.add_argument(
        "--report",
        default="/srv/fast/anatomy4-shenandoah/train-candidates/coarse-multi/auto-plantae-report.json",
    )
    coarse.add_argument("--leaf-threshold", type=float, default=0.24)
    coarse.add_argument("--flower-threshold", type=float, default=0.24)
    coarse.add_argument("--cluster-threshold", type=float, default=0.28)
    coarse.set_defaults(func=command_coarse_multi)

    import_blind = sub.add_parser(
        "import-blind", help="frozen blind benchmark -> test-only eval dataset"
    )
    import_blind.add_argument("--annotations", required=True)
    import_blind.add_argument("--manifest", required=True)
    import_blind.add_argument("--out", required=True)
    import_blind.add_argument("--images-root", default=None)
    import_blind.add_argument("--frozen-audit", default=None)
    import_blind.add_argument("--no-verify-frozen", action="store_true")
    import_blind.add_argument("--no-verify-sha256", action="store_true")
    import_blind.add_argument("--include-needs-review", action="store_true")
    import_blind.add_argument(
        "--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy"
    )
    import_blind.add_argument("--overlays", type=int, default=6)
    import_blind.set_defaults(func=command_import_blind)

    materialize = sub.add_parser(
        "materialize", help="audit proposals + review -> contract dataset"
    )
    materialize.add_argument("--proposals", required=True)
    materialize.add_argument("--review", required=True)
    materialize.add_argument("--out", required=True)
    materialize.add_argument("--manifest", default=None)
    materialize.add_argument("--images-dir", default=None)
    materialize.add_argument(
        "--classes", default=None, help="comma-separated subset in contract order"
    )
    materialize.add_argument("--seed", default="anatomy4-v1")
    materialize.add_argument(
        "--negative-policy",
        choices=("reviewed-rejected", "explicit", "none"),
        default="reviewed-rejected",
    )
    materialize.add_argument("--negative-image", action="append", default=[])
    materialize.add_argument("--negative-file", default=None)
    materialize.add_argument("--background-review", default=None)
    materialize.add_argument("--require-fully-reviewed", action="store_true")
    materialize.add_argument("--allow-missing-images", action="store_true")
    materialize.add_argument(
        "--copy-mode", choices=("copy", "hardlink", "symlink"), default="copy"
    )
    materialize.add_argument("--keep-duplicate-boxes", action="store_true")
    materialize.add_argument("--no-verify-sha256", action="store_true")
    materialize.add_argument("--no-require-test", action="store_true")
    materialize.add_argument("--no-require-val", action="store_true")
    materialize.set_defaults(func=command_materialize)

    prepare = sub.add_parser("prepare", help="write the Ultralytics view")
    prepare.add_argument("--dataset", required=True)
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--audit-out", default=None)
    prepare.add_argument(
        "--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    prepare.add_argument("--no-require-test", action="store_true")
    prepare.add_argument("--allow-test-only", action="store_true")
    prepare.set_defaults(func=command_prepare)

    train = sub.add_parser("train", help="train yolo26n detect")
    train.add_argument("--data", required=True, help="prepared data.yaml or dataset dir")
    train.add_argument("--prepare-out", default=None)
    train.add_argument("--model", default="yolo26n.pt")
    train.add_argument("--epochs", type=int, default=80)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--workers", type=int, default=4)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=20261112)
    train.add_argument("--patience", type=int, default=20)
    train.add_argument("--no-amp", action="store_true")
    train.add_argument("--optimizer", default="auto")
    train.add_argument("--lr0", type=float, default=None)
    train.add_argument("--lrf", type=float, default=None)
    train.add_argument("--cos-lr", action="store_true")
    train.add_argument("--warmup-epochs", type=float, default=3.0)
    train.add_argument("--cls", type=float, default=0.5)
    train.add_argument("--mosaic", type=float, default=1.0)
    train.add_argument("--close-mosaic", type=int, default=10)
    train.add_argument("--oversample-class", default=None)
    train.add_argument("--oversample-factor", type=int, default=1)
    train.add_argument("--project", default="runs/anatomy4")
    train.add_argument("--name", default="train")
    train.add_argument("--gpu-budget-gb", type=float, default=3.2)
    train.add_argument("--allow-low-vram", action="store_true")
    train.add_argument("--dry-run", action="store_true")
    train.set_defaults(func=command_train)

    evaluate = sub.add_parser("evaluate", help="held-out evaluation")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--prepared", default=None)
    evaluate.add_argument("--weights", default=None)
    evaluate.add_argument(
        "--eval-onnx",
        "--onnx",
        dest="eval_onnx",
        default=None,
        help="evaluate the exported ONNX graph (arq-api one2many decode)",
    )
    evaluate.add_argument("--split", default="test", choices=("train", "val", "test"))
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--imgsz", type=int, default=640)
    evaluate.add_argument("--batch", type=int, default=8)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--conf", type=float, default=0.25)
    evaluate.add_argument("--conf-thresholds", default="0.25")
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.add_argument("--conf-floor", type=float, default=0.001)
    evaluate.add_argument("--nms-iou", type=float, default=0.45)
    evaluate.add_argument("--max-det", type=int, default=300)
    evaluate.add_argument("--max-overlays", type=int, default=48)
    evaluate.add_argument("--max-contact-images", type=int, default=240)
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(func=command_evaluate)

    export = sub.add_parser("export", help="ONNX export + verification")
    export.add_argument("--weights", required=True)
    export.add_argument("--out", default=None)
    export.add_argument("--no-verify", action="store_true")
    export.set_defaults(func=command_export)

    pipeline = sub.add_parser("pipeline", help="validate -> train -> eval -> export -> manifest")
    pipeline.add_argument("--dataset", required=True)
    pipeline.add_argument("--test-dataset", default=None)
    pipeline.add_argument("--out", required=True)
    pipeline.add_argument("--name", default="anatomy4")
    pipeline.add_argument("--model", default="yolo26n.pt")
    pipeline.add_argument("--epochs", type=int, default=80)
    pipeline.add_argument("--imgsz", type=int, default=640)
    pipeline.add_argument("--batch", type=int, default=8)
    pipeline.add_argument("--workers", type=int, default=4)
    pipeline.add_argument("--device", default="auto")
    pipeline.add_argument("--seed", type=int, default=20261112)
    pipeline.add_argument("--patience", type=int, default=20)
    pipeline.add_argument("--no-amp", action="store_true")
    pipeline.add_argument("--optimizer", default="auto")
    pipeline.add_argument("--lr0", type=float, default=None)
    pipeline.add_argument("--lrf", type=float, default=None)
    pipeline.add_argument("--cos-lr", action="store_true")
    pipeline.add_argument("--warmup-epochs", type=float, default=3.0)
    pipeline.add_argument("--cls", type=float, default=0.5)
    pipeline.add_argument("--mosaic", type=float, default=1.0)
    pipeline.add_argument("--close-mosaic", type=int, default=10)
    pipeline.add_argument("--oversample-class", default=None)
    pipeline.add_argument("--oversample-factor", type=int, default=1)
    pipeline.add_argument("--gpu-budget-gb", type=float, default=3.2)
    pipeline.add_argument("--allow-low-vram", action="store_true")
    pipeline.add_argument(
        "--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    pipeline.add_argument("--eval-batch", type=int, default=8)
    pipeline.add_argument("--conf", type=float, default=0.25)
    pipeline.add_argument("--iou", type=float, default=0.5)
    pipeline.add_argument("--max-det", type=int, default=300)
    pipeline.add_argument("--max-overlays", type=int, default=48)
    pipeline.add_argument("--max-contact-images", type=int, default=240)
    pipeline.add_argument(
        "--conf-thresholds", default="0.10,0.25,0.50",
        help="comma-separated test confidence thresholds",
    )
    pipeline.add_argument("--tune-conf-on-val", action="store_true")
    pipeline.add_argument(
        "--tune-thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.75",
    )
    pipeline.add_argument("--evaluate-val", action="store_true")
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.set_defaults(func=command_pipeline)

    publish = sub.add_parser(
        "publish", help="register the run with arq-api (dry-run supported)"
    )
    publish.add_argument("--manifest", required=True)
    publish.add_argument("--dataset", default=None, help="default: manifest dataset root")
    publish.add_argument("--test-dataset", default=None)
    publish.add_argument("--materialization", default=None)
    publish.add_argument("--base", default=None, help="default: $ARQ_BASE")
    publish.add_argument("--dataset-id", default=None)
    publish.add_argument("--model-id", default=None)
    publish.add_argument("--name", default=None)
    publish.add_argument("--batch-size", type=int, default=200)
    publish.add_argument("--skip-items", action="store_true")
    publish.add_argument("--no-pending", action="store_true")
    publish.add_argument("--max-previews", type=int, default=8)
    publish.add_argument("--no-previews", action="store_true")
    publish.add_argument("--allow-synthetic", action="store_true")
    publish.add_argument("--allow-zero-support", action="store_true")
    publish.add_argument("--test-image", default=None)
    publish.add_argument("--threshold", type=float, default=0.25)
    publish.add_argument("--iou-threshold", type=float, default=0.45)
    publish.add_argument("--timeout", type=float, default=60.0)
    publish.add_argument("--retries", type=int, default=3)
    publish.add_argument("--dry-run", action="store_true")
    publish.add_argument("--report-out", default=None)
    publish.set_defaults(func=command_publish)

    smoke = sub.add_parser("smoke", help="synthetic end-to-end")
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--dataset", default=None)
    smoke.add_argument("--epochs", type=int, default=2)
    smoke.add_argument("--imgsz", type=int, default=160)
    smoke.add_argument("--batch", type=int, default=4)
    smoke.add_argument("--workers", type=int, default=0)
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--gpu-budget-gb", type=float, default=3.2)
    smoke.add_argument("--dry-run", action="store_true")
    smoke.set_defaults(func=command_smoke)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
