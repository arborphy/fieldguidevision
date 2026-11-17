# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-YOLO-TRAIN-EVAL-EXPORT role=canonical
"""CLI entrypoint: python experiments/anatomy4/cli.py <command>."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.anatomy4.altsheets import (
    empty_image_stats,
    render_class_sheets,
    render_needle_detail,
)
from experiments.anatomy4.audit import compute_audit, write_audit
from experiments.anatomy4.export import export_proposals
from experiments.anatomy4.compare import (
    load_review_csv,
    load_review_dir,
    markdown_table,
    summarize,
)
from experiments.anatomy4.owlv2 import run_owlv2
from experiments.anatomy4.worldv2 import run_worldv2
from experiments.anatomy4.contracts import load_config
from experiments.anatomy4.download import download_manifest
from experiments.anatomy4.labels import load_review, load_verification, write_yolo_dataset
from experiments.anatomy4.needle_clip_filter import run_needle_clip_filter
from experiments.anatomy4.needle_crop import run_needle_second_pass
from experiments.anatomy4.manifest import (
    AREA_INSTANCE_ID,
    build_manifest,
    build_targeted_manifest,
    load_manifest,
    write_manifest,
)
from experiments.anatomy4.prelabel import configure_logging, run_prelabel
from experiments.anatomy4.yoloe import run_yoloe, run_yoloe_variants
from experiments.anatomy4.policy import fit_policies, render_policy_markdown
from experiments.anatomy4.train_queue import (
    load_manifest_metadata,
    run_train_verification,
    select_photo_url,
    select_production_candidates,
)
from experiments.anatomy4.verify import (
    build_report,
    load_openrouter_settings,
    load_review_rows,
    render_disagreement_cards,
    render_report_markdown,
    run_verification,
)
from experiments.anatomy4.render import (
    render_background_sample,
    render_contact_sheets,
    render_image_overlay,
    render_review_sample,
    write_review_sheet,
)


def _db_command(args: argparse.Namespace) -> list[str]:
    return [
        "docker",
        "exec",
        args.db_container,
        "psql",
        "-U",
        args.db_user,
        "-d",
        args.db_name,
    ]


def command_build_manifest(args: argparse.Namespace) -> int:
    if getattr(args, "mode", "broad") == "targeted":
        manifest = build_targeted_manifest(
            area_instance_id=args.area_instance_id,
            seed=args.seed,
            needle_observations=args.needle_observations,
            photos_per_needle_observation=args.photos_per_needle_observation,
            leaf_observations=args.leaf_observations,
            flower_observations=args.flower_observations,
            background_observations=args.background_observations,
            db_command=_db_command(args),
        )
    else:
        manifest = build_manifest(
            area_instance_id=args.area_instance_id,
            sample_size=args.sample_size,
            seed=args.seed,
            plant_fraction=args.plant_fraction,
            needle_minimum=args.needle_minimum,
            db_command=_db_command(args),
        )
    write_manifest(manifest, args.out)
    print(json.dumps(manifest["source_counts"], indent=1))
    print(json.dumps(manifest["strata"], indent=1))
    print(f"manifest: {args.out}")
    return 0


def command_download(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    download_manifest(
        manifest,
        args.images_dir,
        vault_root=args.vault_root,
        vault_base_url=args.vault_base_url,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    out_manifest = args.out_manifest or str(Path(args.images_dir).parent / "manifest.downloaded.json")
    write_manifest(manifest, out_manifest)
    print(json.dumps(manifest["download_summary"], indent=1))
    print(f"manifest: {out_manifest}")
    return 0


def command_prelabel(args: argparse.Namespace) -> int:
    configure_logging()
    manifest = load_manifest(args.manifest)
    config = load_config(args.config)
    result = run_prelabel(
        manifest,
        config,
        args.out,
        device=args.device,
        resume=args.resume,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=1))
    print(f"proposals: {Path(args.out) / 'proposals.json'}")
    return 0


def command_prelabel_yoloe(args: argparse.Namespace) -> int:
    configure_logging()
    manifest = load_manifest(args.manifest)
    config = load_config(args.config)
    result = run_yoloe(
        manifest,
        config,
        args.out,
        model_name=args.model,
        imgsz=args.imgsz,
        device=args.device,
        conf_floor=args.conf_floor,
        resume=args.resume,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=1))
    print(f"proposals: {Path(args.out) / 'proposals.json'}")
    return 0


def command_prelabel_yoloe_variants(args: argparse.Namespace) -> int:
    configure_logging()
    manifest = load_manifest(args.manifest)
    config = load_config(args.config)
    variants_doc = json.loads(Path(args.variants).read_text())
    variants = variants_doc["variants"] if isinstance(variants_doc, dict) else variants_doc
    result = run_yoloe_variants(
        manifest,
        config,
        variants,
        args.out,
        model_name=args.model,
        imgsz=args.imgsz,
        device=args.device,
        conf_floor=args.conf_floor,
        bucket=args.bucket,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=1))
    print(f"proposals: {Path(args.out) / 'proposals.json'}")
    return 0


def command_prelabel_world(args: argparse.Namespace) -> int:
    configure_logging()
    manifest = load_manifest(args.manifest)
    config = load_config(args.config)
    result = run_worldv2(
        manifest,
        config,
        args.out,
        model_name=args.model,
        imgsz=args.imgsz,
        device=args.device,
        conf_floor=args.conf_floor,
        max_per_class=args.max_per_class,
        bucket=args.bucket,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=1))
    print(f"proposals: {Path(args.out) / 'proposals.json'}")
    return 0


def command_prelabel_owlv2(args: argparse.Namespace) -> int:
    configure_logging()
    manifest = load_manifest(args.manifest)
    config = load_config(args.config)
    result = run_owlv2(
        manifest,
        config,
        args.out,
        model_name=args.model,
        device=args.device,
        score_threshold=args.score_threshold,
        max_per_class=args.max_per_class,
        bucket=args.bucket,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=1))
    print(f"proposals: {Path(args.out) / 'proposals.json'}")
    return 0


def command_render_alt_sheets(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    from collections import Counter

    buckets = Counter(
        str(image.get("audit_bucket") or "unknown")
        for image in manifest.get("images", [])
        if image.get("selected_for_audit")
    )
    for class_name in args.classes:
        sheets, review_csv = render_class_sheets(
            manifest,
            args.out,
            class_name=class_name,
            per_class=args.per_class,
            columns=args.columns,
        )
        if class_name == "needle" and args.needle_detail:
            render_needle_detail(manifest, args.out, per_class=args.per_class)
    export = {
        "summary": manifest.get("summary", {}),
        "buckets": dict(buckets),
        "empty_images": empty_image_stats(manifest),
    }
    (Path(args.out) / "empty-stats.json").write_text(json.dumps(export, indent=1))
    print(json.dumps(export, indent=1))
    return 0


def command_export(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    thresholds = None
    if args.thresholds:
        thresholds = json.loads(Path(args.thresholds).read_text())
    summary = export_proposals(
        manifest,
        args.out,
        thresholds=thresholds,
        max_box_area=args.max_box_area,
        per_class_limit=args.per_class_limit,
    )
    print(json.dumps(summary, indent=1))
    return 0


def command_compare(args: argparse.Namespace) -> int:
    results = []
    for name, manifest_path, review_path in args.run:
        manifest = load_manifest(manifest_path)
        reviews = (
            load_review_dir(review_path)
            if any(Path(part).is_dir() for part in str(review_path).split(","))
            else load_review_csv(review_path)
        )
        results.append(
            summarize(
                manifest,
                reviews,
                dataset=f"{args.dataset}:{name}" if args.dataset else name,
            )
        )
    print(markdown_table(results))
    output = json.dumps(results, indent=1, default=str)
    Path(args.out).write_text(output)
    print(f"wrote: {args.out}")
    return 0


def command_prelabel_hybrid(args: argparse.Namespace) -> int:
    from .hybrid import run_hybrid

    blind_paths = list(args.blind_manifest or [])
    stats = run_hybrid(
        args.manifest,
        args.out,
        device=args.device,
        classes=args.classes,
        limit_per_class=args.limit,
        blind_manifest_paths=blind_paths,
        render_sheets=not args.no_sheets,
    )
    print(json.dumps(stats["proposals_by_class"], indent=1))
    print(f"stats: {Path(args.out) / 'stats.json'}")
    return 0


def command_reindex_hybrid(args: argparse.Namespace) -> int:
    from .hybrid import reindex_hybrid

    stats = reindex_hybrid(
        args.out,
        manifest_path=args.manifest,
        iou_threshold=args.nms_iou,
        max_per_class=args.max_per_class,
        render_sheets=not args.no_sheets,
    )
    print(json.dumps(stats["proposals_by_class"], indent=1))
    print(f"stats: {Path(args.out) / 'stats.json'}")
    return 0


def command_render(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    sheets = render_contact_sheets(manifest, args.out)
    review_path = write_review_sheet(manifest, Path(args.out) / "review.csv")
    sample_sheets, sample_csv = render_review_sample(manifest, args.out, per_class=args.sample_per_class)
    image_paths = {
        str(image.get("photo_id") or image.get("image_id")): image.get("download", {}).get("path")
        for image in manifest.get("images", [])
    }
    by_image: dict[str, list[dict]] = {}
    for proposal in manifest.get("proposals", []):
        if proposal.get("status") == "proposed":
            by_image.setdefault(proposal["photo_id"], []).append(proposal)
    overlays = 0
    for photo_id, proposals in by_image.items():
        path = image_paths.get(photo_id)
        if not path:
            continue
        render_image_overlay(path, proposals, Path(args.out) / "images" / f"{photo_id}.jpg")
        overlays += 1
    background_sheets, background_csv = render_background_sample(
        manifest, args.out, per_bucket=args.background_per_bucket
    )
    print(
        json.dumps(
            {
                "contact_sheets": len(sheets),
                "review_sample_sheets": len(sample_sheets),
                "background_sheets": len(background_sheets),
                "image_overlays": overlays,
                "review": review_path,
                "review_sample": sample_csv,
                "review_background": background_csv,
            },
            indent=1,
        )
    )
    return 0


def command_needle_second_pass(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    summary = run_needle_second_pass(
        manifest,
        args.out,
        model_name=args.model,
        imgsz=args.imgsz,
        device=args.device,
        crop_margin=args.crop_margin,
        min_candidate_score=args.min_candidate_score,
        second_prompts=args.prompts,
        conf_floor=args.conf_floor,
        limit=args.limit,
        first_prompt_only=args.keep_first,
    )
    print(json.dumps(summary, indent=1))
    return 0


def command_needle_clip_filter(args: argparse.Namespace) -> int:
    summary = run_needle_clip_filter(
        args.second_pass,
        args.out,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        download_root=args.download_root,
    )
    print(json.dumps(summary, indent=1))
    return 0


def command_merge_runs(args: argparse.Namespace) -> int:
    inputs = [load_manifest(path) for path in args.inputs]
    images: dict[str, dict] = {}
    proposals: list[dict] = []
    runs: list[dict] = []
    for manifest in inputs:
        for image in manifest.get("images", []):
            key = str(image.get("photo_id") or image.get("image_id"))
            images.setdefault(key, image)
        proposals.extend(manifest.get("proposals", []))
        runs.append(manifest.get("provenance"))
    merged = {
        "schema": "arq.anatomy4.prelabel/v1",
        "contract": inputs[0].get("contract") if inputs else None,
        "created_at": inputs[0].get("created_at") if inputs else None,
        "backend": "yoloe-variants-merged",
        "provenance": {"runs": runs},
        "images": list(images.values()),
        "proposals": proposals,
        "summary": {
            "images_requested": len(images),
            "images_processed": len(images),
            "proposals": len(proposals),
            "proposals_by_class": {
                name: len([p for p in proposals if p.get("class_name") == name])
                for name in ("leaf", "needle", "flower", "inflorescence")
            },
            "proposals_by_prompt": {},
            "total_runtime_seconds": round(
                sum(
                    float(record.get("elapsed_seconds") or 0)
                    for manifest in inputs
                    for record in manifest.get("images", [])
                ),
                2,
            ),
        },
    }
    from collections import Counter

    merged["summary"]["proposals_by_prompt"] = dict(
        Counter(str(p.get("prompt_variant") or p.get("prompt") or "") for p in proposals)
    )
    write_manifest(merged, args.out)
    print(json.dumps(merged["summary"], indent=1))
    print(f"merged: {args.out}")
    return 0


def command_labels(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    reviews = load_review(args.review) if args.review else {}
    verification = load_verification(args.verdicts) if args.verdicts else None
    summary = write_yolo_dataset(
        manifest,
        args.out,
        reviews=reviews,
        verification=verification,
        verification_profile=args.verify_profile,
        require_both=args.require_both,
    )
    print(json.dumps(summary, indent=1))
    return 0


def command_audit(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    reviews = load_review(args.review) if args.review else None
    audit = compute_audit(manifest, reviews)
    if getattr(args, "background_review", None):
        import csv as _csv

        counts = {"empty_correct": 0, "missed_organ": 0, "inspected": 0}
        with Path(args.background_review).open(newline="") as handle:
            for row in _csv.DictReader(handle):
                verdict = (row.get("verdict") or "").strip().lower()
                if not verdict:
                    continue
                counts["inspected"] += 1
                if verdict in {"empty_correct", "empty", "correct"}:
                    counts["empty_correct"] += 1
                elif verdict in {"missed_organ", "missed", "fn"}:
                    counts["missed_organ"] += 1
        counts["accuracy"] = (
            counts["empty_correct"] / counts["inspected"]
            if counts["inspected"]
            else None
        )
        audit["empty_review"] = counts
    write_audit(audit, args.out)
    print((Path(args.out) / "audit.md").read_text())
    return 0


def command_verify_crops(args: argparse.Namespace) -> int:
    settings = load_openrouter_settings(env_file=args.env_file, model=args.model)
    report = run_verification(
        proposals_path=args.proposals,
        images_dir=args.images_dir,
        review_path=args.review,
        manifest_path=args.manifest,
        out_dir=args.out,
        settings=settings,
        batch_size=args.batch_size,
        margin=args.margin,
        max_calls=args.max_calls,
        min_confidence=args.min_confidence,
        attempts=args.attempts,
        second_pass=not args.no_second_pass,
        second_pass_margin=args.second_pass_margin,
        second_pass_confidence=args.second_pass_confidence,
        limit=args.limit,
        sample_per_class=args.sample_per_class,
        proposal_ids=(
            [item.strip() for item in args.proposal_ids.split(",") if item.strip()]
            if args.proposal_ids
            else None
        ),
        prompt_variant=args.prompt_variant,
        crop_max_side=args.crop_max_side,
    )
    print(json.dumps(report["overall"], indent=1))
    print(json.dumps(report["recommended_policy"], indent=1))
    print(f"report: {Path(args.out) / 'report.md'}")
    return 0


def command_train_verify(args: argparse.Namespace) -> int:
    input_paths = (args.yoloe_proposals, args.owlv2_proposals, args.manifest)
    if any("blind-test" in str(path).lower() for path in input_paths + (args.images_dir,)):
        raise ValueError("blind-test paths are forbidden for train verification")
    if not all("train-candidates" in str(path) for path in input_paths):
        raise ValueError("all candidate inputs must come from train-candidates")
    if Path(args.out).exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing output directory: {args.out}")
    if args.workers < 1 or args.workers > 4:
        raise ValueError("bounded production concurrency must be between 1 and 4 workers")
    if args.max_crops < 1 or args.max_crops > 1200:
        raise ValueError("production crop cap must be between 1 and 1200")

    yoloe_data = json.loads(Path(args.yoloe_proposals).read_text())
    owl_data = json.loads(Path(args.owlv2_proposals).read_text())
    selected, selection_meta = select_production_candidates(
        yoloe_proposals=yoloe_data.get("proposals") or [],
        owlv2_proposals=owl_data.get("proposals") or [],
        leaf_count=args.leaf_count,
        inflorescence_count=args.inflorescence_count,
        flower_count=args.flower_count,
        seed=args.selection_seed,
        max_per_photo=args.max_per_photo,
    )
    if len(selected) > args.max_crops:
        raise ValueError(f"selection of {len(selected)} exceeds crop cap {args.max_crops}")
    selection_meta["sources"] = {
        "yoloe": str(Path(args.yoloe_proposals).resolve()),
        "owlv2": str(Path(args.owlv2_proposals).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "images_dir": str(Path(args.images_dir).resolve()),
    }
    manifest = load_manifest_metadata(args.manifest)
    missing_urls = [
        row["proposal_id"] for row in selected if not select_photo_url(row, manifest)
    ]
    if missing_urls:
        raise ValueError(f"photo_url missing for {len(missing_urls)} selected proposals; first={missing_urls[:5]}")

    output = Path(args.out)
    if output.exists():
        selected_file = output / "selected-proposals.json"
        if not selected_file.exists():
            raise FileExistsError("resume requested but selected-proposals.json is missing")
        existing = json.loads(selected_file.read_text()).get("proposals") or []
        if [row.get("proposal_id") for row in existing] != [row.get("proposal_id") for row in selected]:
            raise ValueError("resume selection does not match saved selected-proposals.json")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "selection.json").write_text(json.dumps(selection_meta, indent=2, sort_keys=True))
        (output / "selected-proposals.json").write_text(
            json.dumps({"schema": "arq.anatomy4.train-candidate-selection/v1", "proposals": selected}, sort_keys=True)
        )
    policy_paths = {
        "yoloe-26s": args.yoloe_policy,
        "owlv2-base": args.owlv2_policy,
    }
    policies = {key: json.loads(Path(path).read_text()) for key, path in policy_paths.items()}
    settings = load_openrouter_settings(env_file=args.env_file, model=args.model)
    stats = run_train_verification(
        proposals=selected,
        manifest=manifest,
        images_dir=args.images_dir,
        policies=policies,
        settings=settings,
        out_dir=output,
        requested_policies={
            "leaf": "yoloe-26s",
            "inflorescence": "yoloe-26s",
            "flower": "owlv2-base",
        },
        selection_metadata=selection_meta,
        batch_size=args.batch_size,
        max_workers=args.workers,
        max_crops=args.max_crops,
        audit_fraction=args.audit_fraction,
        audit_seed=args.audit_seed,
        max_retries=args.attempts,
    )
    print(json.dumps({
        "verified": stats["verified"],
        "assisted_label_candidates": stats["assisted_label_candidates"],
        "review_queue_items": stats["review_queue_items"],
        "calls": stats["calls"],
        "cost": stats["usage"]["cost"],
        "stats": str(output / "stats.json"),
    }, indent=2))
    return 0


def command_policy_fit(args: argparse.Namespace) -> int:
    alternatives = {}
    if args.needle_mainsubject_verdicts:
        alternatives["needle"] = {
            "variant": "main_subject",
            "source": args.needle_mainsubject_verdicts,
            "measured": _summarize_verdict_file(args.needle_mainsubject_verdicts),
        }
    policy = fit_policies(
        proposals_path=args.proposals,
        verdicts_path=args.verdicts,
        review_path=args.review,
        model=args.model,
        detector=args.detector,
        target_precision=args.target_precision,
        min_accepts=args.min_accepts,
        min_assisted_support=args.min_assisted_support,
        min_assisted_precision=args.min_assisted_precision,
        loo_min_precision=args.loo_min_precision,
        validation=(
            {"verdicts": args.validation_verdicts, "review": args.validation_review}
            if args.validation_verdicts and args.validation_review
            else None
        ),
        background=(
            {"verdicts": args.background_verdicts, "review": args.background_review}
            if args.background_verdicts and args.background_review
            else None
        ),
        validation_min_precision=args.validation_min_precision,
        validation_min_accepts=args.validation_min_accepts,
        verifier_alternatives=alternatives or None,
        decision_out=Path(args.out) / "decisions.jsonl",
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "policy.json").write_text(json.dumps(policy, indent=2, sort_keys=True))
    (Path(args.out) / "policy.md").write_text(render_policy_markdown(policy))
    print(render_policy_markdown(policy))
    print(f"policy: {Path(args.out) / 'policy.json'} (id {policy['policy_id']})")
    return 0


def _summarize_verdict_file(path: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    accepts = [row for row in rows if (row.get("decision") or {}).get("accept")]
    tp = sum(1 for row in accepts if (row.get("review_verdict") or "") == "tp")
    return {
        "accepts": len(accepts),
        "tp": tp,
        "fp": len(accepts) - tp,
        "precision": round(tp / len(accepts), 4) if accepts else None,
        "recall": round(tp / sum(1 for row in rows if (row.get("review_verdict") or "") == "tp"), 4)
        if any((row.get("review_verdict") or "") == "tp" for row in rows)
        else None,
        "note": "measured on this run's crops; not re-scored by the joint policy fit",
    }


def command_verify_cards(args: argparse.Namespace) -> int:
    verdict_rows = [
        json.loads(line)
        for line in Path(args.verdicts).read_text().splitlines()
        if line.strip()
    ]
    summary = render_disagreement_cards(
        verdict_rows=verdict_rows,
        review_rows=load_review_rows(args.review),
        images_dir=args.images_dir,
        out_dir=args.out,
        min_confidence=args.min_confidence,
    )
    print(json.dumps(summary, indent=1))
    return 0


def command_verify_report(args: argparse.Namespace) -> int:
    from experiments.anatomy4.verify import (
        build_report as _build_report,
        load_openrouter_settings as _settings,
        reparse_raw_calls,
    )

    verdict_rows = [
        json.loads(line)
        for line in Path(args.verdicts).read_text().splitlines()
        if line.strip()
    ]
    if args.raw_calls:
        verdict_rows = reparse_raw_calls(
            raw_calls_path=args.raw_calls,
            verdicts_path=args.verdicts,
            known_names=(
                __import__(
                    "experiments.anatomy4.verify", fromlist=["load_known_names"]
                ).load_known_names(args.manifest)
                if args.manifest
                else {}
            ),
        )
        Path(args.out).mkdir(parents=True, exist_ok=True)
        with (Path(args.out) / "verdicts.jsonl").open("w") as handle:
            for row in verdict_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    review_rows = load_review_rows(args.review)
    settings = _settings(env_file=args.env_file, model=args.model)
    tightness = tuple(item.strip() for item in args.accept_tightness.split(",") if item.strip())
    previous_calls = 0
    previous_report = Path(args.out) / "report.json"
    if not args.raw_calls and previous_report.exists():
        try:
            previous_calls = int(json.loads(previous_report.read_text()).get("calls_used") or 0)
        except (ValueError, OSError):
            previous_calls = 0
    report = _build_report(
        review_rows=review_rows,
        verdict_rows=verdict_rows,
        settings=settings,
        review_path=args.review,
        proposals_path=args.proposals or "",
        crops_dir=str(Path(args.verdicts).parent / "crops"),
        calls_used=len(
            {
                json.loads(line).get("request_id")
                for line in Path(args.raw_calls).read_text().splitlines()
                if line.strip()
            }
        )
        if args.raw_calls
        else (previous_calls or len(verdict_rows)),
        skipped=[],
        min_confidence=args.min_confidence,
        batch_size=1,
        accepted_tightness=tightness,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (Path(args.out) / "report.md").write_text(render_report_markdown(report))
    print(render_report_markdown(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anatomy4 Shenandoah bootstrap")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-manifest")
    build.add_argument("--area-instance-id", default=AREA_INSTANCE_ID)
    build.add_argument("--sample-size", type=int, default=120)
    build.add_argument("--seed", default="anatomy4-v1")
    build.add_argument("--plant-fraction", type=float, default=0.72)
    build.add_argument("--needle-minimum", type=int, default=10)
    build.add_argument("--mode", choices=("broad", "targeted"), default="broad")
    build.add_argument("--needle-observations", type=int, default=60)
    build.add_argument("--photos-per-needle-observation", type=int, default=2)
    build.add_argument("--leaf-observations", type=int, default=110)
    build.add_argument("--flower-observations", type=int, default=126)
    build.add_argument("--background-observations", type=int, default=48)
    build.add_argument("--db-container", default="arq-compose-postgres")
    build.add_argument("--db-user", default="arq")
    build.add_argument("--db-name", default="arq")
    build.add_argument("--out", required=True)
    build.set_defaults(func=command_build_manifest)

    download = sub.add_parser("download")
    download.add_argument("--manifest", required=True)
    download.add_argument("--images-dir", required=True)
    download.add_argument("--vault-root", default=None)
    download.add_argument("--vault-base-url", default=None)
    download.add_argument("--workers", type=int, default=6)
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--out-manifest", default=None)
    download.set_defaults(func=command_download)

    prelabel = sub.add_parser("prelabel")
    prelabel.add_argument("--manifest", required=True)
    prelabel.add_argument("--config", required=True)
    prelabel.add_argument("--out", required=True)
    prelabel.add_argument("--device", default="cuda")
    prelabel.add_argument("--resume", action="store_true")
    prelabel.add_argument("--limit", type=int, default=None)
    prelabel.set_defaults(func=command_prelabel)

    yoloe = sub.add_parser("prelabel-yoloe")
    yoloe.add_argument("--manifest", required=True)
    yoloe.add_argument("--config", required=True)
    yoloe.add_argument("--out", required=True)
    yoloe.add_argument("--model", default="yoloe-26s-seg.pt")
    yoloe.add_argument("--imgsz", type=int, default=1024)
    yoloe.add_argument("--device", default="0")
    yoloe.add_argument("--conf-floor", type=float, default=0.03)
    yoloe.add_argument("--resume", action="store_true")
    yoloe.add_argument("--limit", type=int, default=None)
    yoloe.set_defaults(func=command_prelabel_yoloe)

    variants = sub.add_parser("prelabel-yoloe-variants")
    variants.add_argument("--manifest", required=True)
    variants.add_argument("--config", required=True)
    variants.add_argument("--variants", required=True)
    variants.add_argument("--out", required=True)
    variants.add_argument("--model", default="yoloe-26s-seg.pt")
    variants.add_argument("--imgsz", type=int, default=1024)
    variants.add_argument("--device", default="0")
    variants.add_argument("--conf-floor", type=float, default=0.02)
    variants.add_argument("--bucket", default=None)
    variants.add_argument("--limit", type=int, default=None)
    variants.set_defaults(func=command_prelabel_yoloe_variants)

    world = sub.add_parser("prelabel-world")
    world.add_argument("--manifest", required=True)
    world.add_argument("--config", required=True)
    world.add_argument("--out", required=True)
    world.add_argument("--model", default="yolov8s-worldv2.pt")
    world.add_argument("--imgsz", type=int, default=1280)
    world.add_argument("--device", default="0")
    world.add_argument("--conf-floor", type=float, default=0.01)
    world.add_argument("--max-per-class", type=int, default=10)
    world.add_argument("--bucket", default=None)
    world.add_argument("--limit", type=int, default=None)
    world.set_defaults(func=command_prelabel_world)

    owl = sub.add_parser("prelabel-owlv2")
    owl.add_argument("--manifest", required=True)
    owl.add_argument("--config", required=True)
    owl.add_argument("--out", required=True)
    owl.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    owl.add_argument("--device", default="cuda")
    owl.add_argument("--score-threshold", type=float, default=0.02)
    owl.add_argument("--max-per-class", type=int, default=10)
    owl.add_argument("--bucket", default=None)
    owl.add_argument("--limit", type=int, default=None)
    owl.set_defaults(func=command_prelabel_owlv2)

    hybrid = sub.add_parser("prelabel-hybrid")
    hybrid.add_argument("--manifest", required=True, help="train-candidate manifest.jsonl")
    hybrid.add_argument("--out", required=True)
    hybrid.add_argument("--classes", nargs="+", default=None)
    hybrid.add_argument("--device", default="cuda")
    hybrid.add_argument("--limit", type=int, default=None, help="images per class scope (smoke tests)")
    hybrid.add_argument(
        "--blind-manifest",
        action="append",
        default=None,
        help="blind batch manifest.jsonl (repeatable); overlap aborts the run",
    )
    hybrid.add_argument("--no-sheets", action="store_true")
    hybrid.set_defaults(func=command_prelabel_hybrid)

    reindex = sub.add_parser("reindex-hybrid")
    reindex.add_argument("--out", required=True)
    reindex.add_argument("--manifest", default=None)
    reindex.add_argument("--nms-iou", type=float, default=0.5)
    reindex.add_argument("--max-per-class", type=int, default=None)
    reindex.add_argument("--no-sheets", action="store_true")
    reindex.set_defaults(func=command_reindex_hybrid)

    alt_sheets = sub.add_parser("render-alt-sheets")
    alt_sheets.add_argument("--manifest", required=True)
    alt_sheets.add_argument("--out", required=True)
    alt_sheets.add_argument("--classes", nargs="+", default=["leaf", "needle", "flower", "inflorescence"])
    alt_sheets.add_argument("--per-class", type=int, default=24)
    alt_sheets.add_argument("--columns", type=int, default=3)
    alt_sheets.add_argument("--needle-detail", action="store_true")
    alt_sheets.set_defaults(func=command_render_alt_sheets)

    export = sub.add_parser("export-proposals")
    export.add_argument("--manifest", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--thresholds", default=None, help="JSON {class: threshold}")
    export.add_argument("--max-box-area", type=float, default=0.60)
    export.add_argument("--per-class-limit", type=int, default=None)
    export.set_defaults(func=command_export)

    compare = sub.add_parser("compare")
    compare.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("NAME", "MANIFEST", "REVIEW_CSV"),
        required=True,
    )
    compare.add_argument("--dataset", default="")
    compare.add_argument("--out", required=True)
    compare.set_defaults(func=command_compare)

    render = sub.add_parser("render")
    render.add_argument("--manifest", required=True)
    render.add_argument("--out", required=True)
    render.add_argument("--sample-per-class", type=int, default=24)
    render.add_argument("--background-per-bucket", type=int, default=32)
    render.set_defaults(func=command_render)

    second = sub.add_parser("needle-second-pass")
    second.add_argument("--manifest", required=True)
    second.add_argument("--out", required=True)
    second.add_argument("--model", default="yoloe-26s-seg.pt")
    second.add_argument("--imgsz", type=int, default=1024)
    second.add_argument("--device", default="0")
    second.add_argument("--crop-margin", type=float, default=0.15)
    second.add_argument("--min-candidate-score", type=float, default=0.05)
    second.add_argument("--prompts", nargs="+", default=None)
    second.add_argument("--conf-floor", type=float, default=0.02)
    second.add_argument("--limit", type=int, default=None)
    second.add_argument("--keep-first", action="store_true")
    second.set_defaults(func=command_needle_second_pass)

    clip_filter = sub.add_parser("needle-clip-filter")
    clip_filter.add_argument("--second-pass", required=True)
    clip_filter.add_argument("--out", required=True)
    clip_filter.add_argument("--model", default="ViT-B/32")
    clip_filter.add_argument("--device", default="cuda")
    clip_filter.add_argument("--batch-size", type=int, default=32)
    clip_filter.add_argument("--download-root", default="/models/clip")
    clip_filter.set_defaults(func=command_needle_clip_filter)

    merge = sub.add_parser("merge-runs")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=command_merge_runs)

    labels = sub.add_parser("labels")
    labels.add_argument("--manifest", required=True)
    labels.add_argument("--review", default=None)
    labels.add_argument("--verdicts", default=None, help="verify-crops verdicts.jsonl")
    labels.add_argument("--verify-profile", default="balanced", choices=["strict", "balanced", "triage"])
    labels.add_argument("--require-both", action="store_true")
    labels.add_argument("--out", required=True)
    labels.set_defaults(func=command_labels)

    audit = sub.add_parser("audit")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--review", default=None)
    audit.add_argument("--background-review", default=None)
    audit.add_argument("--out", required=True)
    audit.set_defaults(func=command_audit)

    verify = sub.add_parser("verify-crops")
    verify.add_argument("--proposals", required=True)
    verify.add_argument("--images-dir", required=True)
    verify.add_argument("--review", required=True)
    verify.add_argument("--manifest", default=None)
    verify.add_argument("--out", required=True)
    verify.add_argument("--model", default=None, help="OpenRouter model id (default: OPENROUTER_MODEL or openai/gpt-4o-mini)")
    verify.add_argument("--env-file", default=None)
    verify.add_argument("--batch-size", type=int, default=4)
    verify.add_argument("--margin", type=float, default=0.18)
    verify.add_argument("--max-calls", type=int, default=200)
    verify.add_argument("--min-confidence", type=float, default=0.7)
    verify.add_argument("--attempts", type=int, default=3)
    verify.add_argument("--second-pass-margin", type=float, default=0.55)
    verify.add_argument("--second-pass-confidence", type=float, default=0.7)
    verify.add_argument("--no-second-pass", action="store_true")
    verify.add_argument("--limit", type=int, default=None)
    verify.add_argument("--sample-per-class", type=int, default=None)
    verify.add_argument("--proposal-ids", default=None, help="Comma-separated targeted subset")
    verify.add_argument("--prompt-variant", default="v1", choices=["v1", "main_subject"])
    verify.add_argument("--crop-max-side", type=int, default=900)
    verify.set_defaults(func=command_verify_crops)

    train_verify = sub.add_parser("train-verify")
    train_verify.add_argument("--yoloe-proposals", required=True)
    train_verify.add_argument("--owlv2-proposals", required=True)
    train_verify.add_argument("--manifest", required=True)
    train_verify.add_argument("--images-dir", required=True)
    train_verify.add_argument("--yoloe-policy", required=True)
    train_verify.add_argument("--owlv2-policy", required=True)
    train_verify.add_argument("--out", required=True)
    train_verify.add_argument("--model", default="openai/gpt-4o-mini")
    train_verify.add_argument("--env-file", default=None)
    train_verify.add_argument("--leaf-count", type=int, default=400)
    train_verify.add_argument("--inflorescence-count", type=int, default=400)
    train_verify.add_argument("--flower-count", type=int, default=400)
    train_verify.add_argument("--max-per-photo", type=int, default=4)
    train_verify.add_argument("--selection-seed", default="anatomy4-production-train-verify-v1")
    train_verify.add_argument("--batch-size", type=int, default=4)
    train_verify.add_argument("--workers", type=int, default=4)
    train_verify.add_argument("--max-crops", type=int, default=1200)
    train_verify.add_argument("--attempts", type=int, default=3)
    train_verify.add_argument("--audit-fraction", type=float, default=0.10)
    train_verify.add_argument("--audit-seed", default="anatomy4-curator-audit-v1")
    train_verify.add_argument("--resume", action="store_true", help="Reuse successful raw_calls.jsonl responses in an existing output directory")
    train_verify.set_defaults(func=command_train_verify)

    policy_fit = sub.add_parser("policy-fit")
    policy_fit.add_argument("--proposals", required=True)
    policy_fit.add_argument("--verdicts", required=True)
    policy_fit.add_argument("--review", required=True)
    policy_fit.add_argument("--out", required=True)
    policy_fit.add_argument("--model", default=None)
    policy_fit.add_argument("--detector", default=None, help="Proposal source label, e.g. yoloe-26s / owlv2-base")
    policy_fit.add_argument("--target-precision", type=float, default=0.90)
    policy_fit.add_argument("--min-accepts", type=int, default=5)
    policy_fit.add_argument("--min-assisted-support", type=int, default=5)
    policy_fit.add_argument("--min-assisted-precision", type=float, default=0.90)
    policy_fit.add_argument("--loo-min-precision", type=float, default=0.85)
    policy_fit.add_argument("--validation-verdicts", default=None)
    policy_fit.add_argument("--validation-review", default=None)
    policy_fit.add_argument("--validation-min-precision", type=float, default=0.80)
    policy_fit.add_argument("--validation-min-accepts", type=int, default=3)
    policy_fit.add_argument("--background-verdicts", default=None)
    policy_fit.add_argument("--background-review", default=None)
    policy_fit.add_argument(
        "--needle-mainsubject-verdicts",
        default=None,
        help="Optional measured alternative verifier run (main_subject needle).",
    )
    policy_fit.set_defaults(func=command_policy_fit)

    verify_cards = sub.add_parser("verify-cards")
    verify_cards.add_argument("--verdicts", required=True)
    verify_cards.add_argument("--review", required=True)
    verify_cards.add_argument("--images-dir", required=True)
    verify_cards.add_argument("--out", required=True)
    verify_cards.add_argument("--min-confidence", type=float, default=0.7)
    verify_cards.set_defaults(func=command_verify_cards)

    verify_report = sub.add_parser("verify-report")
    verify_report.add_argument("--verdicts", required=True)
    verify_report.add_argument("--review", required=True)
    verify_report.add_argument("--out", required=True)
    verify_report.add_argument("--raw-calls", default=None, help="Re-parse stored raw responses (no spend)")
    verify_report.add_argument("--manifest", default=None)
    verify_report.add_argument("--proposals", default=None)
    verify_report.add_argument("--model", default=None)
    verify_report.add_argument("--env-file", default=None)
    verify_report.add_argument("--accept-tightness", default="keep")
    verify_report.add_argument("--min-confidence", type=float, default=0.7)
    verify_report.set_defaults(func=command_verify_report)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
