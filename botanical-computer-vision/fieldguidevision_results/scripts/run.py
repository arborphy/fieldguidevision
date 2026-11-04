"""Frozen OpenCLIP -> group means -> Stable unCLIP -> cosine retrieval audit.

Run --stage validate before renting a GPU. All paths are relative to --root.
Raw projected OpenCLIP embeddings condition the decoder; only retrieval uses L2
normalization. No training, refitting of clusters, or spherical interpolation.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFile

# Same policy as the repository's existing Modal embedding extraction.
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL = "ViT-H-14"
WEIGHTS = "laion2b_s32b_b79k"
SEEDS = [0, 1, 2, 3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(x).all() or np.any(norms < 1e-8):
        raise ValueError("Nonfinite or zero embedding")
    return x / norms


def majority(values):
    # Stable lexical tie break, recorded counts preserve ties in the audit.
    counts = Counter(str(v) for v in values)
    return sorted(counts, key=lambda v: (-counts[v], v))[0], dict(sorted(counts.items()))


def load_inputs(args):
    manifest = args.root / "organ_benchmark/split_manifests/individual_disjoint.csv"
    order_file = args.root / "data/embeddings/efficientnet_b0_strict/embeddings.npz"
    files = [manifest, order_file] + [args.root / args.groups / f"clusters_k{k}.npy" for k in (5, 7)]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing exact input files (do not refit):\n" + "\n".join(missing))
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {"image_id", "image_url", "organ_category", "species"} <= rows[0].keys():
        raise ValueError("Manifest is missing required columns")
    by_id = {r["image_id"]: r for r in rows}
    with np.load(order_file, allow_pickle=False) as data:
        ids = data["image_ids"].astype(str)
        splits = data["splits"].astype(str) if "splits" in data else None
    if len(rows) != 1641 or len(by_id) != 1641 or len(ids) != 1641 or len(set(ids)) != 1641:
        raise ValueError("Expected exactly 1,641 unique image IDs in each input")
    if set(ids) != set(by_id) or len({r["species"] for r in rows}) != 63:
        raise ValueError("Manifest/order mismatch or species count is not 63")
    rows = [by_id[i] for i in ids]
    if splits is not None and "split" in rows[0] and list(splits) != [r["split"] for r in rows]:
        raise ValueError("Split mismatch between manifest and EfficientNet order")
    groups = {}
    for k, path in zip((5, 7), files[2:]):
        labels = np.load(path, allow_pickle=False)
        if labels.shape != (1641,) or not np.issubdtype(labels.dtype, np.integer) or len(np.unique(labels)) != k:
            raise ValueError(f"Invalid K={k} cluster array: {labels.shape}, {labels.dtype}")
        groups[k] = labels
    provenance = {str(p.relative_to(args.root)): sha(p) for p in files}
    return ids, rows, groups, provenance


def build_targets(raw, groups, organs, fruit_fallback=False):
    targets = []
    for k in (5, 7):
        for group in sorted(np.unique(groups[k])):
            mask = groups[k] == group
            organ, counts = majority(organs[mask])
            targets.append({"id": f"k{k}_g{group}", "k": k, "group": int(group),
                            "majority_organ": organ, "organ_counts": counts,
                            "size": int(mask.sum()), "embedding": raw[mask].mean(axis=0)})
    endpoints = []
    for organ in ("leaf", "fruit"):
        choices = [t for t in targets if t["k"] == 7 and t["majority_organ"] == organ]
        if not choices and organ == "fruit" and fruit_fallback:
            choices = [t for t in targets if t["k"] == 7 and t["organ_counts"].get("fruit", 0) > 0]
        if not choices:
            raise ValueError(f"No K=7 group has majority organ '{organ}'")
        # If several qualify, highest purity, then support, then smallest group ID.
        endpoints.append(sorted(choices, key=lambda t: (
            -t["organ_counts"].get(organ, 0) / t["size"], -t["size"], t["group"]))[0])
    leaf, fruit = endpoints
    for i, alpha in enumerate(np.linspace(0, 1, 5)):
        targets.append({"id": f"walk_{i}", "k": 7, "group": leaf["group"] if i == 0 else fruit["group"] if i == 4 else None,
                        "alpha": float(alpha), "leaf_group": leaf["group"], "fruit_group": fruit["group"],
                        "fruit_endpoint_majority_organ": fruit["majority_organ"],
                        "fruit_endpoint_fruit_share": fruit["organ_counts"].get("fruit", 0) / fruit["size"],
                        "embedding": (1-alpha)*leaf["embedding"] + alpha*fruit["embedding"]})
    return targets


def download(rows, out):
    import httpx
    cache = out / "photos"
    cache.mkdir(exist_ok=True)

    def one(row):
        path = cache / (hashlib.sha256(row["image_id"].encode()).hexdigest() + ".png")
        if path.exists():
            try:
                with Image.open(path) as im:
                    im.verify()
                return path
            except Exception:
                path.unlink()
        error = None
        for _ in range(3):
            try:
                with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": "FieldGuideVision-research/1.0"}) as client:
                    response = client.get(row["image_url"])
                    response.raise_for_status()
                    payload = response.content
                with Image.open(io.BytesIO(payload)) as im:
                    temp = path.with_suffix('.tmp.png')
                    ImageOps.exif_transpose(im).convert("RGB").save(temp)
                    temp.replace(path)
                return path
            except Exception as exc:
                error = exc
        print(f"Recovering from original corpus archive: {row['image_id']} ({type(error).__name__})", flush=True)
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(one, rows))
    missing = [i for i, p in enumerate(paths) if p is None]
    if missing:
        import gdown
        archive_path = out.parent / 'bioimages_source.zip'
        if not archive_path.exists():
            if not gdown.download(id='18U0D61HWmdFH9kyPSuog6mD6zKgfugTc', output=str(archive_path), quiet=False):
                raise RuntimeError('Original repository corpus archive download failed')
        with zipfile.ZipFile(archive_path) as archive:
            corpus = json.loads(archive.read('bioimages-ne-trees/corpus.json'))
            original = {str(im['image_id']): im['file'] for scene in corpus['scenes'] for im in scene['images']}
            if len(original) != 1899 or not {r['image_id'] for r in rows} <= original.keys():
                raise ValueError('Original corpus does not contain the exact strict image IDs')
            for i in missing:
                row = rows[i]
                path = cache / (hashlib.sha256(row['image_id'].encode()).hexdigest() + '.png')
                with Image.open(io.BytesIO(archive.read('bioimages-ne-trees/images/' + original[row['image_id']]))) as im:
                    ImageOps.exif_transpose(im).convert('RGB').save(path)
                paths[i] = path
        (out / 'photo_recovery.json').write_text(json.dumps({
            'source': 'Original BioImages corpus archive used by repository modal_full_embeddings.py',
            'drive_file_id': '18U0D61HWmdFH9kyPSuog6mD6zKgfugTc',
            'archive_sha256': sha(archive_path),
            'recovered_image_ids': [rows[i]['image_id'] for i in missing],
            'selection': 'Exact image_id match; all 1641 strict images retained',
        }, indent=2))
    print(f'Photos ready: {len(paths)}', flush=True)
    return paths


def encoder():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=WEIGHTS, device="cuda")
    return model.eval(), preprocess


def encode(paths, model, preprocess, batch_size):
    import torch
    parts = []
    for start in range(0, len(paths), batch_size):
        batch = []
        for path in paths[start:start+batch_size]:
            with Image.open(path) as im:
                batch.append(preprocess(im.convert("RGB")))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            features = model.encode_image(torch.stack(batch).to("cuda"), normalize=False)
        parts.append(features.float().cpu().numpy())
    result = np.concatenate(parts)
    unit(result)  # Validate, without changing the decoder conditioning vectors.
    if result.shape[1] != 1024:
        raise ValueError(f"Expected ViT-H 1024-D output, got {result.shape}")
    return result


def nearest(query, real, n=10):
    return np.argsort(-(unit(np.asarray(query)[None, :]) @ unit(real).T)[0], kind="stable")[:n]


def audit_target(target, generated, real, labels, organs, ids):
    target_neighbors = nearest(target["embedding"], real)
    target_organ, target_counts = majority(organs[target_neighbors])
    samples = []
    for vector, seed in zip(generated, SEEDS):
        neighbors = nearest(vector, real)
        organ, counts = majority(organs[neighbors])
        samples.append({"seed": seed, "nearest_image_id": str(ids[neighbors[0]]),
                        "nearest_group": int(labels[neighbors[0]]), "top10_majority_organ": organ,
                        "top10_organ_counts": counts, "top10_image_ids": ids[neighbors].tolist(),
                        "target_cosine": float(unit(vector) @ unit(target["embedding"]))})
    result = {k: v for k, v in target.items() if k != "embedding"}
    result.update({"target_top10_majority_organ": target_organ, "target_top10_organ_counts": target_counts,
                   "samples": samples})
    if target["group"] is not None:
        count = sum(s["nearest_group"] == target["group"] for s in samples)
        result.update({"target_group_hits": count, "generated_count": len(samples), "target_group_share": count / len(samples)})
    else:
        # An interpolated point has no given EfficientNet label. Do not invent truth.
        result["target_group_share"] = None
    if "alpha" in target:
        result["endpoint_group_shares"] = {str(g): sum(s["nearest_group"] == g for s in samples) / len(samples)
                                            for g in (target["leaf_group"], target["fruit_group"])}
    return result


def sheet(targets, out, name, walk=False):
    # Walk: five columns (alpha) x four rows (seed). Group sheets: groups x seeds.
    cell, header = 256, 30
    cols, rows = (5, 4) if walk else (4, len(targets))
    top = 45 if walk else 0
    canvas = Image.new("RGB", (cols*cell, rows*(cell+header)+top), "white")
    draw = ImageDraw.Draw(canvas)
    if walk:
        t = targets[0]
        draw.text((6, 6), f"leaf g{t['leaf_group']} -> fruit-enriched g{t['fruit_group']}; actual endpoint majority: {t['fruit_endpoint_majority_organ']}", fill="black")
        draw.text((6, 23), f"fruit share at endpoint: {t['fruit_endpoint_fruit_share']:.1%}", fill="black")
    for i, target in enumerate(targets):
        for j, seed in enumerate(SEEDS):
            col, row = (i, j) if walk else (j, i)
            x, y = col*cell, row*(cell+header)+top
            title = f"alpha={target['alpha']:.2f}, seed={seed}" if walk else f"{target['id']} {target['majority_organ']} seed={seed}"
            draw.text((x+4, y+6), title, fill="black")
            with Image.open(out / "generated" / f"{target['id']}_seed{seed}.png") as im:
                canvas.paste(ImageOps.fit(im.convert("RGB"), (cell, cell)), (x, y+header))
    canvas.save(out / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--groups", type=Path, default=Path("unclip_mvp/groups"))
    parser.add_argument("--fruit-fallback", action="store_true", help="If no fruit-majority group exists, explicitly use the highest-fruit-share group")
    parser.add_argument("--output", type=Path, default=Path("outputs/unclip_mvp"))
    parser.add_argument("--stage", choices=["validate", "encode", "smoke", "all"], default="validate")
    parser.add_argument("--decoder", default="stabilityai/stable-diffusion-2-1-unclip")
    parser.add_argument("--decoder-revision", help="Pinned HF commit; required for generation")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    ids, rows, groups, provenance = load_inputs(args)
    print(f"Validated {len(ids)} photos, 63 species, K=5 and K=7; input hashes: {provenance}")
    if args.stage == "validate":
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    if args.stage in ("smoke", "all") and not args.decoder_revision:
        parser.error("Generation requires --decoder-revision (HF commit SHA)")
    out = args.root / args.output
    out.mkdir(parents=True, exist_ok=True)
    config = {"inputs": provenance, "encoder": MODEL, "weights": WEIGHTS,
              "decoder": args.decoder, "decoder_revision": args.decoder_revision,
              "seeds": SEEDS, "steps": args.steps, "noise_level": 0, "prompt": "",
              "guidance_scale": 7.5, "resolution": 768,
              "fruit_fallback": args.fruit_fallback,
              "image_loader": "Pillow LOAD_TRUNCATED_IMAGES=True (matches existing repo extraction)",
              "centres": "arithmetic mean of raw unnormalized projected features",
              "retrieval": "cosine over all 1641 real photos; descriptive, not held-out evaluation"}
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output directory has a different config; choose a new --output")
    config_path.write_text(json.dumps(config, indent=2))
    cache = out / "openclip_raw.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if not np.array_equal(ids, data["image_ids"]):
                raise ValueError("Cached embedding order mismatch")
            raw = data["embeddings"]
    else:
        paths = download(rows, out)
        model, preprocess = encoder()
        raw = encode(paths, model, preprocess, args.batch_size)
        del model, preprocess
        gc.collect(); torch.cuda.empty_cache()
        np.savez_compressed(cache, image_ids=ids, embeddings=raw)
    organs = np.array([r["organ_category"] for r in rows])
    targets = build_targets(raw, groups, organs, args.fruit_fallback)
    if args.stage == "encode":
        return
    if args.stage == "smoke":
        targets = targets[:1]
    np.savez_compressed(out / ("smoke_targets.npz" if args.stage == "smoke" else "targets.npz"),
                        target_ids=np.array([t["id"] for t in targets]), embeddings=np.stack([t["embedding"] for t in targets]))
    from diffusers import StableUnCLIPImg2ImgPipeline
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(args.decoder, revision=args.decoder_revision,
                                                       torch_dtype=torch.float16)
    pipe.to("cuda")
    if pipe.image_encoder.config.projection_dim != 1024:
        raise ValueError("Decoder image projection is not ViT-H 1024-D; do not use unclip-small")
    pipe.enable_attention_slicing()
    (out / "generated").mkdir(exist_ok=True)
    generated_paths = []
    for target in targets:
        for seed in SEEDS:
            path = out / "generated" / f"{target['id']}_seed{seed}.png"
            if not path.exists():
                image = pipe(prompt="", image_embeds=torch.tensor(target["embedding"][None], device="cuda", dtype=torch.float16),
                             noise_level=0, generator=torch.Generator(device="cuda").manual_seed(seed),
                             num_inference_steps=args.steps, guidance_scale=7.5, height=768, width=768).images[0]
                image.save(path)
            generated_paths.append(path)
    del pipe
    gc.collect(); torch.cuda.empty_cache()
    model, preprocess = encoder()
    generated = encode(generated_paths, model, preprocess, args.batch_size)
    import importlib.metadata
    results = {"config": config, "gpu": torch.cuda.get_device_name(0),
               "photo_recovery": json.loads((out / 'photo_recovery.json').read_text()) if (out / 'photo_recovery.json').exists() else None,
               "versions": {p: importlib.metadata.version(p) for p in ("torch", "open_clip_torch", "diffusers", "transformers")},
               "walk_note": "Interior points have no prescribed target group; share=null; both endpoint shares reported.",
               "points": [audit_target(t, generated[i*4:(i+1)*4], raw, groups[t["k"]], organs, ids)
                          for i, t in enumerate(targets)]}
    (out / ("smoke_checks.json" if args.stage == "smoke" else "checks.json")).write_text(json.dumps(results, indent=2))
    if args.stage == "all":
        for k in (5, 7):
            sheet([t for t in targets if t["k"] == k and "alpha" not in t], out, f"contact_k{k}.png")
        sheet([t for t in targets if "alpha" in t], out, "leaf_to_fruit.png", walk=True)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
