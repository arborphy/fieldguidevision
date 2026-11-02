"""Frozen YOLO11s embeddings on the exact original 1,641-image cohort."""
import argparse
import csv
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path


def digest(path):
    sha = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def run(args):
    import numpy as np
    import torch
    import ultralytics
    from PIL import Image, ImageFile
    from ultralytics import YOLO

    started = time.perf_counter()
    torch.manual_seed(42)
    torch.set_num_threads(4)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    reference = np.load(args.reference, allow_pickle=False)
    ids = reference["image_ids"].astype(str)
    assert len(ids) == len(set(ids)) == 1641
    archive = zipfile.ZipFile(args.images)
    corpus = json.loads(archive.read("bioimages-ne-trees/corpus.json"))
    rows = {}
    for scene in corpus["scenes"]:
        for image in scene["images"]:
            rows[image["image_id"]] = {**image, "species": scene["scientific_name"]}
    missing = [image_id for image_id in ids if image_id not in rows or "bioimages-ne-trees/images/" + rows[image_id]["file"] not in archive.namelist()]
    assert not missing, f"Missing original images: {missing[:10]}"
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("Requested GPU but CUDA is unavailable")

    model = YOLO(args.weights)
    layers = [10, 16, 19, 22]
    modules = model.model.model
    assert "C2PSA" in modules[10].type and "Detect" in modules[-1].type
    assert list(modules[-1].f) == [16, 19, 22]
    model.model.eval()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    channels = {}
    handles = []
    for index in layers:
        def record_channels(module, inputs, output, index=index):
            channels[index] = output.shape[1]
        handles.append(modules[index].register_forward_hook(record_channels))
    backbone_parts, neck_parts = [], []
    inference_started = time.perf_counter()
    for start in range(0, len(ids), args.batch):
        batch_ids = ids[start:start + args.batch]
        images = []
        for image_id in batch_ids:
            content = archive.read("bioimages-ne-trees/images/" + rows[image_id]["file"])
            with Image.open(io.BytesIO(content)) as image:
                images.append(image.convert("RGB"))
        with torch.inference_mode():
            results = model.predict(images, embed=layers, imgsz=640, rect=False,
                                    device=args.device, half=False, augment=False, verbose=False)
            vectors = torch.stack(list(results)).float()
            blocks = torch.split(vectors, [channels[i] for i in layers], dim=1)
            assert vectors.shape[0] == len(batch_ids)
            assert all(torch.isfinite(block).all() and torch.all(block.norm(dim=1) > 0) for block in blocks)
            backbone = torch.nn.functional.normalize(blocks[0], dim=1)
            neck = torch.nn.functional.normalize(torch.cat([torch.nn.functional.normalize(block, dim=1) for block in blocks[1:]], dim=1), dim=1)
        backbone_parts.append(backbone.cpu().numpy())
        neck_parts.append(neck.cpu().numpy())
        elapsed = time.perf_counter() - inference_started
        print(json.dumps({"completed": min(start+args.batch, len(ids)), "total": len(ids), "seconds": round(elapsed, 2), "channels": channels}), flush=True)
        for image in images:
            image.close()
    for handle in handles:
        handle.remove()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    products = []
    for name, parts, selected in [("backbone", backbone_parts, [10]), ("neck_multiscale", neck_parts, [16,19,22])]:
        matrix = np.concatenate(parts).astype(np.float32)
        assert matrix.shape[0] == 1641 and np.isfinite(matrix).all()
        assert np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5)
        fields = {"image_ids": ids, "embeddings": matrix, "model": np.asarray("Ultralytics YOLO11s COCO pretrained detection"), "normalized": np.asarray(True), "layers": np.asarray(selected), "feature_type": np.asarray(name)}
        if "splits" in reference.files:
            fields["splits"] = reference["splits"]
        target = output / f"yolo11s_{name}_embeddings.npz"
        np.savez_compressed(target, **fields)
        products.append({"file": target.name, "shape": list(matrix.shape), "sha256": digest(target)})
    manifest = output / "image_manifest.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image_id", "file", "species", "organ_category", "individual_id"])
        writer.writeheader()
        for image_id in ids:
            writer.writerow({"image_id": image_id, **{k: rows[image_id].get(k, "") for k in ["file", "species", "organ_category", "individual_id"]}})
    summary = {"products": products, "images": len(ids), "device": args.device,
               "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
               "torch": torch.__version__, "ultralytics": ultralytics.__version__, "python": platform.python_version(),
               "weights_sha256": digest(args.weights), "reference_sha256": digest(args.reference), "image_archive_sha256": digest(args.images),
               "image_source": "Original bioimages-ne-trees archive; no report thumbnails",
               "input_size": 640, "rect": False, "precision": "float32", "training": False,
               "layers": channels, "backbone_pooling": "layer 10 global average pooling then L2 normalization",
               "neck_pooling": "global average pool layers 16,19,22; L2 each block, concatenate, L2 normalize",
               "total_seconds": time.perf_counter()-started}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    run(parser.parse_args())
