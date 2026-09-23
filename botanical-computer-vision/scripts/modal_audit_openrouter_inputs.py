"""Audit the exact image preprocessing used by the OpenRouter benchmark on Modal."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import modal


DRIVE_URL = "https://drive.google.com/uc?id=18U0D61HWmdFH9kyPSuog6mD6zKgfugTc"
MAX_IMAGE_SIDE = 1536
SAMPLE_IDS = {
    "baskauf/38299",
    "baskauf/50998",
    "baskauf/66166",
    "baskauf/89954",
}

runtime = modal.Image.debian_slim(python_version="3.11").pip_install(
    "gdown==5.2.0",
    "pillow==11.3.0",
)
app = modal.App("bioimages-openrouter-input-audit", image=runtime)


@app.function(timeout=30 * 60, cpu=4, memory=8192)
def audit_inputs() -> bytes:
    import base64
    import hashlib
    import zipfile

    import gdown
    from PIL import Image, ImageFile, ImageOps

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    zip_path = "/tmp/bioimages.zip"
    data_root = Path("/tmp/bioimages")
    if not gdown.download(DRIVE_URL, zip_path, quiet=False, fuzzy=True):
        raise RuntimeError("Google Drive download failed")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_root)

    root = data_root / "bioimages-ne-trees"
    corpus = json.loads((root / "corpus.json").read_text())
    images_dir = root / "images"
    records = []
    sample_files: dict[str, bytes] = {}
    filenames = set()
    image_ids = set()

    for scene in corpus["scenes"]:
        species = scene["scientific_name"].strip()
        for image_record in scene["images"]:
            image_id = image_record["image_id"]
            filename = image_record["file"]
            path = images_dir / filename
            original = path.read_bytes()
            output = io.BytesIO()
            with Image.open(io.BytesIO(original)) as opened:
                original_format = opened.format
                original_mode = opened.mode
                original_size = list(opened.size)
                transformed = ImageOps.exif_transpose(opened).convert("RGB")
                transformed.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
                transformed.save(output, format="JPEG", quality=90, optimize=True)
            encoded = output.getvalue()
            with Image.open(io.BytesIO(encoded)) as checked:
                checked.load()
                encoded_size = list(checked.size)
                encoded_format = checked.format
                encoded_mode = checked.mode
                encoded_exif_keys = sorted(str(key) for key in checked.getexif().keys())

            # This round-trip is the exact string form inserted into image_url.url.
            data_url = "data:image/jpeg;base64," + base64.b64encode(encoded).decode("ascii")
            roundtrip = base64.b64decode(data_url.split(",", 1)[1])
            assert roundtrip == encoded

            filenames.add(filename)
            image_ids.add(image_id)
            records.append(
                {
                    "image_id": image_id,
                    "file": filename,
                    "ground_truth_species": species,
                    "original_format": original_format,
                    "original_mode": original_mode,
                    "original_size": original_size,
                    "original_bytes": len(original),
                    "original_sha256": hashlib.sha256(original).hexdigest(),
                    "encoded_format": encoded_format,
                    "encoded_mode": encoded_mode,
                    "encoded_size": encoded_size,
                    "encoded_bytes": len(encoded),
                    "encoded_sha256": hashlib.sha256(encoded).hexdigest(),
                    "encoded_exif_keys": encoded_exif_keys,
                    "data_url_prefix": data_url[:23],
                    "base64_roundtrip_exact": roundtrip == encoded,
                }
            )
            if image_id in SAMPLE_IDS:
                sample_files[f"samples/{image_id.replace('/', '_')}.jpg"] = encoded

    assert len(records) == 1899
    assert len(image_ids) == 1899
    assert len(filenames) == 1899
    assert len(sample_files) == len(SAMPLE_IDS)
    assert all(record["encoded_format"] == "JPEG" for record in records)
    assert all(record["encoded_mode"] == "RGB" for record in records)
    assert all(record["base64_roundtrip_exact"] for record in records)
    assert all(not record["encoded_exif_keys"] for record in records)

    summary = {
        "dataset_images": len(records),
        "unique_image_ids": len(image_ids),
        "unique_filenames": len(filenames),
        "decoded_and_reencoded": len(records),
        "jpeg_rgb_outputs": sum(
            record["encoded_format"] == "JPEG" and record["encoded_mode"] == "RGB"
            for record in records
        ),
        "base64_roundtrip_exact": sum(record["base64_roundtrip_exact"] for record in records),
        "metadata_free_outputs": sum(not record["encoded_exif_keys"] for record in records),
        "sample_ids": sorted(SAMPLE_IDS),
        "records": records,
    }

    files = {
        "input_integrity.json": json.dumps(summary, indent=2).encode(),
        **sample_files,
    }
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


@app.local_entrypoint()
def main(output: str = "outputs/openrouter_vlm_audit") -> None:
    payload = audit_inputs.remote()
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        archive.extractall(output_path)
    summary = json.loads((output_path / "input_integrity.json").read_text())
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
