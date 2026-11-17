# @arq feature=F-FAC-YOLO-ANATOMY-LAB process=P-ANATOMY-DATASET-BUILD role=canonical
"""Fetch audit images and stamp immutable download facts (SHA-256, bytes)."""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

USER_AGENT = "ArborphyAnatomy4Audit/1.0 (+https://arborphy.ai)"
VAULT_CANDIDATES = ("large.jpg", "large.jpeg", "large.png", "original.jpg", "original.jpeg")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extension(url: str | None) -> str:
    if not url:
        return ".jpg"
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in (".jpg", ".jpeg", ".png", ".webp"):
        return suffix
    return ".jpg"


def _vault_local(vault_root: Path | None, photo_id: str) -> Path | None:
    if vault_root is None:
        return None
    directory = vault_root / photo_id
    if not directory.is_dir():
        return None
    for candidate in VAULT_CANDIDATES:
        path = directory / candidate
        if path.is_file() and path.stat().st_size > 0:
            return path
    matches = sorted(p for p in directory.glob("large.*") if p.is_file())
    return matches[0] if matches else None


def _vault_url(base: str | None, photo_id: str, extension: str) -> str | None:
    if not base:
        return None
    return f"{base.rstrip('/')}/{photo_id}/large{extension}"


def _download(url: str, target: Path, timeout: int = 60) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _fetch_one(
    image: dict[str, Any],
    images_dir: Path,
    vault_root: Path | None,
    vault_base_url: str | None,
    overwrite: bool,
) -> tuple[str, dict[str, Any]]:
    photo_id = str(image["photo_id"])
    extension = _extension(image.get("url"))
    target = images_dir / f"{photo_id}{extension}"

    local_vault = _vault_local(vault_root, photo_id)
    if local_vault and not overwrite:
        if target.exists() and target.stat().st_size == local_vault.stat().st_size:
            pass
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_vault, target)
        source = "vault-local"
        source_ref = str(local_vault)
    elif target.exists() and not overwrite:
        source = image.get("download", {}).get("source", "cache")
        source_ref = image.get("download", {}).get("source_ref")
    else:
        source = None
        source_ref = None
        errors: list[str] = []
        candidates: list[tuple[str, str]] = []
        if vault_base_url:
            candidates.append((_vault_url(vault_base_url, photo_id, extension) or "", "vault-http"))
        if image.get("url"):
            candidates.append((str(image["url"]), "source-url"))
        for url, label in candidates:
            if not url:
                continue
            try:
                _download(url, target)
                source, source_ref = label, url
                break
            except (urllib.error.URLError, TimeoutError, OSError) as exc:  # noqa: PERF203
                errors.append(f"{label}:{type(exc).__name__}")
        if source is None:
            return photo_id, {
                "ok": False,
                "errors": errors,
                "path": None,
            }

    actual_bytes = target.stat().st_size
    digest = sha256_file(target)
    width = image.get("width")
    height = image.get("height")
    try:
        from PIL import Image

        with Image.open(target) as img:
            width, height = img.size
    except Exception:  # noqa: BLE001 - dimensions are metadata, not correctness
        pass
    return photo_id, {
        "ok": True,
        "path": str(target),
        "sha256": digest,
        "bytes": actual_bytes,
        "width": width,
        "height": height,
        "source": source,
        "source_ref": source_ref,
        "downloaded_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
    }


def download_manifest(
    manifest: dict[str, Any],
    images_dir: str | Path,
    *,
    vault_root: str | Path | None = None,
    vault_base_url: str | None = None,
    workers: int = 6,
    overwrite: bool = False,
) -> dict[str, Any]:
    target_dir = Path(images_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    vault_path = Path(vault_root) if vault_root else None
    selected = [
        image for image in manifest.get("images", []) if image.get("selected_for_audit")
    ]
    if not selected:
        raise ValueError("manifest has no images with selected_for_audit=true")

    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _fetch_one, image, target_dir, vault_path, vault_base_url, overwrite
            ): image
            for image in selected
        }
        for future in as_completed(futures):
            image = futures[future]
            photo_id, result = future.result()
            if not result["ok"]:
                failures.append({"photo_id": photo_id, "errors": result["errors"]})
                continue
            image["download"] = {
                key: result[key]
                for key in (
                    "sha256",
                    "bytes",
                    "width",
                    "height",
                    "source",
                    "source_ref",
                    "downloaded_at",
                )
            }
            image["download"]["path"] = result["path"]

    manifest["download_summary"] = {
        "selected": len(selected),
        "downloaded": len(selected) - len(failures),
        "failed": len(failures),
        "failures": failures,
        "images_dir": str(target_dir),
    }
    return manifest
