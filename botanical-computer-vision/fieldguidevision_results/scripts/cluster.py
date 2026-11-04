"""Rebuild descriptive EfficientNet groups, explicitly authorized 2026-11-04.

These are NEW partitions, not recovered meeting labels. Fit all 1,641 images.
"""
import csv
import hashlib
import json
from pathlib import Path
import importlib.metadata

import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/embeddings/efficientnet_b0_strict/embeddings.npz"
MANIFEST = ROOT / "organ_benchmark/split_manifests/individual_disjoint.csv"
OUT = ROOT / "unclip_mvp/groups"


def main():
    from run import majority
    with np.load(SOURCE, allow_pickle=False) as data:
        ids = data["image_ids"].astype(str)
        features = data["embeddings"].astype(np.float32)
    with MANIFEST.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {r["image_id"]: r for r in rows}
    if len(ids) != 1641 or len(set(ids)) != 1641 or set(ids) != set(by_id):
        raise ValueError("Strict manifest/order mismatch")
    if not np.isfinite(features).all() or not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4):
        raise ValueError("Expected finite, unit EfficientNet vectors")
    OUT.mkdir(exist_ok=True)
    record = {"status": "newly recomputed; original meeting labels unavailable",
              "fit_scope": "all 1641 photos, descriptive only; no held-out score",
              "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "algorithm": "sklearn KMeans, Euclidean on existing L2-normalized EfficientNet features",
              "random_state": 42, "n_init": 20, "max_iter": 300, "tol": 0.0001,
              "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn")},
              "partitions": {}}
    assignments = []
    for k in (5, 7):
        with threadpool_limits(limits=1):
            model = KMeans(n_clusters=k, random_state=42, n_init=20, algorithm="lloyd", max_iter=300, tol=1e-4).fit(features)
        labels = model.labels_.astype(np.int64)
        path = OUT / f"clusters_k{k}.npy"
        np.save(path, labels, allow_pickle=False)
        composition = []
        for g in range(k):
            organ, counts = majority([by_id[ids[i]]["organ_category"] for i in np.flatnonzero(labels == g)])
            composition.append({"group": g, "size": int((labels == g).sum()), "majority_organ": organ, "organ_counts": counts})
        record["partitions"][str(k)] = {"inertia": float(model.inertia_), "iterations": int(model.n_iter_),
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "groups": composition}
        assignments.extend({"image_id": str(i), "k": k, "group": int(g), "organ_category": by_id[i]["organ_category"]} for i, g in zip(ids, labels))
        print(f"K={k}: " + ", ".join(f"g{g['group']}={g['majority_organ']} ({g['size']})" for g in composition))
    np.save(OUT / "image_ids.npy", ids, allow_pickle=False)
    with (OUT / "assignments.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "k", "group", "organ_category"])
        writer.writeheader(); writer.writerows(assignments)
    (OUT / "provenance.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
