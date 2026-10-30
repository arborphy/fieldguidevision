"""Shared inputs for the minimal meeting experiments (2026-10-30).

Verified inputs only:
  - EfficientNet-B0 raw vectors (Git, SHA-256 checked below)
  - strict manifest + individual-disjoint split
  - shot-scale labels (BioImages + Gemma, provisional)
  - existing K-means cluster IDs for BioCLIP / DINOv3 (K=10/20/50) and EfficientNet (K=20)
BioCLIP / DINOv3 raw vectors are NOT in Git; anything needing them is marked as blocked.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BCV = Path(__file__).resolve().parents[1]
import os
MODEL = os.environ.get("FGV_MODEL", "efficientnet")  # efficientnet | bioclip | dinov3
OUT = Path(__file__).resolve().parent / ("results" if MODEL == "efficientnet" else f"results/{MODEL}")
OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = Path("/home/user/imgcache")  # local cache, rebuilt by fetch_images.sh; images are never committed

# raw vectors per model; BioCLIP/DINOv3 recovered from Drive 2026-10-30 (image_ids/splits verified vs manifest)
EMB = {"efficientnet": ("data/embeddings/efficientnet_b0_strict/embeddings.npz",
                        "b1b6e4f21f8b6ceb3e08f90ac8766e63a9dcce0caa82857068e85dc71679730b"),
       "bioclip": ("data/embeddings/bioclip25_strict/embeddings.npz",
                   "c548c87c868e30cbee85d8b1b33283d4fd787a85aed3a5a3ff9fa823ec721dee"),
       "dinov3": ("data/embeddings/dinov3_strict/embeddings.npz",
                  "63f07c51a39b5b50dd4d23bd97610879a844e4ed0f97111da75d04c523d89cb6")}
EFF_NPZ, EFF_SHA = BCV / EMB[MODEL][0], EMB[MODEL][1]
MANIFEST = BCV / "organ_benchmark/split_manifests/individual_disjoint.csv"
SCALE = BCV / "scale_conditioned_analysis/data/image_visual_groups.csv"
CLUSTERS_BD = BCV / "representation_analysis/data/cluster_assignments.csv"
CLUSTERS_EFF = BCV / "representation_analysis/data/efficientnet_k20_cluster_assignments.csv"
SEED = 42

ORGAN_MAP = {"whole tree (or vine)": "whole tree", "whole tree": "whole tree", "whole plant": "whole tree",
             "stem": "other", "unspecified": "other", "seed": "other"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load():
    """Return (meta DataFrame aligned to rows of X, X float32 L2-normalised vectors of FGV_MODEL)."""
    assert sha256(EFF_NPZ) == EFF_SHA, f"{MODEL} npz SHA-256 mismatch"
    d = np.load(EFF_NPZ, allow_pickle=False)
    ids = d["image_ids"].astype(str)
    X = d["embeddings"].astype(np.float32)
    man = pd.read_csv(MANIFEST).set_index("image_id")
    assert set(ids) == set(man.index) and len(ids) == 1641
    meta = man.loc[ids].reset_index()
    assert (meta["split"].values == d["splits"].astype(str)).all(), "split mismatch npz vs manifest"
    sc = pd.read_csv(SCALE, usecols=["image_id", "shot_scale"]).drop_duplicates("image_id").set_index("image_id")
    meta["shot_scale"] = sc.reindex(meta["image_id"])["shot_scale"].fillna("uncertain").values
    meta["organ"] = meta["organ_category"].map(lambda o: ORGAN_MAP.get(o, o))
    eff = pd.read_csv(CLUSTERS_EFF).set_index("image_id")
    meta["eff_k20"] = eff.loc[meta["image_id"], "cluster_id"].values
    bd = pd.read_csv(CLUSTERS_BD)
    for (rep, k), g in bd.groupby(["representation", "k"]):
        meta[f"{rep}_k{k}"] = g.set_index("image_id").loc[meta["image_id"], "cluster_id"].values
    meta["own_k20"] = meta[{"efficientnet": "eff_k20", "bioclip": "bioclip_k20", "dinov3": "dinov3_k20"}[MODEL]]
    meta["thumb"] = meta["image_id"].str.replace("/", "_") + ".jpg"
    return meta, X


def input_record():
    return {"model": MODEL, "embeddings_npz": str(EFF_NPZ.relative_to(BCV)), "sha256": EFF_SHA,
            "manifest": str(MANIFEST.relative_to(BCV)), "scale_labels": str(SCALE.relative_to(BCV)),
            "clusters": [str(CLUSTERS_BD.relative_to(BCV)), str(CLUSTERS_EFF.relative_to(BCV))]}


def save_json(name: str, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def knn_sets(Z: np.ndarray, k: int = 15) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    return nn.kneighbors(Z, return_distance=False)[:, 1:]


def knn_overlap(A: np.ndarray, B: np.ndarray, k: int = 15) -> float:
    a, b = knn_sets(A, k), knn_sets(B, k)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))
