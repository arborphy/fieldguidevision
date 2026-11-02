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
                  "63f07c51a39b5b50dd4d23bd97610879a844e4ed0f97111da75d04c523d89cb6"),
       # YOLO11s (COCO detection, frozen), team extraction 2026-11-02, two separate representations (see its README)
       "yolo11s_backbone": ("data/embeddings/yolo11s_strict/yolo11s_backbone_embeddings.npz",
                            "ade72e94f74ba81ec7145beb96a6e8d1f4a5e0cf1e6ee35892ad6a62609df353"),
       "yolo11s_neck": ("data/embeddings/yolo11s_strict/yolo11s_neck_multiscale_embeddings.npz",
                        "c96c5c0b37803bca759ea403e8ca4d38b3cf6192d2ddf256737e97bb88dcbf15")}
# any other model (e.g. FGV_MODEL=yolo): data/embeddings/<model>_strict/embeddings.npz or FGV_EMB_PATH;
# same npz layout (image_ids, embeddings, splits); SHA-256 is recorded, not checked, until it is pinned in EMB.
EFF_NPZ = Path(os.environ["FGV_EMB_PATH"]) if "FGV_EMB_PATH" in os.environ else BCV / EMB.get(
    MODEL, (f"data/embeddings/{MODEL}_strict/embeddings.npz", None))[0]
EFF_SHA = EMB[MODEL][1] if MODEL in EMB else None
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
    if EFF_SHA:
        assert sha256(EFF_NPZ) == EFF_SHA, f"{MODEL} npz SHA-256 mismatch"
    d = np.load(EFF_NPZ, allow_pickle=False)
    ids = d["image_ids"].astype(str)
    X = d["embeddings"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)  # no-op for the stored (already normalised) vectors
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
    existing = {"efficientnet": "eff_k20", "bioclip": "bioclip_k20", "dinov3": "dinov3_k20"}
    if MODEL in existing:
        meta["own_k20"] = meta[existing[MODEL]]
    else:  # no stored clustering for a new model: same recipe (K-means K=20, all images, seed 42)
        from sklearn.cluster import KMeans
        meta["own_k20"] = KMeans(20, n_init=10, random_state=SEED).fit_predict(X)
    meta["thumb"] = meta["image_id"].str.replace("/", "_") + ".jpg"
    return meta, X


def input_record():
    return {"model": MODEL, "embeddings_npz": str(EFF_NPZ), "sha256": EFF_SHA or sha256(EFF_NPZ),
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


# K is a choice, not a given: every clustering experiment runs on these values (why: 3 shot scales, 7 organs,
# 20 = the earlier default; large K left out to keep runs short). Figures and examples use MAIN_K = 7 (organ question).
K_LIST = [3, 7, 20]
MAIN_K = 7


def clusters(X, k):
    """K-means labels of FGV_MODEL at K=k (full L2-normalised vectors, all 1,641, n_init=10, seed 42), cached."""
    p = OUT / f"clusters_k{k}.npy"
    if p.exists():
        return np.load(p)
    from sklearn.cluster import KMeans
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(X)
    np.save(p, lab)
    return lab


def describe_clusters(meta, lab):
    """One row per cluster: size, organ / scale composition, n species, and an honest name.
    Organ part: >=80% -> 'leaf'; 50-80% -> 'mostly leaf 64% (+fruit 12%)'; <50% -> 'mixed leaf 45% / twig 30%'.
    Scale part: >=80% -> 'close-up', else 'mixed scale'. Names that still collide get the cluster number."""
    rows = []
    for c in np.unique(lab):
        m = meta[lab == c]
        o = m.organ.value_counts(normalize=True)
        s = m.shot_scale.value_counts(normalize=True)
        o2 = f"{o.index[1]} {o.iloc[1]:.0%}" if len(o) > 1 else ""
        oname = (o.index[0] if o.iloc[0] >= .8 else f"mostly {o.index[0]} {o.iloc[0]:.0%} (+{o2})" if o.iloc[0] >= .5
                 else f"mixed {o.index[0]} {o.iloc[0]:.0%} / {o2}")
        sname = s.index[0] if s.iloc[0] >= .8 else "mixed scale"
        rows.append(dict(cluster=int(c), n=len(m), name=f"{oname} · {sname}", top_organ=o.index[0],
                         top_organ_share=float(o.iloc[0]), organs=", ".join(f"{k} {v:.0%}" for k, v in o.head(3).items()),
                         scales=", ".join(f"{k} {v:.0%}" for k, v in s.head(3).items()),
                         n_species=m.species.nunique(), top_species_share=float(m.species.value_counts(normalize=True).iloc[0])))
    d = pd.DataFrame(rows).set_index("cluster")
    dup = d.name.duplicated(keep=False)
    d.loc[dup, "name"] = d.name[dup] + " #" + d.index[dup].astype(str)
    return d
