"""Exp 8c: the same 1,641 EfficientNet embeddings drawn three ways (PCA-2, t-SNE, UMAP), coloured by organ.
Panel titles give the organ information kept (from exp8_info_matrix.csv). Plot only; same seeds as exp8."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import OUT, SEED, load

meta, X = load()
kept = pd.read_csv(OUT / "exp8_info_matrix.csv").set_index("view").organ_kept
maps = {"PCA-2": PCA(2, random_state=SEED).fit_transform(X),
        "t-SNE-2": TSNE(2, random_state=SEED, init="pca").fit_transform(X),
        "UMAP-2": umap.UMAP(n_components=2, random_state=SEED).fit_transform(X)}
organs = ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark", "cone"]
cols = ["#2e7d32", "#5d4037", "#ad1457", "#ef6c00", "#6d4c41", "#455a64", "#1565c0"]
cols = ["#4c9a2a", "#8c6d31", "#c2185b", "#f57c00", "#7b5ea7", "#607d8b", "#1e88e5"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
for ax, (name, Z) in zip(axes, maps.items()):
    ax.scatter(Z[meta.organ == "other", 0], Z[meta.organ == "other", 1], s=4, c="#cccccc", lw=0)
    for o, c in zip(organs, cols):
        m = (meta.organ == o).values
        ax.scatter(Z[m, 0], Z[m, 1], s=5, c=c, lw=0, alpha=.8, label=o)
    ax.set_title(f"{name.replace('-2', ' (2-D)')}: organ information kept {kept[name]:.0%}" if name in kept
                 else name, fontsize=10, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#dddddd")
axes[-1].legend(markerscale=3, fontsize=8, frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout(); plt.savefig(OUT / "exp8_maps_effnet.png", dpi=140)
print("ok")
