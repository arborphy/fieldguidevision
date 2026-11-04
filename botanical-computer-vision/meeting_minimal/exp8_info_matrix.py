"""Exp 8: information-retention matrix - which view of the embedding keeps which information.

For each view (PCA-2/3/16/64, t-SNE-2, UMAP-2/16 if installed; all 1,641 photos, unsupervised):
  - neighbours kept: share of each photo's 15 nearest neighbours in the full space that are still neighbours
  - organ / shot-scale / species information kept: 5-nearest-neighbour classification accuracy in the view,
    5-fold cross-validation grouped by individual plant, divided by the same accuracy in the full space (100% = nothing lost)
Plus which information each principal component carries: eta^2 (share of the PC's variance explained by organ,
shot scale, species) for PC1-10, PCA fitted on all photos.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import warnings; warnings.filterwarnings("ignore")

from common import MODEL, OUT, SEED, load, knn_overlap, save_json, input_record

meta, X = load()
groups = meta.individual_id.values
labels = {"organ": meta.organ.values, "scale": meta.shot_scale.values, "species": meta.species.values}
ok_scale = meta.shot_scale.values != "uncertain"


def knn_acc(Z, y, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    return cross_val_score(KNeighborsClassifier(5), Z[m], y[m], groups=groups[m], cv=GroupKFold(5)).mean()


full = {k: knn_acc(X, y, ok_scale if k == "scale" else None) for k, y in labels.items()}
views = {f"PCA-{d}": PCA(d, random_state=SEED).fit_transform(X) for d in (2, 3, 16, 64)}
views["t-SNE-2"] = TSNE(2, random_state=SEED, init="pca").fit_transform(X)
try:
    import umap
    for d in (2, 16):
        views[f"UMAP-{d}"] = umap.UMAP(n_components=d, random_state=SEED).fit_transform(X)
except ImportError:
    pass
rows = []
for name, Z in views.items():
    r = {"view": name, "neighbours_kept": knn_overlap(X, Z, 15)}
    for k, y in labels.items():
        r[f"{k}_kept"] = knn_acc(Z, y, ok_scale if k == "scale" else None) / full[k]
    rows.append(r)
    print(MODEL, r)
pd.DataFrame(rows).to_csv(OUT / "exp8_info_matrix.csv", index=False)

S = PCA(10, random_state=SEED).fit_transform(X)
eta = []
for j in range(10):
    s = S[:, j]
    r = {"pc": j + 1}
    for k, y in labels.items():
        m = ok_scale if k == "scale" else np.ones(len(s), bool)
        ss, yy = s[m], y[m]
        r[k] = sum((yy == v).sum() * (ss[yy == v].mean() - ss.mean()) ** 2 for v in np.unique(yy)) / ((ss - ss.mean()) ** 2).sum()
    eta.append(r)
pd.DataFrame(eta).to_csv(OUT / "exp8_pc_information.csv", index=False)
save_json("exp8_info_matrix.json", dict(inputs=input_record(), full_space_knn_accuracy=full, rows=rows, pc_eta2=eta))
