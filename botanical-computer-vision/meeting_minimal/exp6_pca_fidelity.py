"""Exp 6 (+ PCA part of exp 2): how much does PCA-d keep? EfficientNet-B0 raw vectors.
PCA fit on train (1,181); everything reported on validation (249) unless noted.
Metrics per d: explained variance (train), val reconstruction R^2, 15-NN overlap vs full space (val pool),
trustworthiness (val), linear-probe species/organ macro-F1 on val, K-means(K=20) ARI vs full-space K-means (val).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.metrics import adjusted_rand_score, f1_score
import warnings; warnings.filterwarnings("ignore")

from common import OUT, SEED, load, knn_overlap, save_json, input_record

meta, X = load()
tr, va = (meta.split == "train").values, (meta.split == "validation").values
Xtr, Xva = X[tr], X[va]
pca_full = PCA(n_components=256, random_state=SEED).fit(Xtr)


def probe(Ztr, Zva, y):
    # C=100 = validation-selected C of the existing EfficientNet benchmark
    clf = LogisticRegression(max_iter=3000, C=100.0).fit(Ztr, y[tr])
    return f1_score(y[va], clf.predict(Zva), average="macro")


km_full = KMeans(20, n_init=10, random_state=SEED).fit(Xtr).predict(Xva)
rows = []
for d in [2, 8, 16, 32, 64, 128, 256, "full"]:
    if d == "full":
        Ztr, Zva = Xtr, Xva; ev = 1.0; r2 = 1.0; ov = 1.0; tw = 1.0
    else:
        Ztr, Zva = pca_full.transform(Xtr)[:, :d], pca_full.transform(Xva)[:, :d]
        ev = float(pca_full.explained_variance_ratio_[:d].sum())
        rec = Zva @ pca_full.components_[:d] + pca_full.mean_
        r2 = 1 - ((Xva - rec) ** 2).sum() / ((Xva - Xva.mean(0)) ** 2).sum()
        ov = knn_overlap(Xva, Zva, 15)
        tw = trustworthiness(Xva, Zva, n_neighbors=15)
    km = KMeans(20, n_init=10, random_state=SEED).fit(Ztr).predict(Zva)
    rows.append(dict(d=d, train_explained_var=ev, val_recon_r2=float(r2), val_knn15_overlap=ov, val_trustworthiness=float(tw),
                     probe_species_macroF1=probe(Ztr, Zva, meta.species.values),
                     probe_organ_macroF1=probe(Ztr, Zva, meta.organ.values),
                     kmeans20_ARI_vs_full=adjusted_rand_score(km_full, km)))
    print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv(OUT / "exp6_pca_fidelity.csv", index=False)

full = df[df.d == "full"].iloc[0]
ok = df[(df.d != "full") & (df.val_knn15_overlap >= 0.85) & (df.probe_species_macroF1 >= full.probe_species_macroF1 - 0.01)]
dstar = int(ok.d.iloc[0]) if len(ok) else None
# descriptive on all 1,641 for 2-D (comparable to the existing site number for BioCLIP)
Z2 = PCA(2, random_state=SEED).fit_transform(X)
all2 = dict(knn15_overlap_all1641=knn_overlap(X, Z2, 15), trustworthiness_all1641=float(trustworthiness(X, Z2, n_neighbors=15)))
save_json("exp6_summary.json", dict(inputs=input_record(), d_star_rule="smallest d with val 15-NN>=0.85 and species probe within 1pp of full",
                                    d_star=dstar, two_d_all_images=all2, table=rows))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
dd = df[df.d != "full"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
x = dd.d.astype(int)
for col, lab in [("train_explained_var", "explained variance"), ("val_knn15_overlap", "15-NN overlap (val)"),
                 ("val_trustworthiness", "trustworthiness (val)"), ("probe_organ_macroF1", "organ probe F1"),
                 ("probe_species_macroF1", "species probe F1"), ("kmeans20_ARI_vs_full", "K=20 ARI vs full")]:
    ax.plot(x, dd[col], marker="o", label=lab)
ax.set_xscale("log", base=2); ax.set_xlabel("PCA dimensions d (fit on train)"); ax.set_ylim(0, 1.02)
ax.axhline(0.85, color="grey", ls=":", lw=1); ax.legend(fontsize=7, loc="lower right"); ax.set_title("EfficientNet-B0: what PCA-d keeps")
plt.tight_layout(); plt.savefig(OUT / "exp6_pca_fidelity.png", dpi=120)
