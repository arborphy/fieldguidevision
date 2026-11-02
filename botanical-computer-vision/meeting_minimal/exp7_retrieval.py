"""Exp 7: retrieval inversion (no generative model). For latent points we return the nearest REAL images
(cosine, full embedding space of FGV_MODEL, all 1,641 images):
  - each cluster centroid of the model's own K-means, for K in K_LIST (examples shown at MAIN_K): top-6 images; record cosine and whether
    the retrieved images belong to that cluster / share its dominant organ;
  - PC1 and PC2 direction points: mean +/- 2 SD along the PC (fit on train), re-normalised.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import K_LIST, MAIN_K, OUT, SEED, clusters, load, save_json, input_record

meta, X = load()
tr = (meta.split == "train").values
points, cl_of = {}, {}
for k in K_LIST:
    lab = clusters(X, k)
    for c in range(k):
        points[f"K{k} centroid c{c}"] = X[lab == c].mean(0); cl_of[f"K{k} centroid c{c}"] = (lab, c)
pca = PCA(5, random_state=SEED).fit(X[tr])
for k in range(2):
    sd = np.sqrt(pca.explained_variance_[k])
    for sgn, nm in [(-2, "-2SD"), (2, "+2SD")]:
        points[f"PC{k+1} {nm}"] = pca.mean_ + sgn * sd * pca.components_[k]
rows, cases = [], {}
for name, v in points.items():
    v = v / np.linalg.norm(v)
    sim = X @ v
    top = np.argsort(sim)[::-1][:6]
    sub = meta.iloc[top]
    r = dict(point=name, top1_cos=float(sim[top[0]]), top6_mean_cos=float(sim[top].mean()),
             top6_organs=", ".join(sub.organ), top6_scales=", ".join(sub.shot_scale), top6_n_species=sub.species.nunique())
    if name in cl_of:
        lab, c = cl_of[name]
        dom = meta[lab == c].organ.value_counts()
        r.update(K=int(name.split()[0][1:]), cluster_size=int(dom.sum()), cluster_dominant_organ=dom.index[0],
                 cluster_dominant_share=float(dom.iloc[0] / dom.sum()), top6_in_cluster=int((lab[top] == c).sum()),
                 top6_share_dominant_organ=float((sub.organ == dom.index[0]).mean()))
    rows.append(r)
    cases[name] = sub[["image_id", "thumb", "species", "organ", "shot_scale"]].assign(cos=sim[top]).to_dict("records")
df = pd.DataFrame(rows)
df.to_csv(OUT / "exp7_retrieval.csv", index=False)
c = df[df.point.str.contains("centroid")]
per_k = c.groupby("K")[["top6_in_cluster", "top6_share_dominant_organ", "cluster_dominant_share"]].mean().reset_index()
save_json("exp7_retrieval_summary.json", dict(inputs=input_record(), per_k=per_k.to_dict("records"), main_k=MAIN_K,
                                              mean_top6_in_cluster=float(c[c.K == MAIN_K].top6_in_cluster.mean()),
                                              mean_top6_share_dominant_organ=float(c[c.K == MAIN_K].top6_share_dominant_organ.mean()),
                                              mean_cluster_dominant_share=float(c[c.K == MAIN_K].cluster_dominant_share.mean()),
                                              cases={n: v for n, v in cases.items() if n.startswith(f"K{MAIN_K} ") or n.startswith("PC")}))
print(per_k.round(3).to_string(index=False))
