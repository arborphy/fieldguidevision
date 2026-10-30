"""Exp 7: retrieval inversion (no generative model). For latent points we return the nearest REAL images
(cosine, full 1,280-d EfficientNet space, all 1,641 images):
  - each K=20 cluster centroid (existing EfficientNet K-means): top-6 images; record cosine and whether
    the retrieved images belong to that cluster / share its dominant organ;
  - PC1 and PC2 direction points: mean +/- 2 SD along the PC (fit on train), re-normalised.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import OUT, SEED, load, save_json, input_record

meta, X = load()
tr = (meta.split == "train").values
cent = np.stack([X[meta.eff_k20 == c].mean(0) for c in range(20)])
points = {f"centroid c{c}": cent[c] for c in range(20)}
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
    if name.startswith("centroid"):
        c = int(name.split("c")[-1])
        dom = meta[meta.eff_k20 == c].organ.value_counts()
        r.update(cluster_size=int(dom.sum()), cluster_dominant_organ=dom.index[0], cluster_dominant_share=float(dom.iloc[0] / dom.sum()),
                 top6_in_cluster=int((sub.eff_k20 == c).sum()), top6_share_dominant_organ=float((sub.organ == dom.index[0]).mean()))
    rows.append(r)
    cases[name] = sub[["image_id", "thumb", "species", "organ", "shot_scale"]].assign(cos=sim[top]).to_dict("records")
df = pd.DataFrame(rows)
df.to_csv(OUT / "exp7_retrieval.csv", index=False)
c = df[df.point.str.startswith("centroid")]
save_json("exp7_retrieval_summary.json", dict(inputs=input_record(), mean_top6_in_cluster=float(c.top6_in_cluster.mean()),
                                              mean_top6_share_dominant_organ=float(c.top6_share_dominant_organ.mean()),
                                              mean_cluster_dominant_share=float(c.cluster_dominant_share.mean()),
                                              cases=cases))
print(df.round(3).to_string())
