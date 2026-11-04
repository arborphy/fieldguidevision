"""Exp 9: choosing K with several clustering metrics, in the 64-D PCA space chosen in section 1.
K-means (n_init=10, seed 42) on PCA-64 of all 1,641 photos (unsupervised). Per K:
  silhouette (higher better), Davies-Bouldin (lower better), Calinski-Harabasz (higher better),
  BIC of a diagonal Gaussian mixture with K components (lower better),
  stability = mean ARI between the reference grouping and 10 refits on resampled individual plants (higher better),
  organ AMI = agreement with organ labels (external check only, not used to choose K).
Also the composition of the K=5 groups."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_mutual_info_score as ami, adjusted_rand_score as ari, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from common import OUT, SEED, describe_clusters, load, save_json, input_record

meta, X = load()
Z = PCA(64, random_state=SEED).fit_transform(X)
KS = [3, 4, 5, 6, 7, 8, 10, 15, 20]
rng = np.random.default_rng(SEED)
rows_by = meta.groupby("individual_id").indices
uind = list(rows_by)
boots = [np.concatenate([rows_by[u] for u in rng.choice(uind, len(uind))]) for _ in range(10)]
rows, labs = [], {}
for k in KS:
    lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    labs[k] = lab
    stab = [ari(lab, KMeans(k, n_init=3, random_state=SEED + b).fit(Z[ix]).predict(Z)) for b, ix in enumerate(boots)]
    rows.append(dict(K=k, silhouette=silhouette_score(Z, lab), davies_bouldin=davies_bouldin_score(Z, lab),
                     calinski_harabasz=calinski_harabasz_score(Z, lab),
                     bic=GaussianMixture(k, covariance_type="diag", random_state=SEED).fit(Z).bic(Z),
                     stability=float(np.mean(stab)), organ_ami=ami(meta.organ, lab)))
    print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv(OUT / "exp9_k_metrics.csv", index=False)
best = {"silhouette": int(df.K[df.silhouette.idxmax()]), "davies_bouldin": int(df.K[df.davies_bouldin.idxmin()]),
        "calinski_harabasz": int(df.K[df.calinski_harabasz.idxmax()]), "bic": int(df.K[df.bic.idxmin()]),
        "stability": int(df.K[df.stability.idxmax()]), "organ_ami": int(df.K[df.organ_ami.idxmax()])}
d5 = describe_clusters(meta, labs[5]).reset_index()
save_json("exp9_k_metrics.json", dict(inputs=input_record(), space="PCA-64", ks=KS, rows=rows, best_k=best,
                                      k5_groups=d5.to_dict("records")))
print(best)
