"""Exp 4b: K sweep instead of a fixed K=20. Each K has a reason:
  3 = number of shot scales, 7 = number of organ labels, 63 = number of species,
  10/15/20/30/50 = the range used so far, 100 = finer than species.
K-means on the full L2-normalised vectors (all 1,641; descriptive). Per K:
  - AMI (chance-adjusted mutual information) and ARI with organ (7), shot scale (3, 'uncertain' excluded), species (63)
  - matched accuracy: one-to-one Hungarian match of clusters to labels (only meaningful when K = number of labels)
  - cosine silhouette
  - stability: 10 bootstrap resamples of INDIVIDUAL plants, refit, predict all images, ARI vs the full fit
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari, silhouette_score

from common import MODEL, OUT, SEED, load, save_json, input_record

meta, X = load()
KS = [3, 5, 7, 10, 15, 20, 30, 50, 63, 100]
rng = np.random.default_rng(SEED)
uind = meta.individual_id.unique()
rows_by_ind = meta.groupby("individual_id").indices
boots = [np.concatenate([rows_by_ind[u] for u in rng.choice(uind, len(uind))]) for _ in range(10)]
sc_ok = (meta.shot_scale != "uncertain").values


def matched_acc(lab, y):
    ct = pd.crosstab(lab, y).values
    r, c = linear_sum_assignment(-ct)
    return ct[r, c].sum() / len(y)


rows, comp = [], {}
for k in KS:
    km = KMeans(k, n_init=10, random_state=SEED).fit(X)
    lab = km.labels_
    stab = [ari(lab, KMeans(k, n_init=3, random_state=SEED + b).fit(X[ix]).predict(X)) for b, ix in enumerate(boots)]
    rows.append(dict(model=MODEL, K=k,
                     ami_organ=ami(meta.organ, lab), ari_organ=ari(meta.organ, lab),
                     ami_scale=ami(meta.shot_scale[sc_ok], lab[sc_ok]), ari_scale=ari(meta.shot_scale[sc_ok], lab[sc_ok]),
                     ami_species=ami(meta.species, lab), ari_species=ari(meta.species, lab),
                     matched_acc_organ=matched_acc(lab, meta.organ) if k == 7 else np.nan,
                     matched_acc_scale=matched_acc(lab[sc_ok], meta.shot_scale[sc_ok]) if k == 3 else np.nan,
                     matched_acc_species=matched_acc(lab, meta.species) if k == 63 else np.nan,
                     silhouette=silhouette_score(X, lab, metric="cosine"),
                     stability_ari_mean=float(np.mean(stab)), stability_ari_sd=float(np.std(stab))))
    if k in (3, 7):  # composition tables for the two "label-count" Ks
        comp[k] = pd.crosstab(lab, meta.shot_scale if k == 3 else meta.organ).to_dict("index")
    print(rows[-1])
df = pd.DataFrame(rows)
df.to_csv(OUT / "exp4b_k_sweep.csv", index=False)
save_json("exp4b_k_sweep.json", dict(inputs=input_record(), ks=KS, rows=rows, composition={str(k): v for k, v in comp.items()},
                                     note="AMI corrects for chance; plain NMI rises with K by itself. Stability resamples individuals."))
