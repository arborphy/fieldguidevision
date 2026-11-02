"""Exp 4 (redo): soft cluster membership that is measured, not produced by the model's assumptions.

What was wrong with the first run (diagonal GMM, K=20, PCA-32):
  posterior sharpness of a diagonal GMM grows with dimension (per-dimension likelihoods multiply), so
  "only 1.3% ambiguous images" was a property of choosing 32-D, not of the photos; K=20 was fixed without reason;
  the error check existed for EfficientNet only. The diagnosis is re-run and saved below (gmm_dimension_artifact).

New method: bootstrap consensus membership.
  Reference K-means on the full vectors (all 1,641). B=50 runs: resample INDIVIDUAL plants with replacement,
  refit K-means, assign all images, align the run's clusters to the reference (Hungarian on the contingency table).
  membership[i, c] = share of runs in which image i lands in reference cluster c. No temperature, no Gaussian
  assumption, same in any dimension. Entropy is normalised by log K (0 = always the same cluster, 1 = uniform).
Checks (K = 7 organ count, K = 20 previous default):
  - reliability: Spearman of entropy between two halves of the runs (25 vs 25)
  - geometry: Spearman of entropy with d1/d2 (distance to nearest / second-nearest centroid)
  - AUROC of entropy for: species-probe error on test (probe per model, C chosen on validation), mid-range scale
  - for ambiguous images (max membership < 0.8): do the two candidate clusters have different dominant organs?
Multi-organ evidence still needs multi-organ labels (not in Git): not tested.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture
import warnings; warnings.filterwarnings("ignore")

from common import MODEL, OUT, SEED, load, save_json, input_record

meta, X = load()
rng = np.random.default_rng(SEED)
B = 50

# diagnosis of the first run
artifact = []
for d in [2, 4, 8, 16, 32, 64]:
    p = GaussianMixture(20, covariance_type="diag", random_state=SEED).fit(PCA(d, random_state=SEED).fit_transform(X)).predict_proba(
        PCA(d, random_state=SEED).fit_transform(X))
    artifact.append(dict(pca_dim=d, frac_max_lt_0_6=float((p.max(1) < .6).mean()), median_max=float(np.median(p.max(1)))))

# species probe errors on test (validation-selected C)
tr, va, te = (meta.split == s for s in ("train", "validation", "test"))
best = max([1.0, 10.0, 100.0], key=lambda c: f1_score(meta.species[va], LogisticRegression(C=c, max_iter=3000).fit(
    X[tr], meta.species[tr]).predict(X[va]), average="macro"))
wrong = LogisticRegression(C=best, max_iter=3000).fit(X[tr], meta.species[tr]).predict(X[te]) != meta.species[te].values

uind = meta.individual_id.unique()
rows_by_ind = meta.groupby("individual_id").indices
boots = [np.concatenate([rows_by_ind[u] for u in rng.choice(uind, len(uind))]) for _ in range(B)]
sc_ok = (meta.shot_scale != "uncertain").values
res, cases, allm = {}, {}, []
for k in (7, 20):
    ref = KMeans(k, n_init=10, random_state=SEED).fit(X)
    M = np.zeros((B, len(X), k))
    for b, ix in enumerate(boots):
        lab = KMeans(k, n_init=1, random_state=SEED + b).fit(X[ix]).predict(X)
        ct = pd.crosstab(lab, ref.labels_).reindex(index=range(k), columns=range(k), fill_value=0).values
        r, c = linear_sum_assignment(-ct)
        M[b, np.arange(len(X)), c[np.argsort(r)][lab]] = 1
    P = M.mean(0)
    H = -(P * np.log(P + 1e-12)).sum(1) / np.log(k)
    H1 = (lambda Q: -(Q * np.log(Q + 1e-12)).sum(1))(M[:B // 2].mean(0))
    H2 = (lambda Q: -(Q * np.log(Q + 1e-12)).sum(1))(M[B // 2:].mean(0))
    D = np.sort(1 - X @ (ref.cluster_centers_ / np.linalg.norm(ref.cluster_centers_, axis=1, keepdims=True)).T, 1)
    top = P.argsort(1)
    dom = meta.groupby(ref.labels_).organ.agg(lambda s: s.value_counts().index[0])
    amb = P.max(1) < 0.8
    diff_organ = dom.reindex(top[:, -1]).values != dom.reindex(top[:, -2]).values
    res[f"K{k}"] = dict(
        frac_max_lt_0_8=float(amb.mean()), frac_max_lt_0_6=float((P.max(1) < .6).mean()),
        mean_norm_entropy=float(H.mean()), entropy_split_half_spearman=float(spearmanr(H1, H2)[0]),
        entropy_vs_d1_d2_spearman=float(spearmanr(H, D[:, 0] / D[:, 1])[0]),
        auroc_entropy_species_probe_error_test=float(roc_auc_score(wrong, H[te.values])), probe_C=best,
        test_error_rate=float(wrong.mean()),
        auroc_entropy_midrange=float(roc_auc_score(meta.shot_scale[sc_ok] == "mid-range", H[sc_ok])),
        ambiguous_between_different_organ_clusters=float(diff_organ[amb].mean()) if amb.any() else None,
        ambiguous_entropy_by_organ=meta.assign(H=H).groupby("organ").H.mean().round(3).to_dict(),
        ambiguous_entropy_by_scale=meta.assign(H=H).groupby("shot_scale").H.mean().round(3).to_dict())
    o = np.argsort(-H)[:8]
    cases[f"K{k}"] = [dict(image_id=meta.image_id[i], species=meta.species[i], organ=meta.organ[i], shot_scale=meta.shot_scale[i],
                           c1=int(top[i, -1]), p1=float(P[i, top[i, -1]]), c1_organ=dom[top[i, -1]],
                           c2=int(top[i, -2]), p2=float(P[i, top[i, -2]]), c2_organ=dom[top[i, -2]], entropy=float(H[i])) for i in o]
    allm.append(pd.DataFrame({"image_id": meta.image_id, "K": k, "max_membership": P.max(1), "norm_entropy": H,
                              "top1": top[:, -1], "top2": top[:, -2]}))
    print(k, res[f"K{k}"])
pd.concat(allm).to_csv(OUT / "exp4_soft_memberships.csv", index=False)
save_json("exp4_soft_summary.json", dict(inputs=input_record(), method=f"bootstrap consensus, B={B}, resampling individuals",
                                         gmm_dimension_artifact=artifact, results=res, cases=cases,
                                         multi_organ="NOT TESTED - no multi-organ labels in Git"))
