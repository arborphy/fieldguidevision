"""Exp 4: soft cluster membership by bootstrap consensus, for K in K_LIST (3 / 7 / 20 / 63).

Why not the first run: a diagonal GMM's memberships get more extreme as dimension grows (per-dimension likelihoods
multiply), so "1.3% ambiguous in 32-D" was a dimension artefact (22% in 2-D); K=20 was fixed without reason.
That diagnosis is saved below (gmm_dimension_artifact).

Method: reference K-means = common.clusters(X, K). B=50 runs: resample INDIVIDUAL plants with replacement, refit,
assign all images, align run clusters to the reference (Hungarian on the contingency table). membership[i, c] =
share of runs where image i lands in reference cluster c. Ambiguous = max membership < 0.8.

Clusters are described first (common.describe_clusters: organ and scale composition, honest names), then each
ambiguous image is put in one category, using the image's OWN organ label (BioImages organ_category):
  same_organ   both candidate clusters have the same main organ -> ambiguity inside one organ (scale/species/background)
  both_plausible  image's organ = main organ of the top cluster, 2nd cluster is another organ -> second organ may be visible
  leaning_wrong   image's organ = main organ of the 2nd cluster only -> model leans to the other organ group
  neither         neither candidate's main organ is the image's organ -> unusual photo or label to check
  mixed_group     a candidate cluster has no main organ (<50%) -> cannot be read as an organ question
Also: entropy reliability (two halves of the runs), entropy vs geometric boundary (d1/d2), AUROC of entropy for
species-probe errors on test and for mid-range scale. Multi-organ truth still needs multi-organ labels (not in Git).
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

from common import K_LIST, MAIN_K, OUT, SEED, clusters, describe_clusters, load, save_json, input_record

meta, X = load()
rng = np.random.default_rng(SEED)
B = 50
CATS = ["same_organ", "both_plausible", "leaning_wrong", "neither", "mixed_group"]

artifact = []
for d in [2, 4, 8, 16, 32, 64]:
    Z = PCA(d, random_state=SEED).fit_transform(X)
    p = GaussianMixture(20, covariance_type="diag", random_state=SEED).fit(Z).predict_proba(Z)
    artifact.append(dict(pca_dim=d, frac_max_lt_0_6=float((p.max(1) < .6).mean()), median_max=float(np.median(p.max(1)))))

tr, va, te = ((meta.split == s).values for s in ("train", "validation", "test"))
best = max([1.0, 10.0, 100.0], key=lambda c: f1_score(meta.species[va], LogisticRegression(C=c, max_iter=3000).fit(
    X[tr], meta.species[tr]).predict(X[va]), average="macro"))
wrong = LogisticRegression(C=best, max_iter=3000).fit(X[tr], meta.species[tr]).predict(X[te]) != meta.species[te].values

uind = meta.individual_id.unique()
rows_by_ind = meta.groupby("individual_id").indices
boots = [np.concatenate([rows_by_ind[u] for u in rng.choice(uind, len(uind))]) for _ in range(B)]
sc_ok = (meta.shot_scale != "uncertain").values
ent = lambda Q: -(Q * np.log(Q + 1e-12)).sum(1)
res, detail, allm = {}, {}, []
for k in K_LIST:
    ref = clusters(X, k)
    M = np.zeros((B, len(X), k))
    for b, ix in enumerate(boots):
        lab = KMeans(k, n_init=1, random_state=SEED + b).fit(X[ix]).predict(X)
        ct = pd.crosstab(lab, ref).reindex(index=range(k), columns=range(k), fill_value=0).values
        r, c = linear_sum_assignment(-ct)
        M[b, np.arange(len(X)), c[np.argsort(r)][lab]] = 1
    P = M.mean(0)
    H = ent(P) / np.log(k)
    cent = np.stack([X[ref == c].mean(0) for c in range(k)])
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    D = np.sort(1 - X @ cent.T, 1)
    top = P.argsort(1)
    desc = describe_clusters(meta, ref)
    org = desc.top_organ.where(desc.top_organ_share >= .5, None)
    a, b2 = top[:, -1], top[:, -2]
    oa, ob = org.reindex(a).values, org.reindex(b2).values
    y = meta.organ.values
    cat = np.select([pd.isna(oa) | pd.isna(ob), oa == ob, (y == oa), (y == ob)],
                    ["mixed_group", "same_organ", "both_plausible", "leaning_wrong"], "neither")
    amb = P.max(1) < 0.8
    counts = pd.Series(cat[amb]).value_counts().reindex(CATS, fill_value=0)
    res[f"K{k}"] = dict(
        frac_max_lt_0_8=float(amb.mean()), frac_max_lt_0_6=float((P.max(1) < .6).mean()),
        entropy_split_half_spearman=float(spearmanr(ent(M[:B // 2].mean(0)), ent(M[B // 2:].mean(0)))[0]),
        entropy_vs_d1_d2_spearman=float(spearmanr(H, D[:, 0] / D[:, 1])[0]),
        auroc_entropy_species_probe_error_test=float(roc_auc_score(wrong, H[te])), probe_C=best,
        auroc_entropy_midrange=float(roc_auc_score(meta.shot_scale[sc_ok] == "mid-range", H[sc_ok])),
        n_ambiguous=int(amb.sum()), category_counts=counts.to_dict(),
        category_share=(counts / max(amb.sum(), 1)).round(3).to_dict())
    if k == MAIN_K:  # examples: cluster faces, per-category cases, most common cluster pairs
        rec = lambda i: dict(image_id=meta.image_id[i], species=meta.species[i], organ=meta.organ[i], shot_scale=meta.shot_scale[i],
                             c1=int(a[i]), p1=float(P[i, a[i]]), c2=int(b2[i]), p2=float(P[i, b2[i]]), category=str(cat[i]))
        faces = {int(c): [rec(i) for i in np.argsort(1 - X @ cent[c])[:4]] for c in range(k)}
        order = np.argsort(-H)
        by_cat = {ct_: [rec(i) for i in order if amb[i] and cat[i] == ct_][:8] for ct_ in CATS}
        pairs = pd.Series([tuple(sorted((int(a[i]), int(b2[i])))) for i in np.where(amb)[0]]).value_counts().head(5)
        by_pair = [dict(pair=list(p_), n=int(n_), cases=[rec(i) for i in order if amb[i] and tuple(sorted((a[i], b2[i]))) == p_][:8])
                   for p_, n_ in pairs.items()]
        detail = dict(K=k, clusters=desc.reset_index().to_dict("records"), faces=faces, by_category=by_cat, by_pair=by_pair)
    allm.append(pd.DataFrame({"image_id": meta.image_id, "K": k, "max_membership": P.max(1), "norm_entropy": H,
                              "top1": a, "top2": b2, "category_if_ambiguous": np.where(amb, cat, "")}))
    print(k, res[f"K{k}"]["frac_max_lt_0_8"], res[f"K{k}"]["category_share"])
pd.concat(allm).to_csv(OUT / "exp4_soft_memberships.csv", index=False)
save_json("exp4_soft_summary.json", dict(inputs=input_record(), method=f"bootstrap consensus, B={B}, resampling individuals",
                                         gmm_dimension_artifact=artifact, results=res, main=detail,
                                         multi_organ="NOT TESTED - no multi-organ labels in Git"))
