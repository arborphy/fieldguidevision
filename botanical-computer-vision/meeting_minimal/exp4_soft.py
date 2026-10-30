"""Exp 4: reuse existing K comparison (K=10/20/50 x 10 seeds) + minimal soft-membership experiment.

Soft model: diagonal GMM, K=20, on PCA-32 of EfficientNet vectors (fit on all 1,641; descriptive).
Per image: max membership, membership entropy. Checks:
  - seed stability of hard GMM labels (5 seeds, pairwise ARI) and of the entropy ranking (Spearman)
  - BIC for K in {5,10,20,30,50}
  - AUROC of entropy for (a) EfficientNet probe error on the 211 test images, (b) shot_scale == mid-range
    (the "in between" scale). Multi-organ AUROC cannot be computed: no multi-organ labels in Git
    (Gemma multi-labels / Human Gold not exported) -> recorded as not done.
Soft probability is NOT multi-organ evidence; it is model uncertainty about which Gaussian.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from common import BCV, OUT, SEED, load, save_json, input_record

meta, X = load()
Z = PCA(32, random_state=SEED).fit_transform(X)

ms = pd.read_csv(BCV / "representation_analysis/data/multiseed_cluster_summary.csv")
ms_keep = ms[["representation", "k", "seeds", "species_nmi_mean", "organ_nmi_mean", "stability_pairwise_ari_mean"]]
ms_keep.to_csv(OUT / "exp4_existing_k_comparison.csv", index=False)

bic = {}
for k in [5, 10, 20, 30, 50]:
    bic[k] = float(GaussianMixture(k, covariance_type="diag", random_state=SEED, n_init=2).fit(Z).bic(Z))
gms = [GaussianMixture(20, covariance_type="diag", random_state=s, n_init=1).fit(Z) for s in range(SEED, SEED + 5)]
P = [g.predict_proba(Z) for g in gms]
H = [-(p * np.log(p + 1e-12)).sum(1) for p in P]
labels = [p.argmax(1) for p in P]
ari = [adjusted_rand_score(labels[i], labels[j]) for i in range(5) for j in range(i + 1, 5)]
hrho = [spearmanr(H[i], H[j])[0] for i in range(5) for j in range(i + 1, 5)]
p, h = P[0], H[0]
meta["gmm_max"] = p.max(1); meta["gmm_entropy"] = h
meta["gmm_top1"] = p.argsort(1)[:, -1]; meta["gmm_top2"] = p.argsort(1)[:, -2]
meta["gmm_p2"] = np.sort(p, 1)[:, -2]
meta["ari_gmm_vs_kmeans20"] = adjusted_rand_score(labels[0], meta.eff_k20)

pred = pd.read_csv(BCV / "data/embeddings/efficientnet_b0_strict/all_predictions.csv").set_index("image_id")
tm = meta[meta.split == "test"].copy()
tm["wrong"] = ~pred.loc[tm.image_id, "correct"].astype(bool).values
auc_err = roc_auc_score(tm.wrong, tm.gmm_entropy)
sm = meta[meta.shot_scale != "uncertain"]
auc_mid = roc_auc_score(sm.shot_scale == "mid-range", sm.gmm_entropy)
frac_ambig = float((meta.gmm_max < 0.6).mean())

# organ composition of each GMM component (for describing the ambiguous cases)
comp_organ = meta.groupby("gmm_top1").organ.agg(lambda s: f"{s.value_counts().index[0]} {s.value_counts(normalize=True).iloc[0]:.0%}")
amb = meta.sort_values("gmm_entropy", ascending=False).head(12).copy()
amb["top1_component_main_organ"] = amb.gmm_top1.map(comp_organ)
amb["top2_component_main_organ"] = amb.gmm_top2.map(comp_organ)
cols = ["image_id", "thumb", "species", "organ", "shot_scale", "gmm_top1", "gmm_max", "gmm_top2", "gmm_p2", "gmm_entropy",
        "top1_component_main_organ", "top2_component_main_organ"]
amb[cols].to_csv(OUT / "exp4_ambiguous_cases.csv", index=False)
meta[["image_id", "gmm_top1", "gmm_max", "gmm_top2", "gmm_p2", "gmm_entropy"]].to_csv(OUT / "exp4_soft_memberships.csv", index=False)
summ = dict(inputs=input_record(), space="PCA-32 (all 1,641, descriptive)", model="GaussianMixture diag K=20",
            bic=bic, seed_label_ari_mean=float(np.mean(ari)), seed_label_ari_min=float(np.min(ari)),
            seed_entropy_spearman_mean=float(np.mean(hrho)), ari_gmm_vs_existing_kmeans20=float(meta.ari_gmm_vs_kmeans20[0]),
            frac_images_max_membership_lt_0_6=frac_ambig, auroc_entropy_vs_probe_error_test211=auc_err,
            n_test_wrong=int(tm.wrong.sum()), auroc_entropy_vs_midrange_scale=auc_mid,
            multi_organ_auroc="NOT COMPUTED - no multi-organ labels in Git (Gemma multi-label / Human Gold not exported)")
save_json("exp4_soft_summary.json", summ)
print(summ)
