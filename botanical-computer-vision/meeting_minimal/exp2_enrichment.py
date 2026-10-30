"""Exp 2b: composition + enrichment of EXISTING K=20 clusters (BioCLIP, DINOv3, EfficientNet), all 1,641 images.

Descriptive only. Images are nested in individuals (465) and species (63), so Fisher/chi-square
independence fails; no p-values are produced. For each cluster x label:
  count, base rate, fold enrichment (FE), 95% CI from an individual-level bootstrap (500 resamples),
  species-adjusted FE = observed / expected-given-the-cluster's-species-mix (expected = sum over
  images in cluster of P(label | species) from all 1,641), n individuals, n species, top species share.
Clusters with < 5 individuals are flagged.
"""
import numpy as np
import pandas as pd

from common import OUT, SEED, load, save_json, input_record

meta, _ = load()
rng = np.random.default_rng(SEED)
inds = meta.individual_id.values
uniq_ind = np.unique(inds)
ind_rows = {u: np.where(inds == u)[0] for u in uniq_ind}
B = 500
boot_idx = [np.concatenate([ind_rows[u] for u in rng.choice(uniq_ind, len(uniq_ind))]) for _ in range(B)]

out = []
for rep in ["bioclip_k20", "dinov3_k20", "eff_k20"]:
    cl = meta[rep].values
    for lab in ["organ", "shot_scale"]:
        y = meta[lab].values
        p_sp = pd.crosstab(meta.species, meta[lab], normalize="index")
        for c in np.unique(cl):
            inc = cl == c
            sub = meta[inc]
            for v in np.unique(y):
                base = (y == v).mean()
                obs = (y[inc] == v).mean()
                fe = obs / base
                bs = []
                for ix in boot_idx:
                    ci, yi = cl[ix] == c, y[ix]
                    if ci.sum() and (yi == v).any():
                        bs.append((yi[ci] == v).mean() / (yi == v).mean())
                expv = p_sp.loc[sub.species, v].mean() if v in p_sp else np.nan
                out.append(dict(representation=rep.replace("_k20", ""), label_type=lab, cluster=int(c), label=v,
                                n_images=int(inc.sum()), count=int((y[inc] == v).sum()), cluster_share=obs, base_rate=base,
                                fold_enrichment=fe, fe_ci_lo=np.percentile(bs, 2.5), fe_ci_hi=np.percentile(bs, 97.5),
                                species_adjusted_fe=obs / expv if expv > 0 else np.nan,
                                n_individuals=sub.individual_id.nunique(), n_species=sub.species.nunique(),
                                top_species=sub.species.value_counts().index[0],
                                top_species_share=sub.species.value_counts(normalize=True).iloc[0],
                                flag_lt5_individuals=sub.individual_id.nunique() < 5))
df = pd.DataFrame(out)
df.to_csv(OUT / "exp2_enrichment_all.csv", index=False)

# headline: for each cluster, its dominant label; CI lower bound > 1 is "robust to individual resampling"
dom = df[df["count"] >= 5].sort_values("fold_enrichment", ascending=False).groupby(["representation", "label_type", "cluster"]).head(1)
dom.to_csv(OUT / "exp2_enrichment_top_per_cluster.csv", index=False)
summ = {}
for (rep, lt), g in dom.groupby(["representation", "label_type"]):
    summ[f"{rep}/{lt}"] = dict(
        clusters=len(g),
        clusters_dominant_share_ge_0_6=int((g.cluster_share >= .6).sum()),
        clusters_fe_ci_lo_gt_1_5=int((g.fe_ci_lo > 1.5).sum()),
        clusters_species_adjusted_fe_gt_1_5=int((g.species_adjusted_fe > 1.5).sum()),
        median_top_species_share=float(g.top_species_share.median()),
        clusters_lt5_individuals=int(g.flag_lt5_individuals.sum()))
save_json("exp2_enrichment_summary.json", dict(inputs=input_record(), bootstrap=B, unit="individual plant",
                                               no_p_values_reason="images nested in individuals/species; independence assumption fails",
                                               summary=summ))
print(pd.DataFrame(summ).T)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
organs = ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark", "cone", "other"]
cols = plt.cm.tab10(np.arange(8))
for ax, rep in zip(axes, ["bioclip_k20", "dinov3_k20", "eff_k20"]):
    ct = pd.crosstab(meta[rep], meta.organ, normalize="index").reindex(columns=organs, fill_value=0)
    ct = ct.loc[ct.max(1).sort_values(ascending=False).index]
    left = np.zeros(len(ct))
    for o, col in zip(organs, cols):
        ax.barh(range(len(ct)), ct[o], left=left, color=col, label=o); left += ct[o].values
    ax.set_yticks(range(len(ct))); ax.set_yticklabels([f"c{c}" for c in ct.index], fontsize=7)
    ax.set_title(rep.replace("_k20", " K=20") + " — organ composition", fontsize=9); ax.invert_yaxis()
axes[-1].legend(fontsize=7, bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.savefig(OUT / "exp2_organ_composition.png", dpi=110)
