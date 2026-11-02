"""Exp 2b: composition + enrichment of the model's own K-means clusters for every K in K_LIST, all 1,641 images.

Descriptive only. Images are nested in individuals (465) and species (63), so Fisher/chi-square independence
fails; no p-values. For each cluster x label (organ, shot scale): count, base rate, fold enrichment (FE),
95% CI from an individual-level bootstrap (200 resamples), species-adjusted FE (observed / expected from the
cluster's species mix), n individuals, n species, top species share. Clusters with < 5 individuals are flagged.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import K_LIST, MAIN_K, MODEL, OUT, SEED, clusters, describe_clusters, load, save_json, input_record

meta, X = load()
rng = np.random.default_rng(SEED)
inds = meta.individual_id.values
uniq = np.unique(inds)
rows_by = {u: np.where(inds == u)[0] for u in uniq}
boots = [np.concatenate([rows_by[u] for u in rng.choice(uniq, len(uniq))]) for _ in range(200)]

out = []
for k in K_LIST:
    cl = clusters(X, k)
    for lab in ["organ", "shot_scale"]:
        y = meta[lab].values
        p_sp = pd.crosstab(meta.species, meta[lab], normalize="index")
        for c in range(k):
            inc = cl == c
            sub = meta[inc]
            for v in np.unique(y[inc]):
                base, obs = (y == v).mean(), (y[inc] == v).mean()
                bs = [(y[ix][cl[ix] == c] == v).mean() / (y[ix] == v).mean() for ix in boots if (cl[ix] == c).any()]
                out.append(dict(K=k, label_type=lab, cluster=c, label=v, n_images=int(inc.sum()), count=int((y[inc] == v).sum()),
                                cluster_share=obs, base_rate=base, fold_enrichment=obs / base,
                                fe_ci_lo=np.percentile(bs, 2.5), fe_ci_hi=np.percentile(bs, 97.5),
                                species_adjusted_fe=obs / p_sp.loc[sub.species, v].mean(),
                                n_individuals=sub.individual_id.nunique(), n_species=sub.species.nunique(),
                                top_species_share=sub.species.value_counts(normalize=True).iloc[0],
                                flag_lt5_individuals=sub.individual_id.nunique() < 5))
df = pd.DataFrame(out)
df.to_csv(OUT / "exp2_enrichment_all.csv", index=False)
dom = df[df["count"] >= 5].sort_values("fold_enrichment", ascending=False).groupby(["K", "label_type", "cluster"]).head(1)
dom.to_csv(OUT / "exp2_enrichment_top_per_cluster.csv", index=False)
summ = []
for (k, lt), g in dom.groupby(["K", "label_type"]):
    summ.append(dict(model=MODEL, K=int(k), label=lt, clusters=len(g),
                     share_clusters_dominant_ge_0_6=float((g.cluster_share >= .6).mean()),
                     share_clusters_fe_ci_lo_gt_1_5=float((g.fe_ci_lo > 1.5).mean()),
                     share_clusters_species_adj_fe_gt_1_5=float((g.species_adjusted_fe > 1.5).mean()),
                     median_top_species_share=float(g.top_species_share.median()),
                     clusters_lt5_individuals=int(g.flag_lt5_individuals.sum())))
save_json("exp2_enrichment_summary.json", dict(inputs=input_record(), bootstrap=200, unit="individual plant", summary=summ,
                                               clusters_main_k=describe_clusters(meta, clusters(X, MAIN_K)).reset_index().to_dict("records"),
                                               no_p_values_reason="images nested in individuals/species; independence fails"))
print(pd.DataFrame(summ).round(2).to_string(index=False))

organs = ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark", "cone", "other"]
ct = pd.crosstab(clusters(X, MAIN_K), meta.organ, normalize="index").reindex(columns=organs, fill_value=0)
ct = ct.loc[ct.max(axis=1).sort_values(ascending=False).index]
fig, ax = plt.subplots(figsize=(5, 2.8))
left = np.zeros(len(ct))
for o, col in zip(organs, plt.cm.tab10(np.arange(8))):
    ax.barh(range(len(ct)), ct[o], left=left, color=col, label=o); left += ct[o].values
ax.set_yticks(range(len(ct))); ax.set_yticklabels([f"c{c}" for c in ct.index], fontsize=8); ax.invert_yaxis()
ax.set_title(f"{MODEL} K={MAIN_K}: organ composition per cluster", fontsize=9); ax.legend(fontsize=6, bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.savefig(OUT / f"exp2_organ_composition_k{MAIN_K}.png", dpi=110)
