"""Exp 10: what drives each K = 5 group, from the enrichment results (exp2 run with FGV_KLIST=5, FGV_OUTSUB=k5).
Rule, applied in this order to each group:
  species-driven  : one species makes up more than 30% of the group
  organ-driven    : one organ is at least 2x its overall share (fold enrichment >= 2), also after adjusting for the
                    group's species mix (species-adjusted enrichment >= 2)
  distance-driven : otherwise, the distant or mid-range share is at least 2x its overall share (close-up is the
                    majority of all photos, so a close-up majority alone says nothing)
  mixed           : none of the above
Output: share of photos in each type per model (exp10_drivers.json) and a stacked bar chart (exp10_drivers.png)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
MODELS = {"EfficientNet-B0": R / "k5", "DINOv3": R / "dinov3" / "k5", "YOLO11s backbone": R / "yolo11s_backbone" / "k5",
          "YOLO11s neck": R / "yolo11s_neck" / "k5", "BioCLIP 2.5": R / "bioclip" / "k5"}
TYPES = [("organ", "#dd8452"), ("distance", "#4c72b0"), ("species", "#55a868"), ("mixed", "#bdbdbd")]
out = {}
for m, d in MODELS.items():
    e = pd.read_csv(d / "exp2_enrichment_all.csv")
    e = e[e.K == 5]
    groups = []
    for c, g in e.groupby("cluster"):
        org = g[(g.label_type == "organ") & (g.label != "other")]
        sc = g[(g.label_type == "shot_scale") & (g.label != "uncertain")]
        top_sp = float(g.top_species_share.iloc[0])
        best = org.loc[org.fold_enrichment.idxmax()]
        if top_sp > 0.30:
            t = "species"
        elif best.fold_enrichment >= 2 and best.species_adjusted_fe >= 2:
            t = "organ"
        elif sc[sc.label.isin(["distant", "mid-range"])].fold_enrichment.max() >= 2:
            t = "distance"
        else:
            t = "mixed"
        groups.append(dict(cluster=int(c), n=int(g.n_images.iloc[0]), type=t, organ=best.label, fe=float(best.fold_enrichment),
                           fe_species_adj=float(best.species_adjusted_fe), organ_share=float(best.cluster_share), top_species_share=top_sp))
    gdf = pd.DataFrame(groups)
    share = {t: float(gdf[gdf.type == t].n.sum() / gdf.n.sum()) for t, _ in TYPES}
    out[m] = dict(groups=groups, share=share)
    print(m, {k: round(v, 2) for k, v in share.items()}, [(r["organ"], round(r["fe"], 1), r["type"]) for r in groups])
(R / "exp10_drivers.json").write_text(json.dumps(out, indent=2))

fig, ax = plt.subplots(figsize=(5.4, 2.1))
names = list(MODELS)
left = np.zeros(len(names))
for t, c in TYPES:
    v = np.array([out[m]["share"][t] for m in names])
    ax.barh(range(len(names)), v, left=left, color=c, height=0.62, label={"organ": "organ-driven", "distance": "distance-driven",
                                                                           "species": "species-driven", "mixed": "mixed"}[t])
    for i, (l, w) in enumerate(zip(left, v)):
        if w >= 0.08:
            ax.text(l + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=7.5, color="white" if t != "mixed" else "#333")
    left += v
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_title("Share of photos in each type of group (K = 5)", fontsize=9, loc="left")
for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(); plt.savefig(R / "exp10_drivers.png", dpi=130); plt.close()
