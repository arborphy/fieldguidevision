"""Exp 9c: section-2 material in the FULL embedding space.
  - K=5 groups (K-means, all 1,641, seed 42) for EfficientNet and DINOv3: organ make-up, close-up / distant share,
    number of species and top-species share -> exp9c_k5_groups.json + exp9c_k5_groups.png
  - stability across K from the full-space sweep (exp4b_k_sweep.csv) -> exp9c_stability.png"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

R = Path(__file__).resolve().parent / "results"
out = {}
for model, d in [("efficientnet", R), ("dinov3", R / "dinov3")]:
    os.environ["FGV_MODEL"] = model
    import importlib, common
    importlib.reload(common)
    meta, X = common.load()
    lab = KMeans(5, n_init=10, random_state=common.SEED).fit_predict(X)
    rows = []
    for c in range(5):
        m = meta[lab == c]
        o = m.organ.value_counts(normalize=True)
        s = m.shot_scale.value_counts(normalize=True)
        rows.append(dict(group=c, n=len(m), organs={k: float(v) for k, v in o.items()},
                         main=o.index[0], main_share=float(o.iloc[0]), second=o.index[1] if len(o) > 1 else "",
                         second_share=float(o.iloc[1]) if len(o) > 1 else 0.0,
                         close_up=float(s.get("close-up", 0)), mid=float(s.get("mid-range", 0)), distant=float(s.get("distant", 0)),
                         n_species=int(m.species.nunique()), top_species_share=float(m.species.value_counts(normalize=True).iloc[0])))
    rows.sort(key=lambda r: -r["n"])
    out[model] = rows
(R / "exp9c_k5_groups.json").write_text(json.dumps(out, indent=2))

organs = ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark", "cone", "other"]
ocol = ["#4c9a2a", "#8c6d31", "#c2185b", "#f57c00", "#7b5ea7", "#607d8b", "#1e88e5", "#cccccc"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.5))
for ax, (model, name) in zip(axes, [("efficientnet", "EfficientNet-B0"), ("dinov3", "DINOv3")]):
    rows = out[model]
    left = np.zeros(len(rows))
    for o, c in zip(organs, ocol):
        v = np.array([r["organs"].get(o, 0) for r in rows])
        ax.barh(range(len(rows)), v, left=left, color=c, label=o, height=0.7); left += v
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"G{i + 1} (n={r['n']})" for i, r in enumerate(rows)], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50%", "100%"], fontsize=8)
    ax.set_title(f"{name}, K = 5: organ make-up of each group", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1.05), loc="upper left")
plt.tight_layout(); plt.savefig(R / "exp9c_k5_groups.png", dpi=130); plt.close()

MC = {"EfficientNet-B0": ("#1f3a5f", R), "DINOv3": ("#2a9d8f", R / "dinov3"), "BioCLIP 2.5": ("#8c564b", R / "bioclip")}
fig, ax = plt.subplots(figsize=(5.2, 2.6))
for m, (c, d) in MC.items():
    t = pd.read_csv(d / "exp4b_k_sweep.csv")
    ax.plot(t.K, t.stability_ari_mean, color=c, lw=1.8, marker="o", ms=3.5, label=m)
ax.set_xscale("log"); ax.set_xticks([3, 5, 7, 10, 20, 63, 100]); ax.set_xticklabels(["3", "5", "7", "10", "20", "63", "100"], fontsize=8)
ax.axvspan(4.6, 7.6, color="#f1efe8", zorder=0); ax.text(5.9, 0.95, "K 5–7", ha="center", fontsize=8, color="#666")
ax.set_ylim(0.35, 1); ax.tick_params(labelsize=8); ax.set_xlabel("K (number of groups, log scale)", fontsize=8.5)
ax.set_ylabel("Stability (same groups after\nresampling plants)", fontsize=8)
ax.legend(fontsize=7.5, frameon=False, loc="lower left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(R / "exp9c_stability.png", dpi=130); plt.close()
for m, rows in out.items():
    for r in rows:
        print(m, r["n"], r["main"], round(r["main_share"], 2), r["second"], round(r["second_share"], 2), "close", round(r["close_up"], 2), "dist", round(r["distant"], 2), "sp", r["n_species"], round(r["top_species_share"], 2))
