"""Exp 9b: figures for section 2 (plots only).
  exp9_k_metrics.png  one panel per metric, x = K, one line per model, dot = best K
  exp9_k_votes.png    model x metric, cell = best K (companion to the vote table)
  exp9_k5_groups.png  organ make-up of the five K=5 groups, EfficientNet and DINOv3"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
MODELS = {"EfficientNet-B0": R, "BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3",
          "YOLO11s backbone": R / "yolo11s_backbone", "YOLO11s neck": R / "yolo11s_neck"}
MCOL = {"EfficientNet-B0": "#1f3a5f", "BioCLIP 2.5": "#8c564b", "DINOv3": "#2a9d8f", "YOLO11s backbone": "#999999", "YOLO11s neck": "#cccccc"}
MET = [("silhouette", "Silhouette ↑"), ("davies_bouldin", "Davies–Bouldin ↓"), ("calinski_harabasz", "Calinski–Harabasz ↑"),
       ("bic", "BIC ↓"), ("stability", "Stability ↑"), ("organ_ami", "Agreement with organ ↑ (check only)")]
D = {m: pd.read_csv(d / "exp9_k_metrics.csv") for m, d in MODELS.items() if (d / "exp9_k_metrics.csv").exists()}
B = {m: json.loads((d / "exp9_k_metrics.json").read_text())["best_k"] for m, d in MODELS.items() if m in D}

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
for ax, (col, title) in zip(axes.ravel(), MET):
    for m, t in D.items():
        y = t[col] / t[col].abs().max() if col in ("calinski_harabasz", "bic") else t[col]
        ax.plot(t.K, y, color=MCOL[m], lw=1.6, marker="o", ms=3, label=m)
        kb = B[m][col]
        ax.scatter([kb], [y[t.K == kb].iloc[0]], s=55, facecolor="white", edgecolor=MCOL[m], lw=1.8, zorder=5)
    ax.axvspan(4.5, 7.5, color="#f1efe8", zorder=0)
    ax.set_title(title, fontsize=9.5, loc="left"); ax.set_xticks([3, 5, 7, 10, 15, 20]); ax.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if col in ("calinski_harabasz", "bic"):
        ax.set_ylabel("relative to max |value|", fontsize=7)
axes[0, 0].legend(fontsize=7, frameon=False)
fig.supxlabel("K (number of groups); open dot = best K for that model; shaded band = K 5–7", fontsize=9)
plt.tight_layout(); plt.savefig(R / "exp9_k_metrics.png", dpi=140); plt.close()

votes = pd.DataFrame(B).T[[c for c, _ in MET]]
fig, ax = plt.subplots(figsize=(4.4, 2.6))
ax.imshow(np.where(votes.values.astype(float) <= 7, 1, 0), cmap="Blues", vmin=-0.6, vmax=1.4, aspect="auto")
for i in range(votes.shape[0]):
    for j in range(votes.shape[1]):
        ax.text(j, i, int(votes.values[i, j]), ha="center", va="center", fontsize=9)
ax.set_xticks(range(votes.shape[1])); ax.set_xticklabels(["Sil.", "DB", "CH", "BIC", "Stab.", "Organ"], fontsize=8)
ax.set_yticks(range(votes.shape[0])); ax.set_yticklabels(votes.index, fontsize=8)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("Best K per metric (blue = K ≤ 7)", fontsize=9, loc="left")
plt.tight_layout(); plt.savefig(R / "exp9_k_votes.png", dpi=150); plt.close()

organs = ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark", "cone", "other"]
ocol = ["#4c9a2a", "#8c6d31", "#c2185b", "#f57c00", "#7b5ea7", "#607d8b", "#1e88e5", "#cccccc"]
fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
for ax, m in zip(axes, ["EfficientNet-B0", "DINOv3"]):
    g = pd.DataFrame(json.loads((MODELS[m] / "exp9_k_metrics.json").read_text())["k5_groups"])
    left = np.zeros(len(g))
    for o, c in zip(organs, ocol):
        share = g.organs.apply(lambda s: next((float(p.split()[-1][:-1]) / 100 for p in s.split(", ") if p.rsplit(" ", 1)[0] == o), 0.0)).values
        ax.barh(range(len(g)), share, left=left, color=c, label=o); left += share
    ax.set_yticks(range(len(g))); ax.set_yticklabels([f"G{c} (n={n})" for c, n in zip(g.cluster, g.n)], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_title(f"{m}: organ make-up of the 5 groups (top-3 organs shown)", fontsize=9, loc="left"); ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout(); plt.savefig(R / "exp9_k5_groups.png", dpi=140); plt.close()
print(votes)
