"""Exp 8b: one-glance figure for the information-retention matrix (plots only).
x = PCA dimensions kept; y = information kept (% of full space); bold = mean of five models, thin = each model;
grey band = a 3-D plot. Lines are labelled at their ends instead of a legend."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path(__file__).resolve().parent / "results"
DIRS = [R, R / "bioclip", R / "dinov3", R / "yolo11s_backbone", R / "yolo11s_neck"]
im = pd.concat([pd.read_csv(d / "exp8_info_matrix.csv").assign(model=d.name) for d in DIRS if (d / "exp8_info_matrix.csv").exists()])
im = im[im.view.str.startswith("PCA")].assign(dim=lambda t: t.view.str.split("-").str[1].astype(int))
dims = [2, 3, 16, 64]
pos = {d: i for i, d in enumerate(dims)}
SER = [("scale_kept", "Shot scale", "#4c72b0"), ("organ_kept", "Organ", "#dd8452"), ("species_kept", "Species", "#55a868"),
       ("neighbours_kept", "Nearest neighbours", "#8c8c8c")]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.axvspan(pos[3] - 0.25, pos[3] + 0.25, color="#eeeeee", zorder=0)
ax.text(pos[3], 1.06, "3-D plot", ha="center", fontsize=9, color="#555")
ends = []
for col, name, c in SER:
    for _, g in im.groupby("model"):
        g = g.sort_values("dim")
        ax.plot([pos[d] for d in g.dim], g[col].clip(upper=1), color=c, lw=0.8, alpha=0.35)
    m = im.groupby("dim")[col].mean().clip(upper=1)
    ax.plot([pos[d] for d in m.index], m.values, color=c, lw=2.8, marker="o", ms=5, ls="--" if col == "neighbours_kept" else "-")
    ends.append((m.loc[64], f"{name}  {m.loc[64]:.0%}", c))
    ax.text(pos[3], m.loc[3] - 0.045, f"{m.loc[3]:.0%}", ha="center", va="top", fontsize=8, color=c, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
# end labels, spaced so they never overlap
last = 9.0
for y, txt, c in sorted(ends, reverse=True):
    y = min(y, last - 0.065); last = y
    ax.text(pos[64] + 0.12, y, txt, va="center", fontsize=9, color=c, fontweight="bold")
ax.set_xticks(range(len(dims))); ax.set_xticklabels([f"{d}-D" for d in dims])
ax.set_xlim(-0.3, len(dims) - 0.3 + 1.1); ax.set_ylim(0, 1.12)
ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("PCA dimensions kept"); ax.set_ylabel("Information kept (full space = 100%)")
ax.grid(axis="y", color="#e5e5e5", lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("What survives dimension reduction (bold = mean of 5 models, thin = each model)", fontsize=10, loc="left")
plt.tight_layout(); plt.savefig(R / "exp8_info_kept.png", dpi=150)
print("ok")
