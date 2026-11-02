"""Exp 1c: section-1 line charts for every model that has exp1 results (plots only, no new computation).
  exp1c_nsc_organ_models.png     organ task, one line per model
  exp1c_nsc_grouping_models.png  one panel per model: organ task (dashed) + grouping task for each K in K_LIST
x = mean PCA coordinates kept per class (of 64); y = CV macro-F1 relative to Δ=0; dot = Δ chosen by the 1-SE rule."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import K_LIST

R = Path(__file__).resolve().parent / "results"
MODELS = {"EfficientNet-B0": R, "BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3",
          "YOLO11s backbone": R / "yolo11s_backbone", "YOLO11s neck": R / "yolo11s_neck"}
MODELS = {m: d for m, d in MODELS.items() if (d / "exp1_nsc_curves.csv").exists()}


def curve(ax, d, task, **kw):
    c = pd.read_csv(d / "exp1_nsc_curves.csv")
    c = c[c.task == task]
    if c.empty:
        return
    y = c.cv_macroF1 / c.cv_macroF1.iloc[0]
    line = ax.plot(c.coords_mean_per_class, y, ms=2.5, lw=1.3, marker="o", **kw)[0]
    sel = json.loads((d / "exp1_nsc_summary.json").read_text())["results"][task]["one_se_delta"]
    i = (c.delta - sel).abs().values.argmin()
    ax.scatter([c.coords_mean_per_class.iloc[i]], [y.iloc[i]], s=45, color=line.get_color(), edgecolor="k", zorder=5)


def style(ax, title):
    ax.axhline(0.9, color="grey", ls=":", lw=1); ax.invert_xaxis(); ax.set_ylim(0, 1.1)
    ax.set_xlabel("mean PCA coordinates kept per class (of 64)", fontsize=8); ax.set_ylabel("F1 / F1 at Δ=0", fontsize=8)
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7); ax.legend(fontsize=6.5, loc="lower left")


fig, ax = plt.subplots(figsize=(6.5, 3.8))
for m, d in MODELS.items():
    curve(ax, d, "organ (external label)", label=m)
style(ax, "NSC, organ task (external label): accuracy vs number of coordinates")
plt.tight_layout(); plt.savefig(R / "exp1c_nsc_organ_models.png", dpi=120); plt.close()

fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharey=True)
axes = axes.ravel()
for a in axes[len(MODELS):]:
    a.axis("off")
for ax, (m, d) in zip(axes, MODELS.items()):
    curve(ax, d, "organ (external label)", label="organ label", color="k", ls="--")
    for k, col in zip(K_LIST, ["#1b9e77", "#d95f02", "#7570b3"]):
        curve(ax, d, f"grouping approximation: K={k} cluster ID", label=f"grouping K={k}", color=col)
    style(ax, m)
plt.tight_layout(); plt.savefig(R / "exp1c_nsc_grouping_models.png", dpi=120); plt.close()
print(list(MODELS))
