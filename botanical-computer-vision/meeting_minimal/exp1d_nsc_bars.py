"""Exp 1d: per model, coordinates per class needed to keep 90% of the full accuracy (read off the existing NSC curves),
for the organ task and for reproducing the model's own K = 7 grouping. Plot only."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
MODELS = {"EfficientNet-B0": R, "DINOv3": R / "dinov3", "YOLO11s backbone": R / "yolo11s_backbone",
          "YOLO11s neck": R / "yolo11s_neck", "BioCLIP 2.5": R / "bioclip"}


def need90(c, task):
    g = c[c.task == task]
    return float(g[g.cv_macroF1 / g.cv_macroF1.iloc[0] >= 0.9].coords_mean_per_class.min())


out = {}
for m, d in MODELS.items():
    c = pd.read_csv(d / "exp1_nsc_curves.csv")
    out[m] = dict(organ=need90(c, "organ (external label)"), own_k7=need90(c, "grouping approximation: K=7 cluster ID"))
(R / "exp1d_nsc_bars.json").write_text(json.dumps(out, indent=2))

fig, ax = plt.subplots(figsize=(5.6, 2.6))
y = np.arange(len(out))
ax.barh(y - 0.18, [v["organ"] for v in out.values()], 0.34, color="#dd8452", label="tell organs apart")
ax.barh(y + 0.18, [v["own_k7"] for v in out.values()], 0.34, color="#9e9e9e", label="reproduce its own 7 groups")
for i, v in enumerate(out.values()):
    ax.text(v["organ"] + 0.8, i - 0.18, f"{v['organ']:.0f}", va="center", fontsize=7.5)
    ax.text(v["own_k7"] + 0.8, i + 0.18, f"{v['own_k7']:.0f}", va="center", fontsize=7.5, color="#555")
ax.set_yticks(y); ax.set_yticklabels(list(out), fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 64)
ax.set_xlabel("Coordinates per class needed to keep 90% accuracy (of 64)", fontsize=8); ax.tick_params(labelsize=8)
ax.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(R / "exp1d_nsc_bars.png", dpi=130); plt.close()
print(json.dumps(out))
