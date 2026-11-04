"""Exp 5b: AE vs PCA (32 dimensions, recovered table): neighbour retention on the 211 test photos. Plot only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
t = pd.read_csv(R / "exp5_recovered_ae_table.csv")
names = {"bioclip25": "BioCLIP 2.5", "dinov3": "DINOv3", "efficientnet_b0": "EfficientNet-B0"}
models = ["efficientnet_b0", "dinov3", "bioclip25"]
fig, ax = plt.subplots(figsize=(4.6, 2.2))
y = np.arange(len(models))
for off, meth, col in [(-0.18, "PCA", "#1f3a5f"), (0.18, "Autoencoder", "#bdbdbd")]:
    v = [t[(t.model == m) & (t.method == meth)].knn15_retention.iloc[0] for m in models]
    ax.barh(y + off, v, 0.34, color=col, label=meth)
    for i, x in enumerate(v):
        ax.text(x + 0.01, i + off, f"{x:.0%}", va="center", fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels([names[m] for m in models], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.set_xticks([0, .5, 1]); ax.set_xticklabels(["0", "50%", "100%"], fontsize=8)
ax.set_xlabel("Nearest neighbours kept (32 dimensions, test photos)", fontsize=8)
ax.legend(fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(R / "exp5b_ae.png", dpi=130); plt.close()
