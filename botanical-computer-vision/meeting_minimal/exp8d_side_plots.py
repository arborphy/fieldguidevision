"""Exp 8d: small companion figures for Tables 1.1-1.3 (plots only; same colours as Figure 1.1)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
C = {"neighbours_kept": "#8c8c8c", "organ_kept": "#dd8452", "scale_kept": "#4c72b0", "species_kept": "#55a868"}
NAME = {"neighbours_kept": "Neighbours", "organ_kept": "Organ", "scale_kept": "Shot scale", "species_kept": "Species"}


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8)


# 1.1 heatmap: EfficientNet, views x information kept
t = pd.read_csv(R / "exp8_info_matrix.csv").set_index("view")[list(C)].clip(upper=1)
fig, ax = plt.subplots(figsize=(4.2, 3.4))
ax.imshow(t.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        v = t.values[i, j]
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > .6 else "#222")
ax.set_xticks(range(t.shape[1])); ax.set_xticklabels([NAME[c] for c in t.columns], fontsize=8)
ax.set_yticks(range(t.shape[0])); ax.set_yticklabels(t.index, fontsize=8)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("Darker = more information kept", fontsize=9, loc="left")
plt.tight_layout(); plt.savefig(R / "exp8_side_heatmap.png", dpi=150); plt.close()

# 1.2 dumbbell: organ kept at 3-D -> 16-D -> 64-D, five models
models = {"EfficientNet-B0": R, "BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3",
          "YOLO11s backbone": R / "yolo11s_backbone", "YOLO11s neck": R / "yolo11s_neck"}
fig, ax = plt.subplots(figsize=(4.2, 3.0))
for i, (m, d) in enumerate(models.items()):
    v = pd.read_csv(d / "exp8_info_matrix.csv").set_index("view").organ_kept.clip(upper=1)
    xs = [v["PCA-3"], v["PCA-16"], v["PCA-64"]]
    ax.plot(xs, [i] * 3, color="#dddddd", lw=3, zorder=1)
    for x, mk, lab in zip(xs, ["o", "s", "D"], ["3-D", "16-D", "64-D"]):
        ax.scatter(x, i, marker=mk, s=40, color=C["organ_kept"], alpha={"o": .45, "s": .7, "D": 1}[mk], zorder=2, label=lab if i == 0 else None)
ax.set_yticks(range(len(models))); ax.set_yticklabels(list(models), fontsize=8); ax.invert_yaxis()
ax.set_xlim(0.6, 1.02); ax.set_xticks([.6, .7, .8, .9, 1]); ax.set_xticklabels(["60%", "70%", "80%", "90%", "100%"])
ax.set_xlabel("Organ information kept", fontsize=8); ax.legend(fontsize=7, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
ax.grid(axis="x", color="#eeeeee"); ax.set_axisbelow(True); clean(ax)
plt.tight_layout(); plt.savefig(R / "exp8_side_dumbbell.png", dpi=150); plt.close()

# 1.3 grouped bars: eta^2 per PC
p = pd.read_csv(R / "exp8_pc_information.csv").head(6)
fig, ax = plt.subplots(figsize=(4.2, 3.0))
w = 0.26
for k, (col, key) in enumerate([("organ", "organ_kept"), ("scale", "scale_kept"), ("species", "species_kept")]):
    ax.bar(np.arange(6) + (k - 1) * w, p[col], w, color=C[key], label=NAME[key])
ax.set_xticks(range(6)); ax.set_xticklabels([f"PC{i}" for i in p.pc]); ax.set_ylim(0, 1)
ax.set_ylabel("Share explained (η²)", fontsize=8); ax.legend(fontsize=7, frameon=False)
ax.grid(axis="y", color="#eeeeee"); ax.set_axisbelow(True); clean(ax)
plt.tight_layout(); plt.savefig(R / "exp8_side_pcbars.png", dpi=150); plt.close()
print("ok")
