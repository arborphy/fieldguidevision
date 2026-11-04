"""Exp 9d: what the groups are at different K (full embedding space, K-means seed 42), plots + numbers.
  exp9d_eff_k357.png   EfficientNet groups at K = 3, 5, 7: organ make-up, distant share in the label
  exp9d_models_k5.png  DINOv3, YOLO11s backbone, BioCLIP at K = 5
  exp9d_match.json     one-to-one match with shot scale at K = 3 and with organ at K = 7, four models"""
import importlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import pandas as pd
from sklearn.cluster import KMeans

R = Path(__file__).resolve().parent / "results"
ORG = ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark", "cone", "other"]
OCOL = ["#4c9a2a", "#8c6d31", "#c2185b", "#f57c00", "#7b5ea7", "#607d8b", "#1e88e5", "#cccccc"]


def data(model):
    os.environ["FGV_MODEL"] = model
    import common
    importlib.reload(common)
    return common.load()


def match(lab, y):
    ct = pd.crosstab(lab, y).values
    r, c = linear_sum_assignment(-ct)
    return float(ct[r, c].sum() / len(y))


def bars(ax, meta, lab, title):
    groups = sorted(np.unique(lab), key=lambda c: -(lab == c).sum())
    left = np.zeros(len(groups))
    for o, col in zip(ORG, OCOL):
        v = np.array([(meta.organ[lab == c] == o).mean() for c in groups])
        ax.barh(range(len(groups)), v, left=left, color=col, height=0.72, label=o); left += v
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"G{i + 1} n={(lab == c).sum()} · distant {(meta.shot_scale[lab == c] == 'distant').mean():.0%}"
                        for i, c in enumerate(groups)], fontsize=7)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_title(title, fontsize=8.5, loc="left")
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)


res = {}
cache = {}
for model in ["efficientnet", "dinov3", "yolo11s_backbone", "bioclip"]:
    meta, X = data(model)
    labs = {k: KMeans(k, n_init=10, random_state=42).fit_predict(X) for k in (3, 5, 7)}
    ok = (meta.shot_scale != "uncertain").values
    res[model] = dict(k3_scale=match(labs[3][ok], meta.shot_scale[ok]), k5_organ=match(labs[5], meta.organ), k7_organ=match(labs[7], meta.organ))
    cache[model] = (meta, labs)
(R / "exp9d_match.json").write_text(json.dumps(res, indent=2))

meta, labs = cache["efficientnet"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), gridspec_kw={"width_ratios": [3, 5, 7]})
for ax, k in zip(axes, (3, 5, 7)):
    bars(ax, meta, labs[k], f"EfficientNet, K = {k}")
axes[-1].legend(fontsize=6.5, frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout(); plt.savefig(R / "exp9d_eff_k357.png", dpi=125); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(11, 2.4))
for ax, (model, name) in zip(axes, [("dinov3", "DINOv3"), ("yolo11s_backbone", "YOLO11s backbone"), ("bioclip", "BioCLIP 2.5")]):
    meta, labs = cache[model]
    bars(ax, meta, labs[5], f"{name}, K = 5")
axes[-1].legend(fontsize=6.5, frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout(); plt.savefig(R / "exp9d_models_k5.png", dpi=125); plt.close()
print(json.dumps(res, indent=1))
