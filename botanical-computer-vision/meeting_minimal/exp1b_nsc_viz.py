"""Exp 1b: what are the PCA coordinates NSC keeps for organ (EfficientNet), plus a 3-model NSC curve.
 - Refit NSC at the 1-SE Delta from exp1 (same PCA-64 whitened basis, train only), save organ x PC heatmap
   of shrunken standardised deviations (0 = coordinate dropped for that organ).
 - For each surviving PC: Spearman with shot scale / organs / brightness, and 4 real images at each end.
 - Line chart: relative CV macro-F1 vs mean coordinates per class, organ task, all 3 models (results/*/exp1_nsc_curves.csv).
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from common import IMG_DIR, OUT, SEED, load, save_json, input_record

meta, X = load()
tr = (meta.split == "train").values
Z = PCA(64, whiten=True, random_state=SEED).fit(X[tr]).transform(X)
delta = json.loads((OUT / "exp1_nsc_summary.json").read_text())["results"]["organ (external label)"]["one_se_delta"]

# NSC deviations (same formulas as exp1_nsc.NSC)
y = meta.organ.values
k_ = tr & (y != "other")
Zt, yt = Z[k_], y[k_]
cls = np.unique(yt); n, K = len(yt), len(cls)
cents = np.stack([Zt[yt == c].mean(0) for c in cls]); nk = np.array([(yt == c).sum() for c in cls])
s = np.sqrt(sum(((Zt[yt == c] - cents[i]) ** 2).sum(0) for i, c in enumerate(cls)) / (n - K))
d = (cents - Zt.mean(0)) / (np.sqrt(1 / nk - 1 / n)[:, None] * (s + np.median(s)))
dsh = np.sign(d) * np.maximum(np.abs(d) - delta, 0)
keep = np.where((dsh != 0).any(0))[0]
pd.DataFrame(dsh[:, keep], index=cls, columns=[f"PC{j + 1}" for j in keep]).to_csv(OUT / "exp1b_nsc_deviations.csv")

fig, ax = plt.subplots(figsize=(11, 3.4))
v = np.abs(dsh[:, keep]).max()
ax.imshow(dsh[:, keep], cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(len(keep))); ax.set_xticklabels([f"PC{j + 1}" for j in keep], fontsize=8, rotation=90)
ax.set_yticks(range(K)); ax.set_yticklabels(cls, fontsize=9)
ax.set_title(f"EfficientNet NSC organ fingerprint (Δ={delta}): red = organ sits at + end, blue = − end, white = dropped", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "exp1b_nsc_heatmap.png", dpi=120); plt.close()

# axis reading for surviving PCs
bright = np.array([np.asarray(Image.open(IMG_DIR / "tn" / t).convert("L"), dtype=float).mean() for t in meta.thumb])
sc = meta.shot_scale.map({"distant": 0, "mid-range": 1, "close-up": 2}); ok = sc.notna().values
cols = ["image_id", "species", "organ", "shot_scale"]
axes = []
for j in keep:
    rho = {o: spearmanr(Z[:, j], y == o)[0] for o in cls}
    lo, hi = min(rho, key=rho.get), max(rho, key=rho.get)
    o = np.argsort(Z[:, j])
    axes.append(dict(pc=int(j + 1), used_by=", ".join(cls[dsh[:, j] != 0]),
                     rho_scale_close=spearmanr(Z[ok, j], sc[ok])[0], rho_brightness=spearmanr(Z[:, j], bright)[0],
                     reading=(f"− {lo} ({rho[lo]:.2f}) / + {hi} ({rho[hi]:.2f})" if max(-rho[lo], rho[hi]) >= 0.2
                              else "no clear organ correlate (|ρ|<0.2)"),
                     low=meta.iloc[o[:4]][cols].to_dict("records"), high=meta.iloc[o[-4:][::-1]][cols].to_dict("records")))

# 3-model curve
fig, ax = plt.subplots(figsize=(6.5, 3.6))
for name, sub in [("EfficientNet-B0", ""), ("BioCLIP 2.5", "bioclip/"), ("DINOv3", "dinov3/")]:
    c = pd.read_csv(OUT / f"{sub}exp1_nsc_curves.csv")
    c = c[c.task.str.startswith("organ")]
    ax.plot(c.coords_mean_per_class, c.cv_macroF1 / c.cv_macroF1.iloc[0], marker="o", ms=3, label=name)
ax.axhline(0.9, color="grey", ls=":", lw=1)
ax.set_xlabel("mean PCA coordinates kept per organ class (of 64)"); ax.set_ylabel("CV macro-F1 / F1 at Δ=0")
ax.set_title("NSC organ task: accuracy vs sparsity, 3 models", fontsize=9); ax.legend(fontsize=8); ax.invert_xaxis()
plt.tight_layout(); plt.savefig(OUT / "exp1b_nsc_3models.png", dpi=120); plt.close()

save_json("exp1b_nsc_viz.json", dict(inputs=input_record(), delta=delta, n_kept=len(keep), axes=axes))
print(len(keep), [a["reading"] for a in axes])
