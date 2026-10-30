"""Exp 1: Nearest shrunken centroids (Tibshirani 2002), performance vs number of retained coordinates.

Basis: PCA-64 fit on train EfficientNet vectors (coordinates are PCA components, NOT biological features).
Tasks:
  A. organ (external label, BioImages organ_category, 7 classes; 'other' dropped)
  B. "grouping-approximation" task: existing EfficientNet K=20 cluster IDs. Circular by construction
     (clusters were made from these same vectors) - shows how few components reproduce the grouping,
     nothing about biology.
Protocol: Delta grid; 5-fold GroupKFold on train (groups = individual plant) -> mean +/- SE macro-F1 and
retained coordinates; 1-SE rule picks Delta; then single evaluation on test (211) for chosen Delta vs Delta=0.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from common import OUT, SEED, load, save_json, input_record

meta, X = load()
D = 64


class NSC:
    def fit(self, Z, y, delta):
        self.classes = np.unique(y)
        n, K = len(y), len(self.classes)
        xbar = Z.mean(0)
        cents = np.stack([Z[y == k].mean(0) for k in self.classes])
        nk = np.array([(y == k).sum() for k in self.classes])
        within = sum(((Z[y == k] - cents[i]) ** 2).sum(0) for i, k in enumerate(self.classes))
        s = np.sqrt(within / (n - K)); s0 = np.median(s); self.sd = s + s0
        m = np.sqrt(1 / nk - 1 / n)[:, None]
        d = (cents - xbar) / (m * self.sd)
        d_sh = np.sign(d) * np.maximum(np.abs(d) - delta, 0)
        self.active = d_sh != 0  # K x p
        self.cents = xbar + m * self.sd * d_sh
        self.logprior = np.log(nk / n)
        return self

    def predict(self, Z):
        disc = (((Z[:, None, :] - self.cents[None]) / self.sd) ** 2).sum(2) - 2 * self.logprior
        return self.classes[disc.argmin(1)]

    def n_coords(self):
        return int(self.active.any(0).sum()), float(self.active.sum(1).mean())


tr = (meta.split == "train").values
te = (meta.split == "test").values
pca = PCA(D, whiten=True, random_state=SEED).fit(X[tr])
Z = pca.transform(X)
grid = np.round(np.concatenate([[0], np.linspace(0.25, 8, 32)]), 3)
gkf = GroupKFold(5)
results, curves = {}, []
for task, ycol in [("organ (external label)", "organ"), ("grouping approximation: own K=20 cluster ID", "own_k20")]:
    y = meta[ycol].astype(str).values
    keep = y != "other"
    trk, tek = tr & keep, te & keep
    Ztr, ytr, gtr = Z[trk], y[trk], meta.individual_id.values[trk]
    for delta in grid:
        f1s, nc = [], []
        for a, b in gkf.split(Ztr, ytr, gtr):
            m = NSC().fit(Ztr[a], ytr[a], delta)
            f1s.append(f1_score(ytr[b], m.predict(Ztr[b]), average="macro")); nc.append(m.n_coords())
        full = NSC().fit(Ztr, ytr, delta)
        curves.append(dict(task=task, delta=delta, cv_macroF1=np.mean(f1s), cv_se=np.std(f1s, ddof=1) / np.sqrt(5),
                           coords_any_class=full.n_coords()[0], coords_mean_per_class=full.n_coords()[1]))
    c = pd.DataFrame([r for r in curves if r["task"] == task])
    best = c.cv_macroF1.idxmax()
    thr = c.cv_macroF1[best] - c.cv_se[best]
    chosen = c[(c.cv_macroF1 >= thr)].delta.max()
    test = {}
    for name, dl in [("delta0", 0.0), ("delta_1se", chosen)]:
        m = NSC().fit(Ztr, ytr, dl)
        test[name] = dict(delta=float(dl), test_macroF1=f1_score(y[tek], m.predict(Z[tek]), average="macro"),
                          test_acc=float((m.predict(Z[tek]) == y[tek]).mean()), coords_any_class=m.n_coords()[0],
                          coords_mean_per_class=m.n_coords()[1])
    # which components survive per class at the 1-SE delta
    m = NSC().fit(Ztr, ytr, chosen)
    surv = {str(k): [int(j) + 1 for j in np.where(m.active[i])[0]][:12] for i, k in enumerate(m.classes)}
    results[task] = dict(n_train=int(trk.sum()), n_test=int(tek.sum()), n_classes=len(np.unique(ytr)),
                         best_cv_delta=float(c.delta[best]), one_se_delta=float(chosen), test=test,
                         surviving_PCs_at_1se_first12=surv,
                         ratio_F1_1se_over_delta0=test["delta_1se"]["test_macroF1"] / test["delta0"]["test_macroF1"])
    print(task, results[task]["test"])
pd.DataFrame(curves).to_csv(OUT / "exp1_nsc_curves.csv", index=False)
save_json("exp1_nsc_summary.json", dict(inputs=input_record(), basis=f"PCA-{D} whitened, fit on train",
                                        warning="Coordinates are PCA components, not biological features.", results=results))

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
cv = pd.DataFrame(curves)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (task, g) in zip(axes, cv.groupby("task", sort=False)):
    ax.errorbar(g.coords_any_class, g.cv_macroF1, yerr=g.cv_se, marker="o", ms=3, lw=1)
    r = results[task]
    ax.axhline(g.cv_macroF1.iloc[0], color="grey", ls=":", label="Δ=0 (all 64)")
    sel = g[g.delta == r["one_se_delta"]].iloc[0]
    ax.scatter([sel.coords_any_class], [sel.cv_macroF1], color="red", zorder=5, label=f"1-SE Δ={sel.delta}")
    ax.set_xlabel("PCA coordinates retained by ≥1 class"); ax.set_ylabel("5-fold group-CV macro-F1 (train)")
    ax.set_title(task, fontsize=9); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "exp1_nsc_curve.png", dpi=120)
