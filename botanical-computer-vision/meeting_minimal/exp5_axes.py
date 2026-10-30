"""Exp 5: (a) AE result recovered from the hosted site (not recomputed) + one sanity check: a LINEAR AE
(32-d, train 1,181 -> test 211) should match PCA-32 reconstruction. (b) Axis reading: PCs 1-5 of EfficientNet
(fit on train), rank correlation of PC score with shot scale (distant=0, mid=1, close=2), organ one-vs-rest,
and thumbnail brightness; 6 real images at each end.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from common import IMG_DIR, OUT, SEED, load, save_json, input_record

meta, X = load()
tr, te = (meta.split == "train").values, (meta.split == "test").values

# (a) recovered table (scale/meeting/ on the hosted site, saved in recovered_site/scale_meeting_index.html)
recovered = [("bioclip25", "PCA", .000284, .667299), ("bioclip25", "Autoencoder", .000280, .616746),
             ("dinov3", "PCA", .000498, .766509), ("dinov3", "Autoencoder", .000491, .745656),
             ("efficientnet_b0", "PCA", .000296, .729542), ("efficientnet_b0", "Autoencoder", .000301, .721327)]
rec = pd.DataFrame(recovered, columns=["model", "method", "test_mse", "knn15_retention"])
rec.to_csv(OUT / "exp5_recovered_ae_table.csv", index=False)

# linear AE sanity
mu = X[tr].mean(0)
pca = PCA(32, random_state=SEED).fit(X[tr])
pca_mse = float(((X[te] - pca.inverse_transform(pca.transform(X[te]))) ** 2).mean())
torch.manual_seed(SEED)
A = torch.tensor(X[tr] - mu); T = torch.tensor(X[te] - mu)
enc = torch.nn.Linear(X.shape[1], 32, bias=False); dec = torch.nn.Linear(32, X.shape[1], bias=False)
opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
for ep in range(3000):
    opt.zero_grad(); loss = ((dec(enc(A)) - A) ** 2).mean(); loss.backward(); opt.step()
with torch.no_grad():
    lin_mse = float(((dec(enc(T)) - T) ** 2).mean())

# (b) axis reading
pc = PCA(10, random_state=SEED).fit(X[tr])
S = pc.transform(X)
bright = np.array([np.asarray(Image.open(IMG_DIR / "tn" / t).convert("L"), dtype=float).mean() for t in meta.thumb])
scale_ord = meta.shot_scale.map({"distant": 0, "mid-range": 1, "close-up": 2})
ok = scale_ord.notna().values
rows, ends = [], {}
for k in range(5):
    r = dict(pc=k + 1, var=float(pc.explained_variance_ratio_[k]),
             rho_scale_close=spearmanr(S[ok, k], scale_ord[ok])[0], rho_brightness=spearmanr(S[:, k], bright)[0])
    for o in ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark"]:
        r[f"rho_{o}"] = spearmanr(S[:, k], meta.organ == o)[0]
    o_r = {o: r[f"rho_{o}"] for o in ["leaf", "whole tree", "inflorescence", "fruit", "twig", "bark"]}
    lo, hi = min(o_r, key=o_r.get), max(o_r, key=o_r.get)
    r["reading"] = (f"low end ~ {lo}, high end ~ {hi}" if max(abs(o_r[lo]), abs(o_r[hi])) >= 0.2
                    else "no clear organ correlate (all |rho|<0.2)")
    rows.append(r)
    order = np.argsort(S[:, k])
    ends[k + 1] = dict(low=meta.iloc[order[:6]][["image_id", "thumb", "species", "organ", "shot_scale"]].to_dict("records"),
                       high=meta.iloc[order[-6:][::-1]][["image_id", "thumb", "species", "organ", "shot_scale"]].to_dict("records"))
ax = pd.DataFrame(rows)
ax.to_csv(OUT / "exp5_axis_table.csv", index=False)
save_json("exp5_axes_summary.json", dict(inputs=input_record(), recovered_ae_source="hosted site scale/meeting/ (Experiment 6 table)",
                                         linear_ae_sanity=dict(pca32_test_mse=pca_mse, linear_ae32_test_mse=lin_mse,
                                                               ratio=lin_mse / pca_mse, epochs=3000),
                                         axes=rows, ends=ends))
print(ax.round(3).to_string()); print("PCA", pca_mse, "linAE", lin_mse)
