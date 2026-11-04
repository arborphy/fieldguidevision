"""Exp 11: latent-space walk (EfficientNet, full embedding space, unsupervised groups).
Start and end points are centres of EfficientNet's own K = 7 groups (the purest leaf group, the fruit group, the whole-tree
group). Between two centres we take 5 equally spaced points (linear interpolation, re-normalised) and retrieve the
nearest real photos (cosine) for each point, never repeating a photo. Output: exp11_walk.json."""
import json

import numpy as np

from common import OUT, load

meta, X = load()
lab = np.load(OUT / "clusters_k7.npy")
cent = {c: X[lab == c].mean(0) for c in np.unique(lab)}
share = {c: meta[lab == c].organ.value_counts(normalize=True) for c in cent}
pick = lambda organ: max(cent, key=lambda c: share[c].get(organ, 0))
walks = {}
for a, b in [("leaf", "fruit"), ("leaf", "whole tree")]:
    ca, cb = pick(a), pick(b)
    used, steps = set(), []
    for t in np.linspace(0, 1, 5):
        v = (1 - t) * cent[ca] + t * cent[cb]
        v /= np.linalg.norm(v)
        order = [i for i in np.argsort(-(X @ v)) if i not in used][:3]
        used.update(order)
        near = np.argsort(-(X @ v))[:20]
        steps.append(dict(t=float(t), photos=meta.iloc[order][["image_id", "species", "organ", "shot_scale"]].to_dict("records"),
                          share_start=float((meta.organ.values[near] == a).mean()), share_end=float((meta.organ.values[near] == b).mean())))
    walks[f"{a} -> {b}"] = dict(start_group=int(ca), end_group=int(cb), start_share=float(share[ca][a]), end_share=float(share[cb][b]), steps=steps)
    print(a, "->", b, [(round(s["share_start"], 2), round(s["share_end"], 2)) for s in steps])
(OUT / "exp11_walk.json").write_text(json.dumps(walks, indent=2))
