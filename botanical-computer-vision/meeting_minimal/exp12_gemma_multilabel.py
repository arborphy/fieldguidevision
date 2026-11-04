"""Exp 12: do photos that lie between two organ groups (soft clustering, EfficientNet K = 7) show both organs?
Reference: Gemma-3-27B multi-label tags (bioimages_browser/tagging/gemma_reference.csv; model output, not ground truth),
mapped to the 7 organ classes. For every photo the pair = main organs of its top-1 and top-2 groups; we compare how often
Gemma tags BOTH organs for ambiguous photos (max membership < 0.8, groups of different organs) vs clear photos.
Also: hypergeometric (one-sided Fisher) p-values for the organ enrichment of each K = 5 group, BH-adjusted - reported
as is; they are too small because photos of the same plant are not independent."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from common import BCV, load, describe_clusters

R = Path(__file__).resolve().parent / "results"
MAP = {"leaf": ["leaf", "leaf upper surface", "leaf margin", "needle", "needle attachment", "petiole arrangement", "leaf arrangement on twig"],
       "twig": ["twig", "branch", "bud", "stem", "winter twig", "terminal bud", "leaf scar", "leaf arrangement on twig"],
       "bark": ["bark", "trunk", "large-tree bark", "medium-tree or large-branch bark", "small-tree or branch bark", "view up trunk"],
       "fruit": ["fruit", "fruit borne on plant", "immature fruit", "fruit close-up", "opened or sectioned fruit", "seed"],
       "inflorescence": ["inflorescence", "flower", "flower close-up", "flower interior", "frontal flower view", "lateral flower view",
                         "male reproductive structure", "female reproductive structure"],
       "whole tree": ["whole plant", "winter habit"], "cone": ["cone", "closed cone", "open cone"]}
g = pd.read_csv(BCV / "bioimages_browser/tagging/gemma_reference.csv")
tags = {r.image_id: {o for o, ts in MAP.items() if set(json.loads(r.normalized_tags)) & set(ts)} for r in g.itertuples()}

meta, X = load()
lab7 = np.load(R / "clusters_k7.npy")
d7 = describe_clusters(meta, lab7)
org = d7.top_organ.where(d7.top_organ_share >= .5).to_dict()
m = pd.read_csv(R / "k7" / "exp4_soft_memberships.csv")
m = m[m.K == 7].merge(meta[["image_id", "organ"]], on="image_id")
m["o1"], m["o2"] = m.top1.map(org), m.top2.map(org)
m = m[m.o1.notna() & m.o2.notna() & (m.o1 != m.o2) & m.image_id.isin(tags)]
m["both"] = [{a, b} <= tags[i] for i, a, b in zip(m.image_id, m.o1, m.o2)]
m["amb"] = m.max_membership < 0.8
res = dict(ambiguous=dict(n=int(m.amb.sum()), both_tagged=float(m[m.amb].both.mean())),
           clear=dict(n=int((~m.amb).sum()), both_tagged=float(m[~m.amb].both.mean())),
           by_category={c: dict(n=int(len(x)), both_tagged=float(x.both.mean())) for c, x in m[m.amb].groupby("category_if_ambiguous")},
           by_pair={f"{a} <-> {b}": dict(n=int(len(x)), both_tagged=float(x.both.mean()))
                    for (a, b), x in m[m.amb].assign(a=lambda t: t[["o1", "o2"]].min(1), b=lambda t: t[["o1", "o2"]].max(1)).groupby(["a", "b"]) if len(x) >= 20})

# hypergeometric p-values for each K = 5 group's most enriched organ (five models), BH-adjusted
hp = []
for name, d in [("EfficientNet-B0", R / "k5"), ("DINOv3", R / "dinov3" / "k5"), ("YOLO11s backbone", R / "yolo11s_backbone" / "k5"),
                ("YOLO11s neck", R / "yolo11s_neck" / "k5"), ("BioCLIP 2.5", R / "bioclip" / "k5")]:
    e = pd.read_csv(d / "exp2_enrichment_all.csv")
    e = e[(e.K == 5) & (e.label_type == "organ") & (e.label != "other")]
    for c, x in e.groupby("cluster"):
        b = x.loc[x.fold_enrichment.idxmax()]
        N, K_, n, k = 1641, int(round(b.base_rate * 1641)), int(b.n_images), int(b["count"])
        hp.append(dict(model=name, group=int(c), organ=b.label, fold=float(b.fold_enrichment), p=float(hypergeom.sf(k - 1, N, K_, n))))
hp = pd.DataFrame(hp)
o = hp.p.rank(method="first").values
hp["p_bh"] = np.minimum(1, (hp.p * len(hp) / o))
hp.to_csv(R / "exp12_hypergeom.csv", index=False)
res["hypergeom"] = dict(n_tests=len(hp), max_p_bh_fold_ge_2=float(hp[hp.fold >= 2].p_bh.max()), min_p=float(hp.p.min()))
(R / "exp12_gemma.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=1))
