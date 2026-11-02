"""Build meeting_minimal/index_en.html (English version of index.html) from results/*.
Images load from bioimages.vanderbilt.edu (never committed)."""
import json
from html import escape
from pathlib import Path

import pandas as pd

import crossmodel
import sections
from common import OUT, load

H = Path(__file__).resolve().parent
meta, _ = load()
url = dict(zip(meta.image_id, meta.thumbnail_url))
J = lambda n: json.loads((OUT / n).read_text())
s1, s2, s3, s4, s5, s6, s7 = (J(f) for f in ["exp1_nsc_summary.json", "exp2_enrichment_summary.json", "exp3_activation_summary.json",
                                              "exp4_soft_summary.json", "exp5_axes_summary.json", "exp6_summary.json", "exp7_retrieval_summary.json"])


def f(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, float) else str(x)


def table(df, cols=None, d=3):
    df = df[cols] if cols else df
    h = "".join(f"<th>{escape(str(c))}</th>" for c in df.columns)
    b = "".join("<tr>" + "".join(f"<td>{escape(f(v, d))}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<div class=tw><table><tr>{h}</tr>{b}</table></div>"


def strip(recs, cap=lambda r: f"{r['organ']} · {r['shot_scale']}"):
    return "<div class=strip>" + "".join(
        f"<figure><a href='{url[r['image_id']]}' target=_blank><img loading=lazy src='{url[r['image_id']]}' alt='{escape(r['image_id'])}'></a>"
        f"<figcaption>{escape(r['image_id'])}<br>{escape(r['species'])}<br>{cap(r)}</figcaption></figure>" for r in recs) + "</div>"


BADGE = {"v": "<span class='b v'>Verified</span>", "e": "<span class='b e'>Exploratory demo</span>",
         "x": "<span class='b x'>Not done / blocked</span>", "r": "<span class='b r'>Recovered earlier result (not recomputed)</span>"}


def section(n, title, ask, did, result, means, badges):
    return (f"<section id=e{n}><h2>{n}. {title}</h2><div>{' '.join(BADGE[b] for b in badges)}</div>"
            f"<dl><dt>Question</dt><dd>{ask}</dd><dt>What was actually run</dt><dd>{did}</dd>"
            f"<dt>Result</dt><dd>{result}</dd><dt>What it supports</dt><dd>{means}</dd></dl></section>")


secs = []
# 1-4 (K-dependent sections)
secs.append(sections.sec1("en", section, table, strip))
secs.append(sections.sec2("en", section, table, strip, s6))
secs.append(sections.sec3("en", section, table, s3))
secs.append(sections.sec4("en", section, table, strip))

# 5 AE / axes
rec = pd.read_csv(OUT / "exp5_recovered_ae_table.csv")
axt = pd.read_csv(OUT / "exp5_axis_table.csv")
la = s5["linear_ae_sanity"]
ends_html = "".join(f"<h4>PC{k} low end</h4>{strip(v['low'])}<h4>PC{k} high end</h4>{strip(v['high'])}" for k, v in list(s5["ends"].items())[:3])
secs.append(section(5, "AE recovery + real images at both ends of PCA axes",
    "Is a nonlinear AE better than PCA? What do the main PCA directions correspond to?",
    "Recovered the AE vs PCA comparison from the Experiment 6 table on the hosted page scale/meeting/ (raw HTML kept in recovered_site/). One added check: "
    "a linear AE (32-D, 3,000 epochs) should match PCA-32. Axis reading: EfficientNet PCA (fit on train) PC1–5, Spearman with scale (distant=0 → close=2), "
    "each organ one-vs-rest, and thumbnail brightness; 6 real images at each end. No architecture search.",
    table(rec, d=6) + f"<p>Linear-AE check: PCA-32 test MSE {la['pca32_test_mse']:.7f}, linear AE {la['linear_ae32_test_mse']:.7f} "
    f"(ratio {la['ratio']:.3f}). This PCA-32 value also equals the hosted EfficientNet PCA value 0.000296, so code and data line up.</p>" +
    table(axt.drop(columns=["reading"]), d=2) + ends_html,
    "AE: the recovered results show AE MSE on par with PCA and worse neighbour retention, for all 3 models. There is no nonlinear structure worth the "
    "extra complexity, so this line stops here (recovered result + verified linear sanity check). "
    "Axes: PC1 vs scale ρ=−0.65 and vs whole tree ρ=+0.62 (close-up leaves ↔ distant whole trees); PC2 ≈ bark ↔ whole tree; "
    "PC3 ≈ inflorescence/fruit ↔ leaf; PC4 has |ρ|&lt;0.1 with every listed label, <b>no readable meaning</b>. Axis names describe correlations, not causes.",
    ["r", "v", "e"]))

# 6 fidelity
t6 = pd.DataFrame(s6["table"])
secs.append(section(6, "Visualisation fidelity",
    "How well do 2-D / higher-dimensional PCA views keep the neighbour structure of the full space?",
    "EfficientNet, PCA fit on train (1,181) only, evaluated on val (249): 15-NN overlap within the val pool, trustworthiness, reconstruction R², "
    "linear probes (C=100, same as the existing benchmark), K=20 ARI. 2-D numbers also on all 1,641 images for comparison with the site.",
    table(t6) + "<img src='results/exp6_pca_fidelity.png' alt='fidelity'>",
    f"2-D keeps only about {s6['two_d_all_images']['knn15_overlap_all1641']:.0%} of 15 nearest neighbours (all 1,641; 33% in the val pool), K=20 ARI 0.23, "
    "so <b>structure cannot be read off 2-D plots</b>. Under the pre-set rule (15-NN ≥ 0.85 and species probe within 1 pp), <b>no d* exists</b> up to 256 "
    "(256-D gives 0.850 / 0.849, on the boundary). The organ probe saturates around 64-D. "
    "(The species probe is low on val to begin with: EfficientNet is not a species model.)", ["v"]))

# 7 retrieval
secs.append(sections.sec7("en", section, table, strip, s7))

# appendix: three models
cm = crossmodel.rows()
appendix = ""
if len(cm) > 1:
    axes_html = "".join(f"<h4>{n}: PC1–3</h4>" + table(crossmodel.axes(n).head(3)[["pc", "var", "rho_scale_close", "rho_brightness", "reading"]], d=2)
                        for n in crossmodel.DIRS if (crossmodel.DIRS[n] / "exp5_axis_table.csv").exists())
    appendix = (f"<section id=e8><h2>Appendix: the same vector-only experiments on BioCLIP 2.5 and DINOv3</h2><div>{BADGE['v']}</div>"
                "<p>Raw BioCLIP 2.5 (1,641 × 1,024) and DINOv3 (1,641 × 768) vectors were recovered from Drive after the main run. SHA-256 recorded, "
                "image_id order and splits checked against the manifest. Items 1, 4, 5, 6 and 7 were rerun unchanged with <code>FGV_MODEL=bioclip|dinov3</code> "
                "(outputs in results/bioclip/, results/dinov3/). Item 3 still needs the BioCLIP / DINOv3 model weights. The NSC grouping task and retrieval use "
                "each model's own existing K=20 clusters. Entropy-vs-error AUROC is EfficientNet-only (per-image predictions of the other models are not in Git).</p>"
                + table(cm) + axes_html + "<p>" + crossmodel.NOTE_EN + "</p></section>")

css = (H / "index.html").read_text().split("<style>")[1].split("</style>")[0]
head = f"""<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FieldGuideVision Minimal Experiments</title><style>{css}</style></head><body><main>
<h1>Seven minimal experiments · 2026-10-30</h1>
<p class=note>Main data: EfficientNet-B0 raw vectors (Git, SHA-256 checked; all 1,641 image_ids and splits matched row by row against
individual_disjoint.csv). Items 1–7 below were run on EfficientNet; BioCLIP / DINOv3 enter item 2 through their existing cluster IDs, and the
appendix reruns the vector-only items on their recovered raw vectors. Images load from bioimages.vanderbilt.edu.
Labels: organ = BioImages organ_category (single label); scale = BioImages + Gemma (provisional).</p>
<p>{' '.join(BADGE.values())}</p>
<p class=note><b>Verified</b> = fixed protocol, held-out data or resampling control, numbers reproducible from the scripts.
<b>Exploratory demo</b> = individual images and readings not validated against independent labels.</p>
<nav>{''.join(f'<a href=#e{i}>{i}</a>' for i in range(1, 8))}{'<a href=#e8>appendix</a>' if appendix else ''}</nav>"""
foot = """<section><h2>Reproduce</h2><pre>cd botanical-computer-vision/meeting_minimal
bash fetch_images.sh          # 1,641 originals + thumbnails -> /home/user/imgcache (not committed)
bash run_all.sh               # exp1–exp7 on EfficientNet
bash run_models.sh            # vector-only items on BioCLIP / DINOv3, then rebuild both reports</pre>
<p class=note>Dependencies: python 3, numpy, pandas, scikit-learn 1.9, scipy, matplotlib, torch/torchvision 2.x (CPU). All seeds 42. Per-item outputs in results/.</p></section></main></body></html>"""
(H / "index_en.html").write_text(head + "".join(secs) + appendix + foot)
print("wrote", H / "index_en.html")
