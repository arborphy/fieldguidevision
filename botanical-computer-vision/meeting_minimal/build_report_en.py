"""Build meeting_minimal/index_en.html (English version of index.html) from results/*.
Images load from bioimages.vanderbilt.edu (never committed)."""
import json
from html import escape
from pathlib import Path

import pandas as pd

import crossmodel
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
# 1 NSC
o = s1["results"]["organ (external label)"]
g = [v for k, v in s1["results"].items() if k.startswith("grouping")][0]
nsc_tab = pd.DataFrame([dict(task=t, n_train=r["n_train"], n_test=r["n_test"], classes=r["n_classes"],
                             delta_1se=r["one_se_delta"], coords_1se=r["test"]["delta_1se"]["coords_any_class"],
                             coords_per_class_1se=r["test"]["delta_1se"]["coords_mean_per_class"],
                             test_F1_delta0=r["test"]["delta0"]["test_macroF1"], test_F1_1se=r["test"]["delta_1se"]["test_macroF1"],
                             ratio=r["ratio_F1_1se_over_delta0"]) for t, r in s1["results"].items()])
secs.append(section(1, "NSC: performance vs number of retained coordinates",
    "Can a few PCA coordinates keep organ separation, or reproduce the existing cluster grouping?",
    "EfficientNet-B0 raw vectors → PCA-64 (fit on train only, whitened). Own NSC implementation (Tibshirani 2002); 33-point Δ grid; Δ chosen by "
    "5-fold GroupKFold on train (groups = individual plant) with the one-standard-error rule; test evaluated once. "
    "Task A: external organ label (BioImages organ_category, 7 classes). Task B: <b>grouping-approximation task</b>, where the target is the "
    "existing EfficientNet K=20 cluster ID. That target comes from the same vectors, so the task is circular.",
    table(nsc_tab) + "<img src='results/exp1_nsc_curve.png' alt='NSC curve'>" + crossmodel.nsc_details("en", strip, table),
    f"Organ: at the 1-SE Δ={o['one_se_delta']} only {o['test']['delta_1se']['coords_any_class']}/64 coordinates remain "
    f"({o['test']['delta_1se']['coords_mean_per_class']:.1f} per class on average). Test macro-F1 is {o['test']['delta_1se']['test_macroF1']:.3f} "
    f"vs {o['test']['delta0']['test_macroF1']:.3f} at Δ=0 ({o['ratio_F1_1se_over_delta0']:.0%} retained). "
    f"The grouping-approximation task barely sparsifies ({g['test']['delta_1se']['coords_any_class']}/64): K-means groups are spread over most components. "
    "<b>These coordinates are PCA components, not biological features.</b> Single split, single model; cross-fold stability (Jaccard) not done.",
    ["v", "e"]))

# 2 PCA + enrichment
sm = pd.DataFrame(s2["summary"]).T.reset_index().rename(columns={"index": "representation/label"})
top = pd.read_csv(OUT / "exp2_enrichment_top_per_cluster.csv")
ex = top[(top.label_type == "organ")].sort_values("species_adjusted_fe", ascending=False).groupby("representation").head(3)
secs.append(section(2, "Information kept by PCA + composition and enrichment of existing clusters",
    "How much does dimensionality reduction lose? What organ / scale / species mix does each existing K=20 cluster have?",
    "PCA retention: see the table in item 6 (EfficientNet, fit on train / reported on val). Enrichment: existing K=20 cluster IDs of BioCLIP, DINOv3 "
    "and EfficientNet (all 1,641 images) × organ / shot scale. Reported: count, base rate, fold enrichment (FE), 95% CI from a bootstrap over "
    "<b>individual plants</b> (500 resamples), species-adjusted FE (observed / expected from the cluster's species mix), and top-species share. "
    "<b>No p-values</b>: images are nested in 465 individuals and 63 species, so the independence assumption of Fisher / chi-square tests fails.",
    f"PCA (EfficientNet, val): 2-D keeps {s6['table'][0]['train_explained_var']:.1%} of variance, 64-D {s6['table'][4]['train_explained_var']:.1%}, "
    f"256-D {s6['table'][6]['train_explained_var']:.1%}." + table(sm) + "<p>Top 3 organ clusters per model by species-adjusted FE:</p>" +
    table(ex, ["representation", "cluster", "label", "n_images", "cluster_share", "fold_enrichment", "fe_ci_lo", "fe_ci_hi",
               "species_adjusted_fe", "n_individuals", "n_species", "top_species_share"]) +
    "<img src='results/exp2_organ_composition.png' alt='organ composition'>",
    "Descriptive effects. In DINOv3 / EfficientNet, 19–20 of 20 clusters have a dominant organ with FE CI lower bound &gt;1.5, still &gt;1.5 after "
    "adjusting for species mix, and the median top-species share is only 9–12%. These organ clusters are not single-species clusters. "
    "BioCLIP has 4–5 such clusters out of 20, with a median top-species share of 33%, so it clusters by species. This matches the NMI in README §2 "
    "and now also survives individual resampling and species adjustment. Labels: single-label organ_category; scale labels are provisional.",
    ["v"]))

# 3 activation
secs.append(section(3, "Activation mapping (Grad-CAM + occlusion control)",
    "When EfficientNet pushes an image towards a cluster, which image regions does it rely on? Is the map faithful to the score?",
    f"EfficientNet-B0 weights and all 1,641 original images are available, so this is a real computation (CPU, fp32). Check first: re-embedding 40 images "
    f"gives median cosine {s3['reembed_cosine_median']:.4f} to the stored vectors (min {s3['reembed_cosine_min']:.4f}). "
    "Target score = cosine between the image embedding and its K=20 cluster centroid. Per cluster, 2 core + 2 boundary images by distance to the centroid "
    "(seed 42, not hand-picked; 80 images). Occlusion control: replace the top 12/49 Grad-CAM cells with mean grey vs 10 random sets of 12 cells. "
    "Sanity check: re-initialising the last feature block should change the map. "
    "<b>BioCLIP / DINOv3 not run</b>: this needs their model weights and a GPU-scale run, which this round did not do (their raw vectors were recovered later, see appendix).",
    f"Fraction of cases where Grad-CAM occlusion drops the score more than random occlusion: {s3['frac_top_cam_beats_random']:.0%} "
    f"(core {s3['frac_by_kind']['core']:.0%}, boundary {s3['frac_by_kind']['boundary']:.0%}). Median drop {s3['median_drop_top']:.3f} vs random "
    f"{s3['median_drop_random']:.3f}. Spearman between the original map and the map after randomising the last block: median "
    f"{s3['sanity_spearman_vs_randomised_last_block_median']:.2f} (the map depends on the weights; check passed)."
    "<img src='results/exp3_gradcam_cases.png' alt='Grad-CAM cases'><p>The 8 cases above follow a fixed rule (first core and first boundary image "
    "of the 4 largest clusters). In the c0 core case the hot area is background at the top of the image, not the inflorescence.</p>",
    "The maps are <b>faithful</b> to the target score (occlusion control). Whether they fall <b>on the organ is not tested</b>: there are no organ masks, "
    "so no claim such as \"the cluster looks at the leaf, not the background\" is supported. Individual maps are exploratory.",
    ["v", "e"]))

# 4 soft
km = pd.read_csv(OUT / "exp4_existing_k_comparison.csv")
amb = pd.read_csv(OUT / "exp4_ambiguous_cases.csv").head(8)
recs = [dict(image_id=r.image_id, species=r.species, organ=r.organ, shot_scale=r.shot_scale,
             cap=f"{r.organ}·{r.shot_scale}<br>p1={r.gmm_max:.2f}({r.top1_component_main_organ}) p2={r.gmm_p2:.2f}({r.top2_component_main_organ})") for r in amb.itertuples()]
secs.append(section(4, "Soft clustering / K",
    "Which K? Do some images really sit between two clusters, and does that ambiguity mean anything?",
    "Reused the existing K=10/20/50 × 10 seeds results (species/organ NMI, between-seed ARI). New: diagonal GMM (K=20) on EfficientNet PCA-32, 5 seeds; "
    "per image the maximum membership and membership entropy; BIC over K∈{5,10,20,30,50}; whether entropy predicts probe errors on test, and mid-range scale.",
    table(km, d=3) +
    f"<p>GMM BIC: {', '.join(f'K={k}: {v:,.0f}' for k, v in s4['bic'].items())} (lowest at K=30; flattens between K=20 and 30). "
    f"Between-seed label ARI {s4['seed_label_ari_mean']:.2f}, entropy-rank Spearman {s4['seed_entropy_spearman_mean']:.2f}; ARI vs existing K-means "
    f"{s4['ari_gmm_vs_existing_kmeans20']:.2f}. Only {s4['frac_images_max_membership_lt_0_6']:.1%} of images have maximum membership &lt;0.6. "
    f"AUROC of entropy for probe errors: <b>{s4['auroc_entropy_vs_probe_error_test211']:.2f}</b> (211 test images, {s4['n_test_wrong']} wrong); "
    f"for mid-range scale: {s4['auroc_entropy_vs_midrange_scale']:.2f}.</p><p>The 8 highest-entropy real images (p1/p2 = probabilities of the top two "
    "components; in brackets, that component's dominant organ):</p>" + strip(recs, cap=lambda r: r["cap"]),
    "In 32-D the diagonal GMM is very hard: ambiguous images are rare. Entropy does not predict classification errors (0.51 ≈ chance) and carries a weak "
    "signal for mid-range scale (0.64). <b>Soft probability is not multi-organ evidence.</b> Multi-organ AUROC cannot be computed: no multi-label "
    "organ labels are in Git (Gemma multi-label / Human Gold not exported). The example images are exploratory.",
    ["v", "e", "x"]))

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
r7 = pd.read_csv(OUT / "exp7_retrieval.csv")
cs = s7["cases"]
pick = ["centroid c14", "centroid c5", "centroid c0", "centroid c13", "PC1 -2SD", "PC1 +2SD"]
html7 = "".join(f"<h4>{p}</h4>{strip(cs[p], cap=lambda r: f'{r['organ']} · {r['shot_scale']} · cos {r['cos']:.2f}')}" for p in pick)
secs.append(section(7, "Inversion: retrieve the nearest real images",
    "What does a cluster centroid, or a point along an axis, look like?",
    "No generative model is trained. In the full embedding space, retrieve the 6 nearest real images (cosine) to: the 20 EfficientNet K=20 centroids, "
    "and the mean ± 2 SD points along PC1 / PC2 (re-normalised).",
    table(r7[r7.point.str.startswith("centroid")], ["point", "cluster_size", "cluster_dominant_organ", "cluster_dominant_share", "top1_cos",
                                                    "top6_in_cluster", "top6_share_dominant_organ", "top6_n_species"], d=2) + html7,
    f"On average {s7['mean_top6_in_cluster']:.1f}/6 neighbours of a centroid belong to that cluster, and {s7['mean_top6_share_dominant_organ']:.0%} show "
    f"its dominant organ vs {s7['mean_cluster_dominant_share']:.0%} across the whole cluster. Retrieval returns the cluster's typical members, which "
    "overstates cluster purity when shown. PC1 −2SD → close-up leaves, +2SD → distant whole trees, consistent with item 5. This is retrieval, not image generation.",
    ["v", "e"]))

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
