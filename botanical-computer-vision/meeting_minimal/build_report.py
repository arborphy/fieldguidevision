"""Build meeting_minimal/index.html from results/*. Images are loaded from bioimages.vanderbilt.edu (never committed)."""
import json
from html import escape
from pathlib import Path

import pandas as pd

import crossmodel
import ksoft

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
        f"<figure><a href='{url[r['image_id']].replace('/tn/', '/gq/').replace('/t', '/g', 1) if False else url[r['image_id']]}' target=_blank>"
        f"<img loading=lazy src='{url[r['image_id']]}' alt='{escape(r['image_id'])}'></a>"
        f"<figcaption>{escape(r['image_id'])}<br>{escape(r['species'])}<br>{escape(cap(r))}</figcaption></figure>" for r in recs) + "</div>"


BADGE = {"v": "<span class='b v'>经验证结论</span>", "e": "<span class='b e'>探索演示</span>",
         "x": "<span class='b x'>未完成 / 受阻</span>", "r": "<span class='b r'>恢复的旧结果（未重算）</span>"}


def section(n, title, ask, did, result, means, badges):
    return (f"<section id=e{n}><h2>{n}. {title}</h2><div>{' '.join(BADGE[b] for b in badges)}</div>"
            f"<dl><dt>问什么</dt><dd>{ask}</dd><dt>实际做了什么</dt><dd>{did}</dd>"
            f"<dt>结果</dt><dd>{result}</dd><dt>能说明什么</dt><dd>{means}</dd></dl></section>")


secs = []
# 1 NSC
o = s1["results"]["organ (external label)"]; g = [v for k, v in s1["results"].items() if k.startswith("grouping")][0]
nsc_tab = pd.DataFrame([dict(task=t, n_train=r["n_train"], n_test=r["n_test"], classes=r["n_classes"],
                             delta_1se=r["one_se_delta"], coords_1se=r["test"]["delta_1se"]["coords_any_class"],
                             coords_per_class_1se=r["test"]["delta_1se"]["coords_mean_per_class"],
                             test_F1_delta0=r["test"]["delta0"]["test_macroF1"], test_F1_1se=r["test"]["delta_1se"]["test_macroF1"],
                             ratio=r["ratio_F1_1se_over_delta0"]) for t, r in s1["results"].items()])
secs.append(section(1, "NSC：性能—保留坐标数曲线", "用很少的 PCA 坐标能否保持器官区分 / 复现现有聚类分组？",
    "EfficientNet-B0 原始向量 → PCA-64（仅 train 拟合，白化）。自写 NSC（Tibshirani 2002）；Δ 网格 33 点；train 上按个体 GroupKFold 5 折选 Δ（1-SE 规则），test 只评估一次。"
    "任务 A：外部器官标签（BioImages organ_category，7 类）。任务 B：<b>分组近似任务</b>——目标是已有 EfficientNet K=20 cluster ID，与输入同源，具循环性。",
    table(nsc_tab) + "<img src='results/exp1_nsc_curve.png' alt='NSC curve'>" + crossmodel.nsc_details("zh", strip, table),
    f"器官：1-SE Δ={o['one_se_delta']} 时只剩 {o['test']['delta_1se']['coords_any_class']}/64 个坐标（每类平均 {o['test']['delta_1se']['coords_mean_per_class']:.1f} 个），"
    f"test macro-F1 {o['test']['delta_1se']['test_macroF1']:.3f}，Δ=0 为 {o['test']['delta0']['test_macroF1']:.3f}（保留 {o['ratio_F1_1se_over_delta0']:.0%}）。"
    f"而分组近似任务几乎不能稀疏化（{g['test']['delta_1se']['coords_any_class']}/64）——K-means 分组是分散在多数主成分上的。"
    "<b>这些坐标是 PCA 主成分，不是生物学特征</b>；单次 split、单个模型；未做跨折稳定性（Jaccard）。", ["v", "e"]))

# 2 PCA + enrichment
sm = pd.DataFrame(s2["summary"]).T.reset_index().rename(columns={"index": "representation/label"})
top = pd.read_csv(OUT / "exp2_enrichment_top_per_cluster.csv")
ex = top[(top.label_type == "organ")].sort_values("species_adjusted_fe", ascending=False).groupby("representation").head(3)
secs.append(section(2, "PCA 保留信息 ＋ 已有 cluster 的组成与富集", "降维丢了多少？已有 K=20 聚类各自是什么器官 / 尺度 / 物种组成？",
    "PCA 保留量见第 6 项表（EfficientNet，train 拟合 / val 报告）。富集：BioCLIP、DINOv3、EfficientNet 已有的 K=20 cluster ID（全 1,641 张）× 器官 / 拍摄尺度。"
    "报告 count、基线、fold enrichment（FE）、<b>按个体植株</b>的 bootstrap 95% CI（500 次）、物种校正 FE（观测 / 按该簇物种构成的期望），以及最大物种占比。"
    "<b>不报告 p 值</b>：图片嵌套在 465 个个体与 63 个物种内，Fisher/卡方的独立性假设不成立。",
    f"PCA（EfficientNet，val）：2 维保留方差 {s6['table'][0]['train_explained_var']:.1%}，64 维 {s6['table'][4]['train_explained_var']:.1%}，256 维 {s6['table'][6]['train_explained_var']:.1%}。"
    + table(sm) + "<p>各模型物种校正 FE 最高的 3 个器官簇：</p>" +
    table(ex, ["representation", "cluster", "label", "n_images", "cluster_share", "fold_enrichment", "fe_ci_lo", "fe_ci_hi",
               "species_adjusted_fe", "n_individuals", "n_species", "top_species_share"]) +
    "<img src='results/exp2_organ_composition.png' alt='organ composition'>",
    "描述性效应：DINOv3 / EfficientNet 有 19–20/20 个簇的主导器官 FE 下界 &gt;1.5，且按物种构成校正后仍 &gt;1.5，单一物种在簇内占比中位数仅 9–12%——器官簇不是“某个物种的簇”。"
    "BioCLIP 只有 4–5/20 个簇如此，最大物种占比中位数 33%（物种簇）。此结论与 README §2 的 NMI 一致，现在额外通过了个体重采样与物种校正。标签为单标签 organ_category，尺度标签为暂定。",
    ["v"]))

# 3 activation
cases = pd.read_csv(OUT / "exp3_activation_cases.csv")
secs.append(section(3, "Activation mapping（Grad-CAM ＋ 遮挡对照）", "EfficientNet 把图片推向某个簇时，依赖的是图片的哪个区域？热图是否忠实于分数？",
    f"EfficientNet-B0 权重 + 1,641 张原图均可得，因此做了真实计算（CPU fp32）。先复核：重新嵌入 40 张图与库存向量余弦中位数 {s3['reembed_cosine_median']:.4f}（最低 {s3['reembed_cosine_min']:.4f}）。"
    "目标分数 = 图像嵌入与 K=20 簇中心的余弦。每簇按到中心距离抽 2 个核心 + 2 个边界（seed 42，非手选，共 80 张）。"
    "遮挡对照：把 Grad-CAM 最高的 12/49 格换成均值灰，对比 10 次随机 12 格。合理性检查：重置最后一个特征块后热图应改变。"
    "<b>BioCLIP / DINOv3 未做</b>：需要加载它们的模型权重（原始向量后来已恢复，见附录）。",
    f"Grad-CAM 遮挡导致的分数下降大于随机遮挡的比例：{s3['frac_top_cam_beats_random']:.0%}（核心 {s3['frac_by_kind']['core']:.0%}，边界 {s3['frac_by_kind']['boundary']:.0%}）；"
    f"下降中位数 {s3['median_drop_top']:.3f}，随机为 {s3['median_drop_random']:.3f}。最后一块随机化后与原热图的 Spearman 中位数为 {s3['sanity_spearman_vs_randomised_last_block_median']:.2f}（热图依赖权重，检查通过）。"
    "<img src='results/exp3_gradcam_cases.png' alt='Grad-CAM cases'><p>上图是固定规则选出的 8 例（4 个最大簇各取核心、边界各 1 张）。例如 c0 核心一例中，热点在图片上方的背景，而不在花序上。</p>",
    "热图对目标分数是<b>忠实的</b>（遮挡对照）。但<b>它是否落在器官上尚未验证</b>：没有器官掩膜，所以不能得出“簇看的是叶片而非背景”之类的结论。可视化的个例仅为探索演示。",
    ["v", "e"]))

# 4 K choice + soft clustering (redone)
secs.append(ksoft.section4("zh", section, table, strip))

# 5 AE / axes
rec = pd.read_csv(OUT / "exp5_recovered_ae_table.csv")
axt = pd.read_csv(OUT / "exp5_axis_table.csv")
la = s5["linear_ae_sanity"]
ends_html = "".join(f"<h4>PC{k} 低端</h4>{strip(v['low'])}<h4>PC{k} 高端</h4>{strip(v['high'])}" for k, v in list(s5["ends"].items())[:3])
secs.append(section(5, "AE 恢复 ＋ PCA 轴两端的真实图片", "非线性 AE 是否比 PCA 好？主要 PCA 方向对应什么？",
    "从已发布网站 scale/meeting/ 的 Experiment 6 表中恢复了 AE 与 PCA 的对比（原始 HTML 存于 recovered_site/）。只加一个检查：线性 AE（32 维，3,000 epoch）应与 PCA-32 一致。"
    "轴解读：EfficientNet PCA（train 拟合）PC1–5 与尺度（distant=0 → close=2）、各器官 one-vs-rest、缩略图亮度的 Spearman；每端 6 张真实图片。不做架构搜索。",
    table(rec, d=6) + f"<p>线性 AE 检查：PCA-32 test MSE {la['pca32_test_mse']:.7f}，线性 AE {la['linear_ae32_test_mse']:.7f}（比值 {la['ratio']:.3f}），"
    "并且本次 PCA-32 的数值与网站上的 EfficientNet PCA 0.000296 一致——训练代码与数据对齐。</p>" +
    table(axt.drop(columns=["reading"]), d=2) + ends_html,
    "AE：恢复的结果显示 AE 的 MSE 与 PCA 基本相同，而邻居保留更差（3 个模型都是如此）→ 没有值得增加复杂度的非线性结构，这条线到此为止（恢复结果 + 线性 sanity 已验证）。"
    "轴：PC1 与尺度 ρ=−0.65、与 whole tree ρ=+0.62（近景叶片 ↔ 远景整树）；PC2 ≈ 树皮 ↔ 整树；PC3 ≈ 花序 / 果实 ↔ 叶片；PC4 与所列标签的相关均 |ρ|&lt;0.1，<b>没有可读含义</b>。轴的命名是对相关性的描述，并非因果。",
    ["r", "v", "e"]))

# 6 fidelity
t6 = pd.DataFrame(s6["table"])
secs.append(section(6, "可视化保真度", "二维 / 较高维 PCA 图在多大程度上保留了原空间的邻居结构？",
    "EfficientNet，PCA 仅在 train（1,181）上拟合，在 val（249）上评估：val 池内的 15-NN 重叠、trustworthiness、重建 R²、线性探针（C=100，与现有基准一致）、K=20 ARI。另在全 1,641 张上计算 2D 的数值，以便与网站对照。",
    table(t6) + "<img src='results/exp6_pca_fidelity.png' alt='fidelity'>",
    f"2D 只保留约 {s6['two_d_all_images']['knn15_overlap_all1641']:.0%} 的 15 近邻（全 1,641 张；val 池为 33%），K=20 ARI 0.23 → <b>不能从二维图读结构</b>。"
    "按预设规则（15-NN ≥ 0.85 且物种探针差距在 1pp 内），在 ≤256 维中<b>找不到 d*</b>（256 维为 0.850 / 0.849，在边界上）。器官探针在 64 维左右已饱和。"
    "（物种探针在 val 上本来就很低：EfficientNet 不是物种模型。）", ["v"]))

# 7 retrieval
r7 = pd.read_csv(OUT / "exp7_retrieval.csv")
cs = s7["cases"]
pick = ["centroid c14", "centroid c5", "centroid c0", "centroid c13", "PC1 -2SD", "PC1 +2SD"]
html7 = "".join(f"<h4>{p}</h4>{strip(cs[p], cap=lambda r: f'{r['organ']} · {r['shot_scale']} · cos {r['cos']:.2f}')}" for p in pick)
secs.append(section(7, "反演：检索最近的真实图片", "某个簇中心 / 某个轴方向上的点“长什么样”？",
    "不训练生成模型。在完整的 1,280 维 EfficientNet 空间中，用余弦相似度检索离以下潜在点最近的 6 张真实图片：20 个 EfficientNet K=20 簇中心，以及 PC1 / PC2 的 均值 ± 2SD 点（重新归一化）。",
    table(r7[r7.point.str.startswith("centroid")], ["point", "cluster_size", "cluster_dominant_organ", "cluster_dominant_share", "top1_cos",
                                                    "top6_in_cluster", "top6_share_dominant_organ", "top6_n_species"], d=2) + html7,
    f"簇中心的近邻平均有 {s7['mean_top6_in_cluster']:.1f}/6 张属于该簇，主导器官占比 {s7['mean_top6_share_dominant_organ']:.0%}，高于簇整体的 {s7['mean_cluster_dominant_share']:.0%}："
    "检索出的是该簇的“典型”图片（会夸大簇的纯度，展示时需注意）。PC1 −2SD → 近景叶片，+2SD → 远景整树，与第 5 项一致。这是检索，不是图像生成。",
    ["v", "e"]))

cm = crossmodel.rows()
appendix = ""
if len(cm) > 1:
    axes_html = "".join(f"<h4>{n}：PC1–3</h4>" + table(crossmodel.axes(n).head(3)[["pc", "var", "rho_scale_close", "rho_brightness", "reading"]], d=2)
                        for n in crossmodel.DIRS if (crossmodel.DIRS[n] / "exp5_axis_table.csv").exists())
    appendix = (f"<section id=e8><h2>附：同样的纯向量实验在 BioCLIP 2.5 和 DINOv3 上的结果</h2><div>{BADGE['v']}</div>"
                "<p>主实验跑完后，从 Drive 恢复了 BioCLIP 2.5（1,641 × 1,024）和 DINOv3（1,641 × 768）的原始向量，记录了 SHA-256，并核对了 image_id 顺序和 split 与 manifest 一致。"
                "第 1、4、5、6、7 项用 <code>FGV_MODEL=bioclip|dinov3</code> 原样重跑，结果在 results/bioclip/ 和 results/dinov3/。"
                "第 3 项仍需要 BioCLIP / DINOv3 的模型权重。NSC 的分组任务和检索使用各模型自己已有的 K=20 簇。熵对错误的 AUROC 只有 EfficientNet 有，因为其他模型的逐图预测不在 Git 中。</p>"
                + table(cm) + axes_html + "<p>" + crossmodel.NOTE_ZH + "</p></section>")

css = """
:root{--bg:#fbfaf7;--fg:#1d1f1c;--mut:#5c6159;--line:#dcdcd4;--card:#fff;--v:#1f6f43;--e:#8a5a00;--x:#9b2c2c;--r:#3b5b92}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#151714;--fg:#e8e9e4;--mut:#a3a89f;--line:#33362f;--card:#1d201b;--v:#6fcf97;--e:#f2c94c;--x:#f28b82;--r:#8ab4f8}}
body{background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,"PingFang SC","Noto Sans CJK SC",sans-serif;margin:0;padding:0 16px}
main{max-width:1100px;margin:auto;padding:24px 0 80px}h1{font-size:24px;margin:.2em 0}h2{font-size:19px;margin:0 0 6px}h4{margin:12px 0 4px;font-size:13px;color:var(--mut)}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:18px 0}
dt{font-weight:600;margin-top:10px}dd{margin:2px 0 0}img{max-width:100%;height:auto;display:block;margin:8px 0}
.b{display:inline-block;font-size:12px;padding:1px 8px;border-radius:99px;border:1px solid currentColor;margin-right:4px}.v{color:var(--v)}.e{color:var(--e)}.x{color:var(--x)}.r{color:var(--r)}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:12px;margin:6px 0}td,th{border-bottom:1px solid var(--line);padding:3px 7px;text-align:left;white-space:nowrap}
.strip{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px}figure{margin:0;flex:0 0 120px}figure img{width:120px;height:90px;object-fit:cover;margin:0;border-radius:4px}
details{margin:8px 0}summary{cursor:pointer;font-weight:600}.ends{display:grid;grid-template-columns:1fr 1fr;gap:8px}.ends>div{min-width:0}@media(max-width:700px){.ends{grid-template-columns:1fr}}figcaption{font-size:10.5px;color:var(--mut);line-height:1.3}nav a{margin-right:10px;color:var(--fg)}.note{color:var(--mut);font-size:13px}
"""
head = f"""<!doctype html><html lang=zh><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Minimal Meeting Experiments</title><style>{css}</style></head><body><main>
<h1>七个方向的最小实验 · 2026-10-30</h1>
<p class=note>主数据：EfficientNet-B0 原始向量（Git，SHA-256 已校验，1,641 个 image_id 与 individual_disjoint.csv 的划分逐一核对一致）。
第 1–7 项在 EfficientNet 上运行；BioCLIP / DINOv3 在第 2 项中通过已有的 cluster ID 参与，文末附录用恢复的原始向量重跑了纯向量实验。
图片按 URL 从 bioimages.vanderbilt.edu 加载。标签：器官 = BioImages organ_category（单标签）；尺度 = BioImages + Gemma（暂定）。</p>
<p>{' '.join(BADGE.values())}</p>
<p class=note><b>经验证</b> = 有预先固定的协议、held-out 或重采样对照、可从脚本复现的数值。<b>探索演示</b> = 个例图片、未经独立标签验证的解读。</p>
<nav>{''.join(f'<a href=#e{i}>{i}</a>' for i in range(1, 8))}{'<a href=#e8>附：三模型</a>' if appendix else ''}</nav>"""
foot = """<section><h2>复现</h2><pre>cd botanical-computer-vision/meeting_minimal
bash fetch_images.sh          # 下载 1,641 张原图 + 缩略图到 /home/user/imgcache（不进 Git）
bash run_all.sh               # 在 EfficientNet 上依次运行 exp1–exp7 并生成本页
bash run_models.sh            # 在 BioCLIP / DINOv3 上跑纯向量实验，并重新生成两份报告</pre>
<p class=note>依赖：python 3、numpy、pandas、scikit-learn 1.9、scipy、matplotlib、torch/torchvision 2.x（CPU）。所有随机种子为 42。逐项输出见 results/。</p></section></main></body></html>"""
(H / "index.html").write_text(head + "".join(secs) + appendix + foot)
print("wrote", H / "index.html")
