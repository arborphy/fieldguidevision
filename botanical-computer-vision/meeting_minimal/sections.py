"""Report sections that depend on K (1, 2, 3, 4, 7), shared by the Chinese and English builders.
Every number shown is read from results/; K values come from common.K_LIST, examples from common.MAIN_K."""
import json
from pathlib import Path

import pandas as pd

import crossmodel
from common import K_LIST, MAIN_K

R = Path(__file__).resolve().parent / "results"
MODELS = {"EfficientNet-B0": R, "BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3",
          "YOLO11s backbone": R / "yolo11s_backbone", "YOLO11s neck": R / "yolo11s_neck"}
KOK = lambda task: task.startswith("organ") or int(task.split("K=")[1].split()[0]) in K_LIST
KWHY = {3: ("3 种拍摄距离", "3 shot scales"), 7: ("7 种器官", "7 organs"), 20: ("旧默认值", "old default"), 63: ("63 个物种", "63 species")}
CAT = {"same_organ": ("两个候选同器官", "same organ"), "both_plausible": ("两个器官都合理", "both organs plausible"),
       "leaning_wrong": ("偏向了错的器官", "leaning to the wrong organ"), "neither": ("两个都不对", "neither matches"),
       "mixed_group": ("涉及混合组", "involves a mixed group")}
CAT_EXPL = {
    "same_organ": ("两个候选组的主要器官相同，含混发生在同一器官内部（距离、物种或背景不同），与器官判断无关。",
                   "Both candidate groups have the same main organ; the ambiguity is inside one organ (scale, species, background)."),
    "both_plausible": ("照片落在自己器官的组里，但第二候选是另一种器官。可能画面里同时有两种器官，是“两个标签都合理”的候选。",
                       "The photo sits in its own organ's group, but the second candidate is another organ. A second organ may be visible."),
    "leaning_wrong": ("模型更倾向的组不是照片的器官，标签器官只是第二候选，属于“容易认错”的候选。",
                      "The model leans to a group of another organ; the photo's own organ is only the second candidate."),
    "neither": ("两个候选组都不是照片的器官，可能是特殊照片或标签需要复核。",
                "Neither candidate's main organ is the photo's organ: unusual photo, or a label to check."),
    "mixed_group": ("至少一个候选组本身没有主导器官（&lt;50%），不能当作器官问题解读。",
                    "At least one candidate group has no main organ (&lt;50%), so it cannot be read as an organ question.")}


def _j(d, n):
    p = d / n
    return json.loads(p.read_text()) if p.exists() else None


def _ms(n):
    return {m: s for m, d in MODELS.items() if (s := _j(d, n))}


def _t(lang, zh, en):
    return zh if lang == "zh" else en


def _det(title, body, open_=False):
    return f"<details{' open' if open_ else ''}><summary>{title}</summary>{body}</details>"


COVER = [(1, "exp1_nsc_summary.json"), (2, "exp2_enrichment_summary.json"), (4, "exp4_soft_summary.json"),
         (5, "exp5_axes_summary.json"), (6, "exp6_summary.json"), (7, "exp7_retrieval_summary.json")]


def header(lang, table):
    """Key conclusions + which model is covered in which section (computed from the result files)."""
    cov = []
    for m, d in MODELS.items():
        r = {"model": m}
        for n, f in COVER:
            r[str(n)] = "✓" if (d / f).exists() else "—"
        r["3"] = "✓ (K=20)" if m == "EfficientNet-B0" else "—"
        cov.append(r)
    cov = pd.DataFrame(cov)[["model", "1", "2", "3", "4", "5", "6", "7"]]
    dom = {m: next((r["share_clusters_dominant_ge_0_6"] for r in s["summary"] if r["K"] == MAIN_K and r["label"] == "organ"), None)
           for m, s in _ms("exp2_enrichment_summary.json").items()}
    dtxt = ", ".join(f"{m} {v:.0%}" for m, v in dom.items() if v is not None)
    e = _ms("exp4_soft_summary.json")["EfficientNet-B0"]["results"][f"K{MAIN_K}"]
    if lang == "zh":
        pts = [f"<b>分组按什么组织</b>：K={MAIN_K} 时由单一器官主导（≥60%）的组所占比例：{dtxt}。通用视觉模型（EfficientNet、DINOv3）按器官和拍摄距离分组；BioCLIP 按物种；"
               "YOLO11s 是 COCO 通用检测模型、未在植物上训练，器官信息明显更弱。",
               "<b>K 是选择</b>：K=3 对应拍摄距离，K=7 对应器官；NSC 中“要近 64 个坐标才能复现分组”是 K 大造成的（第 1 部分）。",
               f"<b>软聚类</b>：含混真实存在且可重复（EfficientNet K={MAIN_K}：{e['frac_max_lt_0_8']:.0%} 的照片）；按照片自己的标签分为“两个器官都合理 / 偏向错的器官 / 两个都不对 / 同器官内部”（第 4 部分）；不能预测认错物种。",
               "<b>二维图不可信</b>：只保留约 10–14% 的近邻关系（第 6 部分）。",
               "<b>尚未完成</b>：Grad-CAM 只有早期 K=20 的 EfficientNet 结果；YOLO 的 K 扫描、软聚类、主轴未跑；多器官标签未导出。"]
        return ("<section class=summary><h2>结论摘要</h2><ul>" + "".join(f"<li>{x}</li>" for x in pts) + "</ul>"
                "<p class=note>各部分覆盖的模型（✓ = 有结果）：</p>" + table(cov) +
                "<p class=note>K 统一取 3 / 7 / 20（拍摄距离数 / 器官数 / 旧默认），示例图用 K=7。器官标签 = BioImages organ_category（单标签）；距离标签为暂定。"
                "图片从 bioimages.vanderbilt.edu 加载（自包含版已内嵌）。</p></section>")
    pts = [f"<b>What the grouping follows</b>: share of groups dominated by one organ (≥60%) at K={MAIN_K}: {dtxt}. General vision models "
           "(EfficientNet, DINOv3) group by organ and shot scale; BioCLIP by species; YOLO11s (COCO detector, not trained on plants) carries clearly less organ information.",
           "<b>K is a choice</b>: K=3 matches shot scale, K=7 matches organ; needing almost all 64 NSC coordinates is a large-K effect (section 1).",
           f"<b>Soft clustering</b>: ambiguity is real and repeatable (EfficientNet K={MAIN_K}: {e['frac_max_lt_0_8']:.0%} of photos), sorted by the photo's own label into "
           "both plausible / leaning wrong / neither / same organ (section 4); it does not predict species errors.",
           "<b>2-D plots are unreliable</b>: they keep only about 10–14% of neighbour relations (section 6).",
           "<b>Not done yet</b>: Grad-CAM only as the earlier EfficientNet K=20 run; YOLO K scan, soft clustering and axes not run; multi-organ labels not exported."]
    return ("<section class=summary><h2>Key conclusions</h2><ul>" + "".join(f"<li>{x}</li>" for x in pts) + "</ul>"
            "<p class=note>Models covered per section (✓ = results):</p>" + table(cov) +
            "<p class=note>K = 3 / 7 / 20 everywhere (shot scales / organs / old default); examples at K=7. Organ label = BioImages organ_category "
            "(single label); scale labels provisional. Images load from bioimages.vanderbilt.edu (embedded in the standalone copy).</p></section>")


# ---------------------------------------------------------------- 1 NSC
def sec1(lang, section, table, strip):
    S = _ms("exp1_nsc_summary.json")
    rows = []
    for m, s in S.items():
        for task, r in s["results"].items():
            if not KOK(task):
                continue
            rows.append({"model": m, "task": task.replace("grouping approximation: ", "grouping ").replace(" cluster ID", ""),
                         "n_classes": r["n_classes"], "Δ (1-SE)": r["one_se_delta"],
                         _t(lang, "保留坐标", "coords kept"): r["test"]["delta_1se"]["coords_any_class"],
                         _t(lang, "每类平均", "per class"): r["test"]["delta_1se"]["coords_mean_per_class"],
                         "F1 Δ=0": r["test"]["delta0"]["test_macroF1"], "F1 1-SE": r["test"]["delta_1se"]["test_macroF1"],
                         _t(lang, "保留比例", "ratio"): r["ratio_F1_1se_over_delta0"]})
    t = pd.DataFrame(rows)
    e = t[t.model == "EfficientNet-B0"]
    org = e[e.task.str.startswith("organ")].iloc[0]
    kept = "; ".join(f"{m} " + ", ".join(f"K={tk.split('=')[1]}→{c}" for tk, c in zip(g.task, g.iloc[:, 4]))
                     for m, g in t[t.task.str.startswith("grouping")].groupby("model", sort=False))
    return section(1, _t(lang, "NSC：性能—保留坐标数曲线", "NSC: performance vs number of retained coordinates"),
        _t(lang, "用很少的 PCA 坐标，能否保持器官区分，或复现无监督分组？复现分组需要很多坐标，是 K=20 的特例吗？",
           "Can a few PCA coordinates keep organ separation or reproduce the unsupervised grouping? Is needing many coordinates a K=20 artefact?"),
        _t(lang, "每个模型的向量 → PCA-64（只在 train 上拟合，白化）。自写 NSC（Tibshirani 2002），Δ 网格 33 点，按个体植株分组的 5 折交叉验证 + 一倍标准误规则选 Δ，test 只评估一次。"
                 "任务 A：外部器官标签（7 类）。任务 B：<b>分组近似任务</b>，目标是该模型自己的 K-means 分组编号，K = 3 / 7 / 20。目标与输入同源，有循环性。NSC 是有监督方法，这里用作解释工具。",
                 "Each model's vectors → PCA-64 (fit on train only, whitened). Own NSC (Tibshirani 2002), 33-point Δ grid, Δ by 5-fold CV grouped by "
                 "individual plant + one-standard-error rule, test evaluated once. Task A: external organ label (7 classes). Task B: <b>grouping "
                 "approximation</b>, the model's own K-means IDs for K = 3 / 7 / 20 (circular: same vectors). NSC is supervised; used here to explain."),
        "<img src='results/exp1c_nsc_organ_models.png' alt='NSC organ, all models'>"
        + _t(lang, "<p>每个模型：器官任务（黑色虚线）与分组近似任务 K=3/7/20。圆点 = 按一倍标准误规则选出的 Δ；横轴向右 = 保留的坐标越少。</p>",
                   "<p>Per model: organ task (black dashed) and grouping task K=3/7/20. Dot = Δ chosen by the 1-SE rule; to the right = fewer coordinates kept.</p>")
        + "<img src='results/exp1c_nsc_grouping_models.png' alt='NSC grouping per model'>"
        + table(t.pivot_table(index="model", columns="task", values=t.columns[4], sort=False).reset_index(), d=0)
        + _t(lang, "<p class=note>上表：按一倍标准误规则选出 Δ 后保留的坐标数（共 64，越少越精简）。</p>",
                   "<p class=note>Above: coordinates kept at the 1-SE Δ (of 64; fewer = sparser).</p>")
        + _det(_t(lang, "完整数值表（F1、Δ、每类平均坐标）", "Full table (F1, Δ, coordinates per class)"), table(t, d=3))
        + crossmodel.nsc_details(lang, strip, table),
        _t(lang, f"器官：EfficientNet 用 {org.iloc[4]}/64 个坐标保留 {org.iloc[-1]:.0%} 的 F1；BioCLIP、DINOv3 和两个 YOLO 版本都压不下来（64/64）。"
                 f"分组近似任务保留的坐标数（共 64）：{kept}。<b>“要近 64 个坐标才能复现分组”是 K 大造成的</b>：组越少，需要的坐标越少；EfficientNet 在 K=7 时只要约 20 个，和器官标签任务相当；K=20 时各模型都要用到几乎全部坐标。"
                 "<b>坐标是 PCA 主成分，不是生物学特征。</b>",
                 f"Organ: EfficientNet keeps {org.iloc[-1]:.0%} of F1 with {org.iloc[4]}/64 coordinates; BioCLIP, DINOv3 and both YOLO versions do not sparsify (64/64). "
                 f"Coordinates kept for the grouping task (of 64): {kept}. <b>Needing almost all 64 is a large-K effect</b>: fewer groups need fewer "
                 "coordinates; EfficientNet at K=7 needs about 20, close to the organ-label task; at K=20 the models need nearly all of them. "
                 "<b>Coordinates are PCA components, not biological features.</b>"),
        ["v", "e"])


# ---------------------------------------------------------------- 2 enrichment
def sec2(lang, section, table, strip, s6):
    S = _ms("exp2_enrichment_summary.json")
    sm = pd.DataFrame([r for s in S.values() for r in s["summary"] if r["K"] in K_LIST])
    sm = sm.rename(columns={"share_clusters_dominant_ge_0_6": _t(lang, "主导器官≥60%的组", "groups ≥60% one label"),
                            "share_clusters_fe_ci_lo_gt_1_5": _t(lang, "FE 下界>1.5", "FE CI-lo >1.5"),
                            "share_clusters_species_adj_fe_gt_1_5": _t(lang, "物种校正 FE>1.5", "species-adj FE >1.5"),
                            "median_top_species_share": _t(lang, "最大物种占比中位数", "median top-species share")})
    desc = "".join(f"<h4>{m} · K={MAIN_K}</h4>" + table(pd.DataFrame(s["clusters_main_k"])[["cluster", "n", "name", "organs", "scales", "n_species"]])
                   for m, s in S.items())
    figs = "".join(f"<img src='{p.relative_to(R.parent)}' alt='composition'>" for d in MODELS.values()
                   if (p := d / f"exp2_organ_composition_k{MAIN_K}.png").exists())
    return section(2, _t(lang, "PCA 保留信息 ＋ 各组的组成与富集", "Information kept by PCA + composition and enrichment of the groups"),
        _t(lang, "降维丢了多少？不同 K 下，各组分别是什么器官 / 距离 / 物种？", "How much does reduction lose? What organ / scale / species is in each group, at each K?"),
        _t(lang, f"PCA 保留量见第 6 项。富集：每个模型自己的 K-means 分组（K = 3 / 7 / 20，全部 1,641 张）× 器官 / 拍摄距离。报告组内占比、基线、富集倍数（FE）、"
                 "<b>按个体植株</b>重采样 200 次的 95% 区间、按物种构成校正后的 FE、最大物种占比。<b>不报 p 值</b>：照片嵌套在 465 个个体和 63 个物种里，独立性假设不成立。",
                 "PCA retention: item 6. Enrichment: each model's own K-means groups (K = 3 / 7 / 20, all 1,641) × organ / shot scale. Share, base rate, "
                 "fold enrichment (FE), 95% CI from 200 resamples of <b>individual plants</b>, species-adjusted FE, top-species share. <b>No p-values</b>: "
                 "images are nested in 465 individuals and 63 species, so independence fails."),
        _t(lang, f"PCA（EfficientNet）：2 维保留方差 {s6['table'][0]['train_explained_var']:.1%}，64 维 {s6['table'][4]['train_explained_var']:.1%}。",
                 f"PCA (EfficientNet): 2-D keeps {s6['table'][0]['train_explained_var']:.1%} of variance, 64-D {s6['table'][4]['train_explained_var']:.1%}.")
        + table(sm[sm.label == "organ"].pivot_table(index="model", columns="K", values=sm.columns[4], sort=False).reset_index(), d=2)
        + _t(lang, "<p class=note>上表：由单一器官主导（≥60%）的组所占比例。</p>", "<p class=note>Above: share of groups dominated by one organ (≥60%).</p>")
        + _det(_t(lang, "完整表（器官 / 距离 × K，富集与物种校正）", "Full table (organ / scale × K, enrichment, species-adjusted)"),
               table(sm.drop(columns=["clusters_lt5_individuals"]), d=2))
        + _t(lang, "<p class=note>组名规则：≥80% 直接用器官名；50–80% 写“大多是…”并列出第二成分；&lt;50% 写“混合”；距离同理。</p>",
                   "<p class=note>Group naming: ≥80% organ name; 50–80% \"mostly …\" with the second part; &lt;50% \"mixed\"; same for scale.</p>")
        + _det(_t(lang, f"K={MAIN_K} 每组的组成（各模型）", f"Group make-up at K={MAIN_K} (each model)"), desc + figs),
        _t(lang, "描述性结果，按 K 读上表：DINOv3 / EfficientNet 的组在每个 K 下都以单一器官为主，且按物种校正后仍然富集；BioCLIP 的组以物种为主。"
                 "K 越大，同一器官会被拆成多个组（例如多个“近拍叶子”组）。标签：器官为单标签 organ_category，距离标签为暂定。",
                 "Descriptive; read the table per K. DINOv3 / EfficientNet groups are dominated by one organ at every K and stay enriched after species "
                 "adjustment; BioCLIP groups follow species. Larger K splits one organ into several groups (e.g. several close-up leaf groups). "
                 "Labels: single-label organ_category; scale labels provisional."),
        ["v"])


# ---------------------------------------------------------------- 3 Grad-CAM
def sec3(lang, section, table, s3):
    pk = (pd.DataFrame([r for r in s3["per_k"] if r["K"] in K_LIST]) if "per_k" in s3 else  # per-K rerun pending: earlier K=20 run
          pd.DataFrame([dict(K=20, n=s3["n_cases"], frac_top_beats_random=s3["frac_top_cam_beats_random"],
                             median_drop_top=s3["median_drop_top"], median_drop_random=s3["median_drop_random"])]))
    return section(3, _t(lang, "Activation mapping（Grad-CAM ＋ 遮挡对照）· 早期结果，未重跑", "Activation mapping (Grad-CAM + occlusion) · earlier run, not redone"),
        _t(lang, "EfficientNet 把照片推向某一组时依赖哪块区域？热图是否忠实于分数？", "Which image regions push EfficientNet towards a group? Is the map faithful?"),
        _t(lang, f"EfficientNet-B0 真实计算（CPU fp32）。先复核：重新嵌入 40 张图，与库存向量余弦中位数 {s3['reembed_cosine_median']:.4f}。目标分数 = 嵌入与组中心的余弦。"
                 "每个 K（3 / 7 / 20）每组取 2 个核心 + 2 个边界样本（seed 42，非手选）。遮挡：Grad-CAM 最高的 12/49 格 vs 10 次随机 12 格。合理性检查：重置最后一块后热图应改变。"
                 "BioCLIP / DINOv3 需要加载模型权重，本轮未做。",
                 f"Real EfficientNet-B0 computation (CPU fp32). Check: re-embedding 40 images gives median cosine {s3['reembed_cosine_median']:.4f} to stored vectors. "
                 "Target = cosine to the group centre. For each K (3 / 7 / 20), 2 core + 2 boundary images per group (seed 42, not hand-picked). "
                 "Occlusion: top 12/49 Grad-CAM cells vs 10 random sets. Sanity: re-initialising the last block must change the map. BioCLIP / DINOv3 need their weights; not run."),
        table(pk, d=3) + "<img src='results/exp3_gradcam_cases.png' alt='Grad-CAM'>"
        + _t(lang, f"<p>上图：K={MAIN_K} 下 4 个最大组各取核心、边界 1 张。热图越红表示模型越依赖该区域，蓝色表示几乎没用到。</p>",
                   f"<p>Above: K={MAIN_K}, one core and one boundary image of the 4 largest groups. Red = relied on most, blue = barely used.</p>"),
        _t(lang, f"各个 K 下热图都忠实于分数（遮挡热点比随机遮挡掉分更多的比例见表）；随机化最后一块后热图明显改变（Spearman {s3['sanity_spearman_vs_randomised_last_block_median']:.2f}），检查通过。"
                 "<b>热图是否落在植物上没有验证</b>（没有器官掩膜），个例只作演示。",
                 f"At every K the maps are faithful to the score (table: share where hiding the hot area hurts more than random). Randomising the last "
                 f"block changes the map (Spearman {s3['sanity_spearman_vs_randomised_last_block_median']:.2f}), check passed. <b>Whether maps land on the plant is "
                 "not tested</b> (no organ masks); individual maps are illustrations."),
        ["v", "e"])


# ---------------------------------------------------------------- 4 K + soft clustering
def sec4(lang, section, table, strip):
    zh = lang == "zh"
    ks = {m: pd.read_csv(d / "exp4b_k_sweep.csv") for m, d in MODELS.items() if (d / "exp4b_k_sweep.csv").exists()}
    k = pd.concat([v.assign(model=m) for m, v in ks.items()])
    k = k[k.K.isin(K_LIST)]
    match = k.apply(lambda r: {3: r.matched_acc_scale, 7: r.matched_acc_organ, 63: r.matched_acc_species}.get(r.K), axis=1)
    kt = pd.DataFrame({"model": k.model, "K": k.K, _t(lang, "K 的含义", "why"): k.K.map(lambda x: KWHY[x][0 if zh else 1]),
                       "AMI organ": k.ami_organ, "AMI scale": k.ami_scale, "AMI species": k.ami_species,
                       _t(lang, "一一对应准确率", "matched acc."): match.map(lambda v: "—" if pd.isna(v) else f"{v:.2f}"),
                       _t(lang, "稳定性 ARI", "stability ARI"): k.stability_ari_mean})
    S = _ms("exp4_soft_summary.json")
    rows = []
    for m, s in S.items():
        for kk, v in s["results"].items():
            if int(kk[1:]) not in K_LIST:
                continue
            r = {"model": m, "K": kk[1:], _t(lang, "含混比例", "ambiguous"): v["frac_max_lt_0_8"],
                 _t(lang, "可重复性 ρ", "split-half ρ"): v["entropy_split_half_spearman"], "AUROC error": v["auroc_entropy_species_probe_error_test"]}
            r.update({CAT[c][0 if zh else 1]: v["category_share"][c] for c in CAT})
            rows.append(r)
    st = pd.DataFrame(rows)
    E = S["EfficientNet-B0"]
    main, art = E["main"], pd.DataFrame(E["gmm_dimension_artifact"])
    name = {c["cluster"]: c["name"] for c in main["clusters"]}
    cap = lambda r: (f"{_t(lang, '标签', 'label')}: {r['organ']} · {r['shot_scale']}<br>c{r['c1']} {r['p1']:.2f} / c{r['c2']} {r['p2']:.2f}")
    faces = "".join(f"<h4>c{c} · {name[int(c)]}</h4>" + strip(v, cap=lambda r: f"{r['organ']} · {r['shot_scale']}")
                    for c, v in main["faces"].items())
    cats = "".join(f"<h4>{CAT[c][0 if zh else 1]} — {CAT_EXPL[c][0 if zh else 1]}</h4>"
                   + (strip(v, cap=cap) if v else _t(lang, "<p>无</p>", "<p>none</p>")) for c, v in main["by_category"].items())
    pairs = "".join(f"<h4>c{p['pair'][0]} ({name[p['pair'][0]]}) ↔ c{p['pair'][1]} ({name[p['pair'][1]]}) · n={p['n']}</h4>"
                    + strip(p["cases"], cap=cap) for p in main["by_pair"])
    e7 = E["results"][f"K{MAIN_K}"]
    sh = e7["category_share"]
    return section(4, _t(lang, "K 的选择 ＋ 软聚类", "Choice of K + soft clustering"),
        _t(lang, "无监督聚类中 K 是一个选择：不同 K 各对应什么？哪些照片夹在两组之间：两个标签都合理、容易认错，还是只是同一器官内部的差别？",
           "In unsupervised clustering K is a choice: what does each K correspond to? Which photos sit between two groups: two plausible labels, a likely mistake, or just variation within one organ?"),
        _t(lang, "<b>K</b>：取 3（拍摄距离数）、7（器官数）、20（旧默认），大 K 不跑，在完整向量上跑 K-means，看与器官 / 距离 / 物种的一致程度（AMI，已扣除随 K 增大的虚高）、"
                 "K 等于标签数时的一一对应准确率，以及按个体植株重采样的稳定性。"
                 "<br><b>软归属</b>：按个体植株重采样 50 次、每次重新聚类，照片落入各组的比例就是它的软归属；最大归属 &lt;0.8 记为含混。"
                 "之前用的对角高斯混合模型维度越高越“硬”（同一数据 2 维 22% 含混、32 维 1.3%，见下表），那个结论已作废。"
                 "<br><b>标签先理清</b>：先描述每一组的器官和距离构成并起不重名的名字，再用照片<b>自己的</b>器官标签，把含混照片分成 5 类（见下方各类说明）。",
                 "<b>K</b>: 3 (shot scales), 7 (organs), 20 (old default); large K left out. K-means on full vectors; agreement with organ / scale / species "
                 "(AMI, corrected for chance), one-to-one matched accuracy when K equals the number of labels, stability over resamples of individual plants."
                 "<br><b>Soft membership</b>: resample individual plants 50 times and recluster; the share of runs a photo lands in each group is its "
                 "membership; max membership &lt;0.8 = ambiguous. The earlier diagonal-GMM result is withdrawn: it gets harder with dimension "
                 "(22% ambiguous in 2-D, 1.3% in 32-D, table below)."
                 "<br><b>Labels first</b>: each group is described by its organ and scale make-up with a unique name; ambiguous photos are then sorted into 5 "
                 "categories using the photo's <b>own</b> organ label (explained below)."),
        _t(lang, "<p>各 K 的分组与标签一致性（含 K=5/10/15 的扫描在 results/*/exp4b_k_sweep.csv）：</p>", "<p>Agreement of groups with labels per K (scan incl. K=5/10/15 in results/*/exp4b_k_sweep.csv):</p>")
        + table(kt, d=2)
        + _det(_t(lang, "旧方法为什么作废：对角 GMM 的维度假象（EfficientNet，K=20）", "Why the old method is withdrawn: diagonal-GMM dimension artefact (EfficientNet, K=20)"), table(art, d=3))
        + _t(lang, "<p>软归属：含混比例与含混照片的类别构成（每行各类之和为 1）：</p>", "<p>Soft membership: share ambiguous and what kind of ambiguity (categories sum to 1 per row):</p>")
        + table(st, d=2)
        + f"<details open><summary>{_t(lang, f'EfficientNet K={MAIN_K}：每一组长什么样（离组中心最近的 4 张）', f'EfficientNet K={MAIN_K}: what each group looks like (4 photos nearest the centre)')}</summary>{faces}</details>"
        + f"<details open><summary>{_t(lang, '最常见的 5 对“夹在中间”的组合（每对 6 张）', 'The 5 most common in-between pairs (6 photos each)')}</summary>{pairs}</details>"
        + f"<details><summary>{_t(lang, '按类别看含混照片（每类最多 8 张）', 'Ambiguous photos by category (up to 8 each)')}</summary>{cats}</details>",
        _t(lang, f"<b>K</b>：EfficientNet / DINOv3 在 K=3 与距离、K=7 与器官对应最好，BioCLIP 在这几个 K 下与器官、距离都对应很弱（它按物种组织）；没有哪个 K 在所有方面都最好，所以全报告都按多个 K 报告。"
                 f"<br><b>软聚类</b>（EfficientNet K={MAIN_K}）：{e7['frac_max_lt_0_8']:.0%} 的照片含混，且可重复（ρ {e7['entropy_split_half_spearman']:.2f}）。其中"
                 f"“两个器官都合理” {sh['both_plausible']:.0%}、“偏向错的器官” {sh['leaning_wrong']:.0%}、“两个都不对” {sh['neither']:.0%}、“同器官内部” {sh['same_organ']:.0%}。"
                 "K 越大，含混越多是同一器官内部或混合组的差别，不再是器官问题。含混不能预测认错物种。"
                 "<b>“两个器官都合理”只是候选</b>：真正判断画面里有没有两种器官，需要多器官标签（尚未导出）或人工看图。",
                 f"<b>K</b>: EfficientNet / DINOv3 match scale best at K=3 and organ at K=7; BioCLIP matches neither at these K (it is organised by species). No K is best for everything, "
                 f"so the whole report is shown for several K.<br><b>Soft clustering</b> (EfficientNet K={MAIN_K}): {e7['frac_max_lt_0_8']:.0%} of photos are ambiguous, "
                 f"repeatably (ρ {e7['entropy_split_half_spearman']:.2f}): both organs plausible {sh['both_plausible']:.0%}, leaning to the wrong organ "
                 f"{sh['leaning_wrong']:.0%}, neither {sh['neither']:.0%}, same organ {sh['same_organ']:.0%}. At larger K ambiguity becomes mostly within-organ "
                 "or mixed-group variation. Ambiguity does not predict species errors. <b>\"Both plausible\" is a candidate list</b>: whether two organs are "
                 "really visible needs multi-organ labels (not exported) or a human look."),
        ["v", "e", "x"])


# ---------------------------------------------------------------- 7 retrieval
def sec7(lang, section, table, strip, s7):
    S = _ms("exp7_retrieval_summary.json")
    pk = pd.DataFrame([dict(model=m, **r) for m, s in S.items() for r in s["per_k"] if r["K"] in K_LIST])
    cs = s7["cases"]
    pick = [n for n in cs if n.startswith(f"K{MAIN_K} ")][:4] + ["PC1 -2SD", "PC1 +2SD"]
    html = "".join(f"<h4>{p}</h4>" + strip(cs[p], cap=lambda r: f"{r['organ']} · {r['shot_scale']} · cos {r['cos']:.2f}") for p in pick)
    return section(7, _t(lang, "反演：检索最近的真实图片", "Inversion: retrieve the nearest real images"),
        _t(lang, "某一组的中心、某条轴上的点“长什么样”？", "What does a group centre, or a point along an axis, look like?"),
        _t(lang, "不训练生成模型。在完整向量空间里，用余弦相似度找离以下点最近的 6 张真实照片：每个模型、每个 K（3 / 7 / 20）的各组中心，以及 PC1 / PC2 的 均值 ± 2SD 点。",
                 "No generative model. In the full space, the 6 nearest real photos (cosine) to: every group centre for each model and K (3 / 7 / 20), and the mean ± 2 SD points on PC1 / PC2."),
        table(pk.pivot_table(index="model", columns="K", values="top6_share_dominant_organ", sort=False).reset_index(), d=2)
        + _t(lang, "<p class=note>上表：检索出的前 6 张中，属于该组主导器官的比例。</p>", "<p class=note>Above: share of the top-6 retrieved photos showing the group's dominant organ.</p>")
        + _det(_t(lang, "完整表", "Full table"), table(pk.rename(columns={"top6_in_cluster": _t(lang, "前6张属于该组", "top-6 in group"),
                                 "top6_share_dominant_organ": _t(lang, "前6张主导器官占比", "top-6 dominant organ"),
                                 "cluster_dominant_share": _t(lang, "整组主导器官占比", "group dominant organ")}), d=2))
        + _t(lang, f"<p>示例：EfficientNet K={MAIN_K} 的前 4 个组中心，以及 PC1 两端。</p>", f"<p>Examples: EfficientNet K={MAIN_K}, first 4 centres, and both ends of PC1.</p>") + html,
        _t(lang, "每个 K 下，检索出的照片几乎都属于该组，且主导器官占比高于整组平均：检索给出的是最典型的成员，会让组看起来比实际更整齐。PC1 −2SD → 近拍叶子，+2SD → 远景整树。这是检索，不是生成。",
                 "At every K the retrieved photos almost all belong to the group and show its dominant organ more often than the group overall: retrieval "
                 "shows the most typical members and makes groups look tidier than they are. PC1 −2SD → close-up leaves, +2SD → distant trees. Retrieval, not generation."),
        ["v", "e"])
