"""Briefing report organised around the eight requests from the previous review.
Writes boss_report_zh.html and boss_report_en.html (self-contained: figures and thumbnails embedded).
Reuses existing results only; every number is read from results/."""
import base64
import json
from html import escape
from pathlib import Path

import pandas as pd

from common import IMG_DIR, MAIN_K, load

H = Path(__file__).resolve().parent
R = H / "results"
BCV = H.parent
M3 = {"EfficientNet-B0": R, "BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3"}
M5 = {**M3, "YOLO11s backbone": R / "yolo11s_backbone", "YOLO11s neck": R / "yolo11s_neck"}
meta, _ = load()
species = dict(zip(meta.image_id, meta.species))


def J(d, n):
    return json.loads((d / n).read_text())


def img(p, alt=""):
    return f"<img class=fig src='data:image/png;base64,{base64.b64encode(Path(p).read_bytes()).decode()}' alt='{escape(alt)}'>"


def thumb(image_id):
    p = IMG_DIR / "tn" / (image_id.replace("/", "_") + ".jpg")
    return f"data:image/jpeg;base64,{base64.b64encode(p.read_bytes()).decode()}"


def strip(recs, cap):
    return "<div class=strip>" + "".join(
        f"<figure><img src='{thumb(r['image_id'])}' alt='{escape(r['image_id'])}'><figcaption>{cap(r)}</figcaption></figure>"
        for r in recs) + "</div>"


def table(df, d=2, pct=False):
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)
    f = lambda v: (f"{v:.0%}" if pct else f"{v:.{d}f}") if isinstance(v, float) else escape(str(v))
    first = df.iloc[0] if len(df) else []
    h = "".join(f"<th{' class=n' if len(df) and num(first.iloc[i]) else ''}>{escape(str(c))}</th>" for i, c in enumerate(df.columns))
    b = "".join("<tr>" + "".join(f"<td{' class=n' if num(v) else ''}>{f(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<div class=tw><table><tr>{h}</tr>{b}</table></div>"


def panel(label, body, finding):
    """Numbered figure / table with a one-sentence finding underneath."""
    return f"<div class=panel><div class=plabel>{label}</div>{body}<p class=finding>{finding}</p></div>"


# ------------------------------------------------------------------ numbers
s6 = {m: J(d, "exp6_summary.json") for m, d in M5.items()}
fid = pd.DataFrame([{"model": m, "2-D": s["two_d_all_images"]["knn15_overlap_all1641"],
                     "64-D": {str(r["d"]): r for r in s["table"]}["64"]["val_knn15_overlap"],
                     "var64": {str(r["d"]): r for r in s["table"]}["64"]["train_explained_var"],
                     "organ_f1": {str(r["d"]): r for r in s["table"]}["full"]["probe_organ_macroF1"],
                     "species_f1": {str(r["d"]): r for r in s["table"]}["full"]["probe_species_macroF1"]} for m, s in s6.items()])
ks = pd.concat([pd.read_csv(d / "exp4b_k_sweep.csv").assign(model=m) for m, d in M3.items()])
kt = pd.DataFrame([{"model": m,
                    "K3": g[g.K == 3].matched_acc_scale.iloc[0], "K7": g[g.K == 7].matched_acc_organ.iloc[0],
                    "stab3": g[g.K == 3].stability_ari_mean.iloc[0], "stab7": g[g.K == 7].stability_ari_mean.iloc[0],
                    "stab20": g[g.K == 20].stability_ari_mean.iloc[0]} for m, g in ks.groupby("model", sort=False)])
soft = J(R, "exp4_soft_summary.json")
e7 = soft["results"][f"K{MAIN_K}"]
main = soft["main"]
org = {c["cluster"]: c["top_organ"] for c in main["clusters"]}
short = {c["cluster"]: f"{c['top_organ']} {c['top_organ_share']:.0%}" for c in main["clusters"]}
mm = pd.read_csv(R / "exp4_soft_memberships.csv")
mm = mm[(mm.K == MAIN_K) & mm.category_if_ambiguous.notna() & (mm.category_if_ambiguous != "")]
mm["pair"] = [" ↔ ".join(sorted((org[a], org[b]))) for a, b in zip(mm.top1, mm.top2)]
pairs = mm.groupby("pair").agg(n=("image_id", "size"), plausible=("category_if_ambiguous", lambda s: (s == "both_plausible").sum()))
pairs = pairs.sort_values("n", ascending=False).head(6).reset_index()
pairs["share"] = pairs.n / len(mm)
art = pd.DataFrame(soft["gmm_dimension_artifact"])
enr = pd.DataFrame([{**next(r for r in J(d, "exp2_enrichment_summary.json")["summary"] if r["K"] == MAIN_K and r["label"] == "organ"), "model": m}
                    for m, d in M5.items()])
nsc = {m: J(d, "exp1_nsc_summary.json")["results"] for m, d in M5.items()}
nsc_t = pd.DataFrame([{"model": m, "organ": r["organ (external label)"]["test"]["delta_1se"]["coords_any_class"],
                       "per_class": r["organ (external label)"]["test"]["delta_1se"]["coords_mean_per_class"],
                       "ratio": r["organ (external label)"]["ratio_F1_1se_over_delta0"],
                       **{f"K{k}": r[f"grouping approximation: K={k} cluster ID"]["test"]["delta_1se"]["coords_any_class"] for k in (3, 7, 20)}}
                      for m, r in nsc.items()])
im = pd.concat([pd.read_csv(d / "exp8_info_matrix.csv").assign(model=m) for m, d in M5.items() if (d / "exp8_info_matrix.csv").exists()])
im_t = im[["model", "view", "neighbours_kept", "organ_kept", "scale_kept", "species_kept"]]
im_sp = im[im.view.isin(["PCA-3", "PCA-64"])].pivot_table(index="model", columns="view", values="species_kept", sort=False).reset_index()
im_org = im[im.view.isin(["PCA-3", "PCA-16", "PCA-64"])].pivot_table(index="model", columns="view", values="organ_kept", sort=False)[["PCA-3", "PCA-16", "PCA-64"]].reset_index()
effm = pd.read_csv(R / "exp8_info_matrix.csv").set_index("view")
pc_t = pd.read_csv(R / "exp8_pc_information.csv").head(6)
pc_t["main"] = pc_t[["organ", "scale", "species"]].idxmax(axis=1)
s5 = J(R, "exp5_axes_summary.json")
ae = pd.read_csv(R / "exp5_recovered_ae_table.csv")
s3 = J(R, "exp3_activation_summary.json")
s7 = J(R, "exp7_retrieval_summary.json")
ret = pd.DataFrame([{"model": m, **{f"K{r['K']:.0f}": r["top6_share_dominant_organ"] for r in J(d, "exp7_retrieval_summary.json")["per_k"]}}
                    for m, d in M3.items()])
n_amb = len(mm)
n_pl = int((mm.category_if_ambiguous == "both_plausible").sum())
eff_f = fid.set_index("model")


def pct(x):
    return f"{x:.0%}"


# ------------------------------------------------------------------ content
def build(lang):
    zh = lang == "zh"
    T = lambda a, b: a if zh else b
    cap_photo = lambda r: f"{escape(species.get(r['image_id'], ''))}<br>{r['organ']} · {r['shot_scale']}"
    cap_soft = lambda r: (f"{escape(species.get(r['image_id'], ''))}<br><b>{T('标签', 'label')}: {r['organ']}</b><br>"
                          f"{T('候选', 'cand.')}1 {short[r['c1']]} ({r['p1']:.2f})<br>{T('候选', 'cand.')}2 {short[r['c2']]} ({r['p2']:.2f})")

    def sec(n, title, ask, method, tried, result, take, status):
        st = {"done": T("已完成", "Done"), "part": T("部分完成", "Partial"), "next": T("下一步", "Next")}[status]
        return (f"<section id=s{n}><div class=hd><span class=num>{n}</span><h2>{title}</h2><span class='st {status}'>{st}</span></div>"
                f"<dl><dt>{T('目的', 'Objective')}</dt><dd>{ask}</dd><dt>{T('方法', 'Method')}</dt><dd>{method}</dd>"
                f"<dt>{T('尝试', 'What we tried')}</dt><dd>{tried}</dd></dl><h4 class=rh>{T('结果', 'Results')}</h4><div class=res>{result}</div>"
                f"<p class=take><b>{T('结论 / 下一步', 'Decision / next step')}：</b>{take}</p></section>")

    E, B, D = eff_f.loc["EfficientNet-B0"], eff_f.loc["BioCLIP 2.5"], eff_f.loc["DINOv3"]
    kE, kD = kt.set_index("model").loc["EfficientNet-B0"], kt.set_index("model").loc["DINOv3"]
    en = enr.set_index("model").share_clusters_dominant_ge_0_6
    nt = nsc_t.set_index("model")
    yb, yn = eff_f.loc["YOLO11s backbone"], eff_f.loc["YOLO11s neck"]

    rows = [
        (1, T("可视化的可信度", "Reliability of visualisation"), T("信息保留矩阵：各降维方式保留多少器官 / 距离 / 物种信息", "Information-retention matrix: organ / scale / species kept by each reduction"),
         T(f"二维只保留 {pct(fid['2-D'].min())}–{pct(fid['2-D'].max())}", f"2-D keeps only {pct(fid['2-D'].min())}–{pct(fid['2-D'].max())}"), "done"),
        (1, T("降维保留了什么信息", "What dimension reduction keeps"), T("测不同维数保留的信息", "Measured information kept per dimension"),
         T(f"64 维保留 {pct(fid['64-D'].min())}–{pct(fid['64-D'].max())} 的近邻", f"64-D keeps {pct(fid['64-D'].min())}–{pct(fid['64-D'].max())} of neighbours"), "done"),
        (2, T("聚类数 K 的选择", "Choosing the number of clusters K"), T("K=3/7/20 对比 + 稳定性", "K = 3 / 7 / 20 with stability"),
         T("K=3 对应拍摄距离，K=7 对应器官", "K=3 matches shot scale, K=7 matches organ"), "done"),
        (2, T("软聚类", "Soft clustering"), T("重采样共识软归属", "Bootstrap-consensus soft membership"),
         T(f"{pct(e7['frac_max_lt_0_8'])} 的照片夹在两组之间", f"{pct(e7['frac_max_lt_0_8'])} of photos sit between two groups"), "done"),
        (3, T("富集分析", "Enrichment"), T("每组的器官 / 距离 / 物种构成", "Organ / scale / species make-up of each group"),
         T("通用模型按器官分组，BioCLIP 按物种", "General models group by organ, BioCLIP by species"), "done"),
        (4, T("收缩质心（NSC）", "Shrunken centroids (NSC)"), "NSC",
         T(f"EfficientNet 每个器官约 {nt.loc['EfficientNet-B0','per_class']:.0f} 个坐标就够；其他模型不行", f"EfficientNet needs ~{nt.loc['EfficientNet-B0','per_class']:.0f} coordinates per organ; others cannot be compressed"), "done"),
        (5, T("Autoencoder 与主轴", "Autoencoder and axes"), T("AE vs PCA；主轴两端的真实照片", "AE vs PCA; real photos at both ends of each axis"),
         T("AE 没有提升；PC1 = 拍摄远近", "AE adds nothing; PC1 = camera distance"), "done"),
        (6, T("从隐空间回到图片", "Back from latent space to images"), T("检索最近的真实照片", "Retrieve the nearest real photos"),
         T("能展示每组的典型样子；生成图片不可行", "Shows each group's typical look; generation did not work"), "done"),
        (6, T("激活热图", "Activation mapping"), T("EfficientNet 上做 Grad-CAM + 遮挡对照", "Grad-CAM + occlusion on EfficientNet"),
         T("热图可信；是否落在器官上尚未验证", "Maps are faithful; whether they land on the organ is not yet tested"), "part"),
        (7, T("多标签", "Multiple labels"), T("用软聚类找出“两个器官都合理”的候选", "Soft clustering lists \"both organs plausible\" candidates"),
         T(f"{n_pl} 张候选已列出，需多标签数据验证", f"{n_pl} candidates listed; needs multi-label data to verify"), "next"),
        (8, T("团队新方向：YOLO", "Team direction: YOLO"), T("两个版本放进同样的分析", "Both versions through the same analyses"),
         T("器官信息弱，不建议直接用现成 embedding", "Weak organ information; raw embeddings not recommended"), "done"),
    ]
    st_lbl = {"done": T("已完成", "Done"), "part": T("部分", "Partial"), "next": T("下一步", "Next")}
    overview = ("<div class=tw><table class=ov><tr>" + "".join(f"<th>{h}</th>" for h in
                [T("主题", "Topic"), T("做了什么", "What we did"), T("一句话结果", "Result in one line"), T("状态", "Status"), "§"]) + "</tr>"
                + "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td><span class='st {s}'>{st_lbl[s]}</span></td><td><a href=#s{n}>{n}</a></td></tr>"
                          for n, a, b, c, s in rows) + "</table></div>")

    fid_t = fid.rename(columns={"2-D": T("二维近邻保留", "2-D neighbours kept"), "64-D": T("64维近邻保留", "64-D neighbours kept"),
                                "var64": T("64维方差", "64-D variance"), "organ_f1": T("器官 F1", "organ F1"), "species_f1": T("物种 F1", "species F1")})
    kt_t = kt.rename(columns={"K3": T("K=3 对上距离", "K=3 matches scale"), "K7": T("K=7 对上器官", "K=7 matches organ"),
                              "stab3": T("稳定性 K=3", "stability K=3"), "stab7": T("稳定性 K=7", "stability K=7"), "stab20": T("稳定性 K=20", "stability K=20")})
    pairs_t = pairs[["pair", "n", "share", "plausible"]].rename(columns={"pair": T("在哪两种器官之间", "between which two organs"),
                                                                     "share": T("占含混照片", "share of ambiguous"), "plausible": T("其中“两个都合理”", "of which both plausible")})
    enr_t = enr[["model", "share_clusters_dominant_ge_0_6", "median_top_species_share"]].rename(columns={
        "share_clusters_dominant_ge_0_6": T(f"K={MAIN_K} 由单一器官主导的组", f"K={MAIN_K} groups dominated by one organ"),
        "median_top_species_share": T("组内最大物种占比（中位数）", "median top-species share")})
    nsc_tt = nsc_t.rename(columns={"organ": T("器官：保留坐标", "organ: coords kept"), "per_class": T("器官：每类平均", "organ: per class"),
                                   "ratio": T("器官：保留准确率", "organ: accuracy kept"), "K3": T("分组 K=3", "grouping K=3"),
                                   "K7": T("分组 K=7", "grouping K=7"), "K20": T("分组 K=20", "grouping K=20")})
    big = pairs.pair.iloc[0]
    ex_pair = next(p for p in main["by_pair"] if " ↔ ".join(sorted((org[p["pair"][0]], org[p["pair"][1]]))) == big)
    pc1 = s5["ends"]["1"]
    cent = [n for n in s7["cases"] if n.startswith(f"K{MAIN_K} ")]

    S = []
    S.append(sec(1, T("在什么空间里分析：降维保留了什么信息", "Which space to analyse in: what dimension reduction keeps"),
        T("量化不同降维方式（二维 / 三维图、16 / 64 维）对器官、拍摄距离、物种信息的保留程度，并确定各主成分承载的信息类型，据此选定分析空间。",
          "Quantify how much organ, shot-scale and species information each reduction (2-D / 3-D plots, 16 / 64 dimensions) retains, identify the information carried by each principal component, and select the analysis space accordingly."),
        T("每张照片是一个高维向量（EfficientNet 1,280 维、BioCLIP 1,024、DINOv3 768、YOLO 512 / 896）。对每种降维方式算两件事：<br>"
          "<b>近邻保留</b>：原空间里最像的 15 张照片，降维后还剩几张；<br>"
          "<b>信息保留</b>：在降维后的空间里用“最近邻投票”判断器官 / 距离 / 物种（按植株分组交叉验证），准确率占原空间的百分比，100% = 没丢信息。<br>"
          "另算每个主成分的差异中有多少能被器官 / 距离 / 物种解释（η²，0–1）。全部是无监督降维，标签只用来评估。",
          "Each photo is a high-dimensional vector (EfficientNet 1,280, BioCLIP 1,024, DINOv3 768, YOLO 512 / 896). For each reduction we measure:<br>"
          "<b>Neighbours kept</b>: of each photo's 15 most similar photos in the full space, how many remain;<br>"
          "<b>Information kept</b>: nearest-neighbour vote for organ / scale / species in the reduced space (cross-validated, grouped by plant), as a "
          "percentage of the full-space accuracy; 100% = nothing lost.<br>Also how much of each principal component's variation is explained by organ / "
          "scale / species (η², 0–1). All reductions are unsupervised; labels are used only to evaluate."),
        T("PCA 2 / 3 / 16 / 64 维与 t-SNE 二维（五个模型）；UMAP 2 / 16 维（EfficientNet）。", "PCA to 2 / 3 / 16 / 64 dimensions and 2-D t-SNE (five models); UMAP to 2 / 16 dimensions (EfficientNet)."),
        panel(T("图 1.1 · 降维后各类信息的保留比例", "Figure 1.1 · Information kept after reduction"), img(R / "exp8_info_kept.png", "information kept"),
              T("拍摄距离在任何降维下都几乎完整保留；器官信息约需 16 维；物种信息需要 64 维。三维图中物种信息平均只剩 32%。",
                "Shot scale survives any reduction; organ information needs about 16 dimensions; species needs 64. In a 3-D plot species information falls to 32% on average."))
        + panel(T("图 1.2 · 同一批照片的三种二维画法（EfficientNet，按器官着色）", "Figure 1.2 · The same photos drawn three ways (EfficientNet, coloured by organ)"),
                img(R / "exp8_maps_effnet.png", "maps"),
                T(f"二维 PCA 中叶、花序、果实、小枝混在一起；t-SNE 与 UMAP 都能把器官分开。t-SNE 还保留了物种信息（{pct(effm.loc['t-SNE-2','species_kept'])}，UMAP 为 {pct(effm.loc['UMAP-2','species_kept'])}），因此展示统一用 t-SNE。",
                  f"In 2-D PCA leaf, inflorescence, fruit and twig overlap; t-SNE and UMAP both separate organs. t-SNE also keeps species information "
                  f"({pct(effm.loc['t-SNE-2','species_kept'])} vs {pct(effm.loc['UMAP-2','species_kept'])} for UMAP), so figures use t-SNE."))
        + panel(T("表 1.1 · EfficientNet 各降维方式的信息保留", "Table 1.1 · Information kept by each reduction, EfficientNet"),
                table(im_t[im_t.model == "EfficientNet-B0"].drop(columns="model").rename(columns={
                    "view": T("降维方式", "view"), "neighbours_kept": T("近邻保留", "neighbours kept"), "organ_kept": T("器官信息", "organ"),
                    "scale_kept": T("距离信息", "shot scale"), "species_kept": T("物种信息", "species")}), pct=True),
                T(f"PCA 64 维在三类信息上都接近完整，且近邻关系保留最多（{pct(effm.loc['PCA-64','neighbours_kept'])}）；二维图在任何方法下近邻都不到一半。",
                  f"64-D PCA keeps all three kinds of information almost completely and the most neighbour relations ({pct(effm.loc['PCA-64','neighbours_kept'])}); "
                  "every 2-D view keeps fewer than half."))
        + panel(T("表 1.2 · 五个模型的器官信息保留", "Table 1.2 · Organ information kept, five models"), table(im_org.rename(columns={"model": T("模型", "model")}), pct=True),
                T(f"所有模型的三维图都丢掉 {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} 的器官信息；64 维时全部恢复到 {pct(im_org['PCA-64'].min())} 以上。",
                  f"Every model's 3-D plot loses {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} of organ information; at 64 dimensions all "
                  f"recover to at least {pct(im_org['PCA-64'].min())}."))
        + panel(T("表 1.3 · EfficientNet 前 6 个主成分承载的信息（η²，0–1）", "Table 1.3 · Information carried by EfficientNet's first 6 components (η², 0–1)"),
                table(pc_t.assign(main=pc_t["main"].map(lambda v: {"organ": T("器官", "organ"), "scale": T("距离", "scale"), "species": T("物种", "species")}[v])).rename(
                    columns={"pc": "PC", "organ": T("器官", "organ"), "scale": T("距离", "scale"), "species": T("物种", "species"), "main": T("主要承载", "mainly carries")}), 2),
                T("前三个主成分主要区分器官（PC1 同时区分拍摄远近），物种信息分散在后面的成分中，所以只看前三维时看不到物种。",
                  "The first three components mainly separate organs (PC1 also camera distance); species information is spread over later components, so it is invisible in the first three.")),
        T(f"三维 PCA 图丢掉 {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} 的器官信息、保留的近邻关系不到 30%，<b>不能用来判断器官结构或聚类</b>。"
          "PCA 64 维对器官、距离、物种信息都接近完整保留，<b>后续所有分析都在 64 维中进行</b>。展示用二维图统一采用 t-SNE：图中的分组可信，但距离、簇大小和位置不具有含义。",
          f"A 3-D PCA plot loses {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} of organ information and keeps fewer than 30% of neighbour "
          "relations, so <b>it cannot be used to judge organ structure or clusters</b>. 64-D PCA keeps organ, scale and species information almost completely: "
          "<b>all further analysis runs in 64 dimensions</b>. Display figures use 2-D t-SNE: its grouping is reliable, but distances, cluster sizes and positions carry no meaning."), "done"))

    S.append(sec(2, T("聚类数 K 的选择与软聚类", "Choice of K and soft clustering"),
        T("以数值指标选定聚类数 K，并用软归属识别处于两组之间的照片。", "Select the number of clusters K with numerical criteria, and identify photos lying between two groups with soft membership."),
        T("K 不是给定的，取有含义的值：K=3（3 种拍摄距离）、K=7（7 种器官）、K=20（旧默认）。看分组和标签能否一一对上，以及按植株重采样后是否稳定。"
          "<br>软归属：按植株重采样 50 次、每次重新聚类，照片落入各组的比例就是它的归属；最大归属 &lt;0.8 记为“含混”。",
          "K is a choice; we use meaningful values: K=3 (three shot scales), K=7 (seven organs), K=20 (old default). Do groups match labels one-to-one, and "
          "are they stable when plants are resampled?<br>Soft membership: resample plants 50 times and recluster; the share of runs a photo lands in each "
          "group is its membership. Max membership &lt;0.8 = ambiguous."),
        T(f"第一版用高斯混合模型，结果是“几乎没有含混照片”。复查发现这是维度造成的假象：同一数据在 2 维时 {pct(art.frac_max_lt_0_6.iloc[0])} 含混，32 维只剩 {pct(art.frac_max_lt_0_6.iloc[4])}。已作废，改用重采样方法。",
          f"First version (Gaussian mixture) said almost no photo is ambiguous. On review this was a dimension artefact: same data, {pct(art.frac_max_lt_0_6.iloc[0])} "
          f"ambiguous in 2-D, {pct(art.frac_max_lt_0_6.iloc[4])} in 32-D. Withdrawn and replaced by resampling."),
        table(kt_t, 2)
        + T(f"<p>EfficientNet K={MAIN_K}：{pct(e7['frac_max_lt_0_8'])} 的照片含混（{n_amb} 张），且可重复（两半独立运行相关 {e7['entropy_split_half_spearman']:.2f}）。它们最常夹在：</p>",
            f"<p>EfficientNet K={MAIN_K}: {pct(e7['frac_max_lt_0_8'])} of photos are ambiguous ({n_amb}), repeatably (two independent halves correlate "
            f"{e7['entropy_split_half_spearman']:.2f}). Most often between:</p>")
        + table(pairs_t, 2)
        + T(f"<p class=cap>例：夹在“{big}”之间的照片</p>", f"<p class=cap>Example: photos between {big}</p>") + strip(ex_pair["cases"], cap_soft),
        T("K=3 对应拍摄距离，K=7 对应器官，K=20 没有依据。软聚类找到的含混是真实且可重复的，大多发生在相邻器官之间（果实 ↔ 小枝、花序 ↔ 叶），但它不能预测认错物种。",
          "K=3 corresponds to shot scale, K=7 to organ; K=20 has no basis. The ambiguity is real and repeatable, mostly between neighbouring organs "
          "(fruit ↔ twig, inflorescence ↔ leaf), but it does not predict species errors."), "done"))

    S.append(sec(3, T("各组构成：富集分析", "Group composition: enrichment analysis"), T("量化每组的器官、拍摄距离与物种构成，判断各组由何种因素驱动。", "Quantify the organ, shot-scale and species make-up of each group and determine what drives it."),
        T("对每组计算各器官 / 距离的占比与全体平均的倍数（富集倍数），并按物种构成校正；置信区间按植株重采样得到。照片嵌套在植株和物种内，不满足独立性假设，所以只报告效应大小，不报 p 值。",
          "For every group: share of each organ / scale relative to the overall rate (fold enrichment), adjusted for species mix, with intervals from "
          "resampling plants. Photos are nested in plants and species, so independence fails: effect sizes only, no p-values."),
        T("五个模型 × K=3/7/20。", "Five models × K = 3 / 7 / 20."),
        table(enr_t, 2) + img(R / f"exp2_organ_composition_k{MAIN_K}.png", "composition"),
        T(f"EfficientNet（{pct(en['EfficientNet-B0'])}）和 DINOv3（{pct(en['DINOv3'])}）的组大多由单一器官主导，且不是“同一个物种的组”；BioCLIP（{pct(en['BioCLIP 2.5'])}）按物种分组；YOLO 只有约三成。",
          f"EfficientNet ({pct(en['EfficientNet-B0'])}) and DINOv3 ({pct(en['DINOv3'])}) groups are mostly dominated by one organ and are not single-species "
          f"groups; BioCLIP ({pct(en['BioCLIP 2.5'])}) groups follow species; YOLO about a third."), "done"))

    S.append(sec(4, T("决定分组的维度：收缩质心（NSC）", "Dimensions that define the groups: shrunken centroids (NSC)"), T("找出区分器官、以及复现聚类所需的最少维度，衡量信息的集中程度。", "Find the fewest dimensions needed to separate organs and to reproduce the clusters, as a measure of how concentrated the information is."),
        T("NSC（最近收缩质心）：给每类算中心，看它在每个坐标上偏离平均多少；<b>Δ 是门槛</b>，偏离小于 Δ 的坐标对这一类删掉。Δ=0 全保留，Δ 越大越精简。"
          "用按植株分组的 5 折交叉验证 + “一倍标准误”规则选 Δ（统计上和最好一样好的最精简模型），测试集只评估一次。坐标是 PCA 方向，不是生物学特征。",
          "NSC (nearest shrunken centroids): a centre per class and its deviation on every coordinate; <b>Δ is a threshold</b>, coordinates deviating less "
          "than Δ are dropped for that class. Δ=0 keeps all; larger Δ is sparser. Δ chosen by 5-fold CV grouped by plant and the one-standard-error rule "
          "(sparsest model statistically as good as the best); test set used once. Coordinates are PCA directions, not biological traits."),
        T("两个任务：(a) 外部器官标签；(b) 复现模型自己 K=3/7/20 的分组。五个模型。", "Two tasks: (a) external organ labels; (b) reproducing the model's own K = 3 / 7 / 20 groups. Five models."),
        img(R / "exp1c_nsc_organ_models.png", "organ") + img(R / "exp1c_nsc_grouping_models.png", "grouping") + table(nsc_tt, 2)
        + T("<p class=cap>读图：横轴向右 = 保留的坐标越少；纵轴 = 相对全部坐标的准确率；圆点 = 选出的 Δ。先平后断崖 = 信息集中；一开始就下滑 = 信息分散。</p>",
            "<p class=cap>Reading: to the right = fewer coordinates; y = accuracy relative to all coordinates; dot = chosen Δ. Flat then a cliff = concentrated; "
            "sliding from the start = spread out.</p>")
        + img(R / "exp1b_nsc_heatmap.png", "heatmap")
        + T("<p class=cap>EfficientNet：每个器官靠哪些方向识别（红 / 蓝 = 偏向方向的两端，白 = 不用）。</p>",
            "<p class=cap>EfficientNet: which directions identify each organ (red / blue = the two ends, white = not used).</p>"),
        T(f"只有 EfficientNet 能把器官压缩到每类约 {nt.loc['EfficientNet-B0','per_class']:.0f} 个坐标且几乎不掉准确率；其他模型的器官信息分散。"
          "复现分组需要的坐标随 K 增大而增多，之前“要用近 64 个坐标”是 K=20 造成的。",
          f"Only EfficientNet compresses organ to ~{nt.loc['EfficientNet-B0','per_class']:.0f} coordinates per class with almost no loss; elsewhere organ "
          "information is spread out. Coordinates needed to reproduce the groups grow with K; \"almost all 64\" was a K=20 effect."), "done"))

    S.append(sec(5, T("Autoencoder 与主轴", "Autoencoder and axes"), T("比较非线性降维（Autoencoder）与 PCA，并解释主要主成分的含义。", "Compare non-linear reduction (autoencoder) with PCA and interpret the main principal components."),
        T("32 维 Autoencoder 与 PCA 比较重建误差和近邻保留；再取 PCA 主轴两端的真实照片，并计算主轴与拍摄距离、器官的相关。",
          "A 32-D autoencoder vs PCA on reconstruction error and neighbour retention; then real photos at both ends of each principal axis and its "
          "correlation with shot scale and organ."),
        T("AE 结果从上次网站恢复，并用线性 AE 复核（与 PCA 一致，说明代码和数据对齐）。", "AE results recovered from the previous site and checked with a linear AE (matches PCA, so code and data line up)."),
        table(ae, 6)
        + T(f"<p class=cap>EfficientNet PC1 两端（与拍摄距离相关 ρ = {s5['axes'][0]['rho_scale_close']:.2f}）：</p>",
            f"<p class=cap>EfficientNet PC1, both ends (correlation with shot scale ρ = {s5['axes'][0]['rho_scale_close']:.2f}):</p>")
        + strip(pc1["low"], cap_photo) + strip(pc1["high"], cap_photo),
        T("AE 的误差和 PCA 几乎一样、近邻保留更差，非线性方法没带来收益，不再继续。第一主轴就是“拍得多远”：一端近拍叶子，一端远景整树。",
          "The AE matches PCA on error and is worse on neighbours: no gain from non-linearity, line closed. The first axis is camera distance: close-up "
          "leaves at one end, distant whole trees at the other."), "done"))

    S.append(sec(6, T("从隐空间回到图片", "Back from the latent space to images"),
        T("由隐空间中的点检索对应的真实图片，并以激活热图定位模型依赖的图像区域。", "Map points in the latent space back to real photos, and locate the image regions the model relies on with activation maps."),
        T("反演：对每组中心，检索最相似的真实照片。热图：在 EfficientNet（与 ResNet 同类的卷积网络）上，以“属于某组”为目标做 Grad-CAM，并用遮挡对照检验热图是否可信。",
          "Inversion: for each group centre, retrieve the most similar real photos. Heatmaps: Grad-CAM on EfficientNet (a CNN like ResNet) with "
          "\"belongs to group c\" as the target, checked by occlusion."),
        T(f"上一轮从噪声生成图片，只得到彩色噪点，改为检索。热图在 80 张按规则抽样的照片上做，遮挡热区比随机遮挡掉分更多的比例为 {pct(s3['frac_top_cam_beats_random'])}。",
          f"Generating images from noise (previous round) gave coloured noise; replaced by retrieval. Heatmaps on 80 rule-sampled photos: hiding the hot area "
          f"hurts the score more than a random area in {pct(s3['frac_top_cam_beats_random'])} of cases."),
        table(ret.rename(columns={c: T(f"{c} 检索结果为主导器官", f"{c} retrieved = dominant organ") for c in ["K3", "K7", "K20"]}), 2)
        + T("<p class=cap>两个组中心的最近真实照片：</p>", "<p class=cap>Nearest real photos to two group centres:</p>")
        + strip(s7["cases"][cent[0]], cap_photo) + strip(s7["cases"][cent[2]], cap_photo)
        + img(R / "exp3_gradcam_cases.png", "Grad-CAM")
        + T("<p class=cap>Grad-CAM（早期 K=20 结果）：越红 = 模型越依赖该区域。</p>", "<p class=cap>Grad-CAM (earlier K=20 run): red = relied on most.</p>"),
        T("检索能直观展示每组“长什么样”（展示的是最典型成员，会显得比实际整齐）。热图对模型是可信的，但<b>还没验证是否落在植物器官上</b>，需要器官掩膜。",
          "Retrieval shows what each group looks like (its most typical members, so groups look tidier than they are). Heatmaps are faithful to the model, "
          "but <b>whether they land on the organ is not yet tested</b>; that needs organ masks."), "part"))

    S.append(sec(7, T("每张图多个标签", "Multiple labels per image"),
        T("利用软聚类结果，筛选可能含有多个器官、需要多标签标注的照片。", "Use soft-clustering results to shortlist photos that may contain several organs and need multi-label annotation."),
        T("目前器官标签是单标签。软聚类中，照片落在自己器官的组、但第二候选是另一种器官，记为“两个器官都合理”，作为多标签候选。",
          "Organ labels are single-label today. In soft clustering, a photo in its own organ's group whose second candidate is another organ is marked "
          "\"both organs plausible\": a multi-label candidate."),
        T(f"EfficientNet K={MAIN_K}：{n_pl} 张候选（占含混照片 {pct(n_pl / n_amb)}）。", f"EfficientNet K={MAIN_K}: {n_pl} candidates ({pct(n_pl / n_amb)} of ambiguous photos)."),
        strip(main["by_category"]["both_plausible"], cap_soft),
        T("候选清单已经可以交给标注；要确认画面里是否真有两种器官，需要导出多标签数据（Human Gold / Gemma 多标签）后验证。",
          "The candidate list is ready for annotation; confirming two organs in the frame needs the multi-label data (Human Gold / Gemma multi-label) exported."), "next"))

    yt = pd.DataFrame([{"model": m, T("器官 F1", "organ F1"): eff_f.loc[m, "organ_f1"], T("物种 F1", "species F1"): eff_f.loc[m, "species_f1"],
                        T("NSC 器官每类坐标", "NSC organ coords / class"): nt.loc[m, "per_class"],
                        T(f"K={MAIN_K} 单一器官主导的组", f"K={MAIN_K} organ-dominated groups"): en[m]} for m in M5])
    S.append(sec(8, T("YOLO 对比（团队新方向）", "YOLO comparison (team direction)"),
        T("评估 YOLO 的整图 embedding 是否适合器官结构分析，为是否换用 YOLO 提供依据。", "Assess whether YOLO whole-image embeddings support organ-structure analysis, to inform the decision on switching to YOLO."),
        T("YOLO11s（COCO 通用检测模型，未在植物上训练）的整图 embedding，两个版本：backbone（512 维）和 neck 多尺度（896 维），分别放进与其他模型相同的分析。",
          "Whole-image embeddings of YOLO11s (COCO general detector, not trained on plants), two versions: backbone (512-D) and multi-scale neck (896-D), "
          "run through the same analyses as the other models."),
        T("器官 / 物种探针、NSC、分组构成、检索。", "Organ / species probes, NSC, group make-up, retrieval."),
        table(yt, 2),
        T(f"YOLO 的器官信息最弱（器官 F1：backbone {yb['organ_f1']:.2f}、neck {yn['organ_f1']:.2f}），也无法压缩；它的分组有清楚结构，但不是按器官组织。"
          "<b>若换用 YOLO，需要用我们的器官标签训练，不能直接用现成 embedding。</b>这不代表 YOLO 训练后画器官框的能力差。",
          f"YOLO carries the least organ information (organ F1: backbone {yb['organ_f1']:.2f}, neck {yn['organ_f1']:.2f}) and cannot be compressed; its "
          "groups have clear structure, but not by organ. <b>If we switch to YOLO, it needs training on our organ labels; raw embeddings are not enough.</b> "
          "This says nothing about YOLO's box detection after training."), "done"))

    nxt = T(["导出多标签器官数据（Human Gold），验证“两个器官都合理”候选。", "做器官掩膜，检验热图是否落在植物器官上。",
             "若采用 YOLO：用器官框标注做微调，再用同一套分析复验。", "在 Pound Ridge 外部野外照片上复验以上结论。"],
            ["Export multi-label organ data (Human Gold) and verify the \"both plausible\" candidates.", "Build organ masks to test whether heatmaps land on the organ.",
             "If YOLO is adopted: fine-tune on organ boxes, then rerun the same analyses.", "Re-test the conclusions on external Pound Ridge field photos."])

    css = """
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--line:#e3e3e3;--soft:#f6f6f4;--acc:#1f3a5f;--ok:#2e6b3a;--warn:#8a5a00;--next:#4a4a8a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141414;--fg:#e8e8e8;--mut:#a0a0a0;--line:#333;--soft:#1d1d1d;--acc:#9db8e0;--ok:#8fcf9b;--warn:#e6b85c;--next:#b0b0f0;color-scheme:dark}}
:root[data-theme=dark]{--bg:#141414;--fg:#e8e8e8;--mut:#a0a0a0;--line:#333;--soft:#1d1d1d;--acc:#9db8e0;--ok:#8fcf9b;--warn:#e6b85c;--next:#b0b0f0;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"Segoe UI","PingFang SC","Noto Sans CJK SC",Helvetica,Arial,sans-serif;margin:0;
padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
main{max-width:960px;margin:auto;padding-block:40px 80px}
h1{font-size:26px;margin:0 0 4px;text-wrap:balance}.sub{color:var(--mut);margin:0 0 28px}
h2{font-size:19px;margin:0;text-wrap:balance}h3{font-size:15px;margin:28px 0 8px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut)}
section{border-top:1px solid var(--line);padding-top:26px;margin-top:34px}
.hd{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}.num{color:var(--acc);font-weight:700;font-size:19px;font-variant-numeric:tabular-nums}
blockquote{margin:10px 0 14px;padding:6px 14px;border-left:3px solid var(--acc);color:var(--mut);background:var(--soft)}
dl{display:grid;grid-template-columns:110px 1fr;gap:6px 16px;margin:0}dt{font-weight:600;color:var(--mut);font-size:13px;text-transform:uppercase;letter-spacing:.04em;padding-top:2px}
dd{margin:0;min-width:0}@media(max-width:640px){dl{grid-template-columns:1fr}}
.take{margin:16px 0 0;padding:10px 14px;background:var(--soft);border-radius:4px}
.st{font-size:12px;padding:1px 8px;border:1px solid currentColor;border-radius:3px;white-space:nowrap}.done{color:var(--ok)}.part{color:var(--warn)}.next{color:var(--next)}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;border-bottom:1.5px solid var(--fg);padding:5px 10px 5px 0}td{border-bottom:1px solid var(--line);padding:5px 10px 5px 0;vertical-align:top}
.ov td:first-child{font-weight:600}a{color:var(--acc)}
.fig{max-width:min(100%,820px);height:auto;display:block;margin:12px 0;border:1px solid var(--line);background:#fff}
.strip{display:flex;gap:8px;overflow-x:auto;margin:8px 0 4px;padding-bottom:4px}figure{margin:0;flex:0 0 128px}
figure img{width:128px;height:96px;object-fit:cover;border-radius:3px;display:block}figcaption{font-size:11px;line-height:1.35;color:var(--mut);margin-top:3px}
.cap{font-size:13px;color:var(--mut);margin:14px 0 4px}
.panel{margin:4px 0 26px}.plabel{font-size:12px;font-weight:600;letter-spacing:.02em;color:var(--acc);margin-bottom:4px}
.finding{margin:8px 0 0;padding-left:10px;border-left:2px solid var(--acc);font-size:14px}th.n,td.n{text-align:right}.rh{font-size:13px;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:22px 0 8px;border-top:1px dashed var(--line);padding-top:14px}ol li{margin:4px 0}
"""
    title = T("Field Guide Vision · 无监督结构分析进展", "Field Guide Vision · Unsupervised Structure Update")
    html = (f"<!doctype html><html lang={'zh' if zh else 'en'}><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{css}</style></head><body><main>"
            f"<h1>{title}</h1><p class=sub>{T('2026 年 11 月 · 1,641 张 BioImages 照片 · 63 个物种 · 按个体植株划分训练 / 验证 / 测试', 'November 2026 · 1,641 BioImages photos · 63 species · train / validation / test split by individual plant')}</p>"
            f"<h3>{T('上次讲到哪', 'Where we left off')}</h3>"
            + T("<p>拍摄距离是照片空间的第一层结构：BioCLIP 主要按物种组织，DINOv3 和 EfficientNet 更受拍摄距离和器官影响。您据此提出了以下方向，本报告逐条汇报。</p>",
                "<p>Shot scale is the first layer of structure: BioCLIP is organised mainly by species, while DINOv3 and EfficientNet respond more to shot scale "
                "and organ. Based on this you suggested the directions below; this report answers them one by one.</p>")
            + img(BCV / "scale_conditioned_analysis/assets/representation_by_scale.png", "scale")
            + f"<h3>{T('总览', 'Overview')}</h3>" + overview + "".join(S)
            + f"<section><div class=hd><span class=num>9</span><h2>{T('下一步', 'Next steps')}</h2></div><ol>" + "".join(f"<li>{x}</li>" for x in nxt) + "</ol>"
            + T("<p class=cap>说明：单次数据划分；器官标签为 BioImages 单标签，拍摄距离标签为暂定。代码与全部数值在仓库 meeting_minimal/。</p>",
                "<p class=cap>Notes: one data split; organ labels are single-label BioImages categories, shot-scale labels are provisional. Code and all numbers in meeting_minimal/.</p>")
            + "</section></main></body></html>")
    (H / f"boss_report_{lang}.html").write_text(html)
    print(lang, f"{len(html) / 1e6:.1f} MB")


for lang in ("zh", "en"):
    build(lang)
