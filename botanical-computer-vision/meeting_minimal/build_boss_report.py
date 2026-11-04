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


def duo(left, right):
    """Table and its companion figure side by side (stacked on narrow screens)."""
    return f"<div class=duo><div>{left}</div><div>{right}</div></div>"


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
soft = J(R / "k7", "exp4_soft_summary.json")
e7 = soft["results"][f"K{MAIN_K}"]
main = soft["main"]
org = {c["cluster"]: c["top_organ"] for c in main["clusters"]}
short = {c["cluster"]: f"{c['top_organ']} {c['top_organ_share']:.0%}" for c in main["clusters"]}
mm = pd.read_csv(R / "k7" / "exp4_soft_memberships.csv")
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
mis = J(R, "exp8_misplaced.json")
ZH_ORG = {"leaf": "叶", "whole tree": "整树", "inflorescence": "花序", "fruit": "果实", "twig": "小枝", "bark": "树皮", "cone": "球果"}
mis_zh = "、".join(f"{ZH_ORG[p['label']]}落在{ZH_ORG[p['sits_among']]}堆里（{p['n']} 张）" for p in mis["top_pairs"][:3])
mis_en = ", ".join(f"{p['label']} among {p['sits_among']} ({p['n']})" for p in mis["top_pairs"][:3])
pc_t = pd.read_csv(R / "exp8_pc_information.csv").head(6)
pc_t["main"] = pc_t[["organ", "scale", "species"]].idxmax(axis=1)
K9 = {m: pd.read_csv(d / "exp9_k_metrics.csv") for m, d in M5.items() if (d / "exp9_k_metrics.csv").exists()}
k9e = K9["EfficientNet-B0"].set_index("K")
sil_lo, sil_hi = min(t.silhouette.min() for t in K9.values()), max(t.silhouette.max() for t in K9.values())
drv = json.loads((R / "exp10_drivers.json").read_text()) if (R / "exp10_drivers.json").exists() else None
KS4 = {m: pd.read_csv(d / "exp4b_k_sweep.csv").set_index("K") for m, d in M3.items()}
stab = {m: t.stability_ari_mean.to_dict() for m, t in KS4.items()}
k5full = json.loads((R / "exp9c_k5_groups.json").read_text())
k5sp_lo = min(r["n_species"] for r in k5full["efficientnet"]); k5sp_hi = max(r["n_species"] for r in k5full["efficientnet"])
k5top_hi = max(r["top_species_share"] for r in k5full["efficientnet"])
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
    on = lambda o: T(ZH_ORG.get(o, o), o)
    sh = lambda c: f"{on(org[c])} {short[c].split()[-1]}"
    cap_soft = lambda r: (f"{escape(species.get(r['image_id'], ''))}<br><b>{T('标签', 'label')}: {on(r['organ'])}</b><br>"
                          f"{T('候选', 'cand.')}1 {sh(r['c1'])} ({r['p1']:.2f})<br>{T('候选', 'cand.')}2 {sh(r['c2'])} ({r['p2']:.2f})")

    def sec(n, title, ask, method, tried, result, take, status):
        return (f"<section id=s{n}><div class=topic>{n} · {title}</div><h2>{CLAIM[n]}</h2><p class=aim>{ask}</p>"
                f"<div class=res>{result}</div><p class=take><b>{T('解读', 'Interpretation')}.</b> {take}</p>"
                f"<p class=meth><b>{T('方法', 'Method')}.</b> {method} <b>{T('过程', 'Procedure')}.</b> {tried}</p></section>")

    votes_t = pd.DataFrame([{T("模型", "Model"): m, **{T(a, b): J(M5[m], "exp9_k_metrics.json")["best_k"][c] for c, a, b in [
        ("silhouette", "轮廓系数", "Silhouette"), ("davies_bouldin", "Davies–Bouldin", "Davies–Bouldin"), ("calinski_harabasz", "Calinski–Harabasz", "Calinski–Harabasz"),
        ("bic", "BIC", "BIC"), ("stability", "稳定性", "Stability"), ("organ_ami", "器官一致性", "Organ agreement")]}} for m in K9])
    k5_t = pd.concat([pd.DataFrame(J(M5[m], "exp9_k_metrics.json")["k5_groups"])[["cluster", "n", "name", "scales"]].assign(model=m)
                      for m in ["EfficientNet-B0", "DINOv3"]])[["model", "cluster", "n", "name", "scales"]]
    k5_t.columns = [T("模型", "Model"), T("组", "Group"), "n", T("组成", "Make-up"), T("拍摄距离", "Shot scale")]
    mods = ["EfficientNet-B0", "DINOv3", "BioCLIP 2.5"]
    mt = json.loads((R / "exp9d_match.json").read_text())
    kmean_t = pd.DataFrame([
        ["3", T("拍摄远近", "Camera distance"), KS4["EfficientNet-B0"].loc[3, "stability_ari_mean"], KS4["DINOv3"].loc[3, "stability_ari_mean"],
         T(f"与远 / 中 / 近景对应 {mt['efficientnet']['k3_scale']:.0%} / {mt['dinov3']['k3_scale']:.0%}", f"match to distant / mid / close-up {mt['efficientnet']['k3_scale']:.0%} / {mt['dinov3']['k3_scale']:.0%}")],
        ["5", T("器官大类（最稳定）", "Organ classes (most stable)"), KS4["EfficientNet-B0"].loc[5, "stability_ari_mean"], KS4["DINOv3"].loc[5, "stability_ari_mean"],
         T("5 组对应花序、小枝、叶、整树、树皮五类", "five groups: inflorescence, twig, leaf, whole tree, bark")],
        ["7", T("器官", "Organ"), KS4["EfficientNet-B0"].loc[7, "stability_ari_mean"], KS4["DINOv3"].loc[7, "stability_ari_mean"],
         T(f"与 7 种器官对应 {mt['efficientnet']['k7_organ']:.0%} / {mt['dinov3']['k7_organ']:.0%}", f"match to 7 organs {mt['efficientnet']['k7_organ']:.0%} / {mt['dinov3']['k7_organ']:.0%}")],
        ["20", T("同一器官内的细分", "Splits within organs"), KS4["EfficientNet-B0"].loc[20, "stability_ari_mean"], KS4["DINOv3"].loc[20, "stability_ari_mean"],
         T("稳定性下降，没有对应的标签", "stability drops; no matching label")]],
        columns=["K", T("主要分出的是", "Mainly captures"), T("稳定性 EfficientNet", "Stability EfficientNet"), T("稳定性 DINOv3", "Stability DINOv3"), T("说明（对应程度：EfficientNet / DINOv3）", "Note (match: EfficientNet / DINOv3)")])
    s5k = json.loads((R / "k5" / "exp4_soft_summary.json").read_text())["main"]
    nm5 = {c["cluster"]: c for c in s5k["clusters"]}
    faces5 = "".join(f"<h4>{T('组', 'Group ')}{i + 1} · {T(ZH_ORG.get(nm5[c]['top_organ'], nm5[c]['top_organ']), nm5[c]['top_organ'])} {nm5[c]['top_organ_share']:.0%} · n = {nm5[c]['n']}</h4>"
                     + strip(s5k["faces"][str(c)], cap_photo) for i, c in enumerate(sorted(nm5, key=lambda c: -nm5[c]["n"])))
    seen, pairs_html = set(), ""
    for p_ in main["by_pair"]:
        a_, b_ = org[p_["pair"][0]], org[p_["pair"][1]]
        if a_ == b_ or (a_, b_) in seen:
            continue
        seen.add((a_, b_))
        pairs_html += (f"<h4>{T(ZH_ORG.get(a_, a_), a_)} ↔ {T(ZH_ORG.get(b_, b_), b_)} · n = {p_['n']}</h4>" + strip(p_["cases"], cap_soft))
        if len(seen) == 4:
            break
    k5full_t = pd.DataFrame([[nm, f"G{i + 1}", r["n"], f"{T(ZH_ORG.get(r['main'], r['main']), r['main'])} {r['main_share']:.0%}" + (f" + {T(ZH_ORG.get(r['second'], r['second']), r['second'])} {r['second_share']:.0%}" if r["second_share"] >= .15 else ""),
                              r["close_up"], r["distant"], r["n_species"], r["top_species_share"]]
                             for mk, nm in [("efficientnet", "EfficientNet-B0"), ("dinov3", "DINOv3")] for i, r in enumerate(k5full[mk])],
                            columns=[T("模型", "Model"), T("组", "Group"), T("照片数", "Photos"), T("主要器官", "Main organs"), T("近景", "Close-up"),
                                     T("远景", "Distant"), T("物种数", "Species"), T("最大物种占比", "Top species")])
    E, B, D = eff_f.loc["EfficientNet-B0"], eff_f.loc["BioCLIP 2.5"], eff_f.loc["DINOv3"]
    kE, kD = kt.set_index("model").loc["EfficientNet-B0"], kt.set_index("model").loc["DINOv3"]
    en = enr.set_index("model").share_clusters_dominant_ge_0_6
    nt = nsc_t.set_index("model")
    yb, yn = eff_f.loc["YOLO11s backbone"], eff_f.loc["YOLO11s neck"]
    lo3, hi3 = pct(1 - im_org["PCA-3"].max()), pct(1 - im_org["PCA-3"].min())
    CLAIM = {
        1: T(f"三维 PCA 图丢失 {lo3}–{hi3} 的器官信息；分析在 64 维 PCA 空间中进行",
             f"3-D PCA plots lose {lo3}–{hi3} of organ information; analysis is done in 64-D PCA space"),
        2: T("K = 3 对应拍摄远近，K = 5 最稳定，K = 7 对应器官；通用视觉模型的组由器官决定（排除物种因素后仍成立），约三分之一的照片处于相邻器官之间",
             "K = 3 captures camera distance, K = 5 is most stable, K = 7 matches organs; general vision models group by organ even after controlling for species, and about a third of photos lie between adjacent organs"),
        3: T("只有 EfficientNet 能用少数维度区分器官；YOLO 有清楚的结构，但不是按器官组织的",
             "Only EfficientNet tells organs apart with few dimensions; YOLO has clear structure, but not organised by organ"),
        4: T("Autoencoder 不优于 PCA；照片空间中变化最大的方向是拍摄远近，其次是器官",
             "An autoencoder does not improve on PCA; the largest variation in the photo space is camera distance, then organ"),
        5: T("组中心可由真实照片直观呈现；热图对模型可信，是否对准器官尚待检验",
             "Group centres can be shown with real photos; heatmaps are faithful to the model, alignment with organs is untested"),
        6: T(f"软聚类筛出 {n_pl} 张可能含多个器官的照片，可作为多标签标注的候选",
             f"Soft clustering shortlists {n_pl} photos that may contain several organs, as candidates for multi-label annotation"),
        7: T("YOLO 现成 embedding 的器官信息最弱，需在器官标签上训练后才适用",
             "Off-the-shelf YOLO embeddings carry the least organ information and need training on organ labels"),
    }

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
    pairs_t = pairs[["pair", "n", "share", "plausible"]].assign(pair=pairs.pair.map(lambda x: " ↔ ".join(T(ZH_ORG.get(o, o), o) for o in x.split(" ↔ ")))).rename(columns={"pair": T("在哪两种器官之间", "between which two organs"),
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
        T("<p class=lead>论证顺序：1.1 信息损失 → 1.2 损失的原因 → 1.3 各模型是否一致 → 1.4 选定分析空间 → 1.5 展示方式与用途</p>",
          "<p class=lead>Line of argument: 1.1 information loss → 1.2 why it is lost → 1.3 consistency across models → 1.4 choice of analysis space → 1.5 display and its use</p>")
        + panel(T("1.1 信息损失 · 降维后各类信息的保留比例", "1.1 Information loss · Information kept after reduction"), img(R / "exp8_info_kept.png", "information kept"),
              T("信息保留的算法：在降维后的空间里，每张照片看与它最像的 5 张照片多数是什么器官（距离、物种），以此作为判断；判断正确率除以完整空间中的正确率，100% 即没有丢失。"
                  "结果：拍摄距离在任何降维下都几乎完整保留；器官信息约需 16 维；物种信息需要 64 维。三维图中物种信息平均只剩 32%。",
                "How information kept is computed: in the reduced space, each photo takes the majority organ (scale, species) of its 5 most similar photos as its guess; "
                "the share of correct guesses divided by the same share in the full space; 100% = nothing lost. Result: shot scale survives any reduction; organ information needs about 16 dimensions; species needs 64. In a 3-D plot species information falls to 32% on average."))
        + panel(T("1.2 损失的原因 · EfficientNet 前 6 个主成分承载的信息（η²，0–1）", "1.2 Why it is lost · Information carried by EfficientNet's first 6 components (η², 0–1)"),
                duo(table(pc_t.assign(main=pc_t["main"].map(lambda v: {"organ": T("器官", "organ"), "scale": T("距离", "scale"), "species": T("物种", "species")}[v])).rename(
                    columns={"pc": "PC", "organ": T("器官", "organ"), "scale": T("距离", "scale"), "species": T("物种", "species"), "main": T("主要承载", "mainly carries")}), 2), img(R / "exp8_side_pcbars.png", "PC bars")),
                T("η² 的算法：对每个主成分，把照片按器官（或距离、物种）分组，计算“组与组之间的差别”占这个主成分全部差别的比例；1 表示该方向完全由这一标签决定，0 表示无关。"
                  "结果：前三个主成分主要区分器官（PC1 同时区分拍摄远近），物种信息分散在后面的成分中，所以只看前三维时看不到物种。",
                  "How η² is computed: for each component, photos are grouped by organ (or scale, species), and η² is the share of the component's variation that lies "
                  "between groups; 1 = the direction is fully determined by that label, 0 = unrelated. Result: the first three components mainly separate organs (PC1 also "
                  "camera distance); species information is spread over later components, so it is invisible in the first three."))
        + panel(T("1.3 一致性 · 五个模型的器官信息保留", "1.3 Consistency · Organ information kept, five models"), duo(table(im_org.rename(columns={"model": T("模型", "model")}), pct=True), img(R / "exp8_side_dumbbell.png", "dumbbell")),
                T("器官信息保留按 1.1 的方法计算（只看器官）。"
                  f"所有模型的三维图都丢掉 {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} 的器官信息；64 维时全部恢复到 {pct(im_org['PCA-64'].min())} 以上。",
                  "Organ information kept as defined in 1.1 (organ only). "
                  f"Every model's 3-D plot loses {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} of organ information; at 64 dimensions all "
                  f"recover to at least {pct(im_org['PCA-64'].min())}."))
        + panel(T("1.4 分析空间 · EfficientNet 各降维方式的完整对比", "1.4 Analysis space · All reductions compared, EfficientNet"),
                duo(table(im_t[im_t.model == "EfficientNet-B0"].drop(columns="model").rename(columns={
                    "view": T("降维方式", "view"), "neighbours_kept": T("近邻保留", "neighbours kept"), "organ_kept": T("器官信息", "organ"),
                    "scale_kept": T("距离信息", "shot scale"), "species_kept": T("物种信息", "species")}), pct=True), img(R / "exp8_side_heatmap.png", "heatmap")),
                T(f"各降维方式放在一起比较：PCA 64 维在三类信息上都接近完整，且近邻关系保留最多（{pct(effm.loc['PCA-64','neighbours_kept'])}）；二维图在任何方法下近邻都不到一半。",
                  f"All reductions side by side: 64-D PCA keeps all three kinds of information almost completely and the most neighbour relations ({pct(effm.loc['PCA-64','neighbours_kept'])}); "
                  "every 2-D view keeps fewer than half."))
        + panel(T("1.5 展示方式 · 同一批照片的三种二维画法（EfficientNet，按器官着色）", "1.5 Display · The same photos drawn three ways (EfficientNet, coloured by organ)"),
                img(R / "exp8_maps_effnet.png", "maps"),
                T(f"三张图都直接由完整向量（1,280 维）计算。二维 PCA 中叶、花序、果实、小枝混在一起；t-SNE 与 UMAP 能把器官分开，t-SNE 还保留了物种信息（{pct(effm.loc['t-SNE-2','species_kept'])}，UMAP 为 {pct(effm.loc['UMAP-2','species_kept'])}），因此展示统一用 t-SNE。"
                  f"t-SNE 图中“哪些照片挨在一起”是可信的，堆与堆的距离、位置和大小没有含义。它的用处是找出落在别的器官堆里的照片：EfficientNet 中共 {mis['n']} 张（{pct(mis['share'])}），最多的是{mis_zh}；这些照片可能拍摄特殊、标签有误或含多个器官，清单已导出供人工复核。",
                  f"All three maps are computed from the full vectors (1,280-D). In 2-D PCA leaf, inflorescence, fruit and twig overlap; t-SNE and UMAP separate organs, and t-SNE also keeps species information "
                  f"({pct(effm.loc['t-SNE-2','species_kept'])} vs {pct(effm.loc['UMAP-2','species_kept'])} for UMAP), so figures use t-SNE. On a t-SNE map, which photos sit together is reliable; "
                  f"distances, positions and sizes of groups are not. Its use is to find photos inside another organ's group: {mis['n']} for EfficientNet ({pct(mis['share'])}), most often {mis_en}. "
                  "These may be unusual shots, label errors or multi-organ photos; the list is exported for manual review.")),
        T(f"三维 PCA 图丢掉 {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} 的器官信息、保留的近邻关系不到 30%，<b>不能用来判断器官结构或聚类</b>。"
          "PCA 64 维对器官、距离、物种信息都接近完整保留，<b>后续所有分析都在 64 维中进行</b>。展示用二维图统一采用 t-SNE：图中的分组可信，但距离、簇大小和位置不具有含义。",
          f"A 3-D PCA plot loses {pct(1 - im_org['PCA-3'].max())}–{pct(1 - im_org['PCA-3'].min())} of organ information and keeps fewer than 30% of neighbour "
          "relations, so <b>it cannot be used to judge organ structure or clusters</b>. 64-D PCA keeps organ, scale and species information almost completely: "
          "<b>all further analysis runs in 64 dimensions</b>. Display figures use 2-D t-SNE: its grouping is reliable, but distances, cluster sizes and positions carry no meaning."), "done"))

    S.append(sec(2, T("聚类数 K、各组构成与软聚类", "Number of groups K, group make-up and soft clustering"),
        T("用数值确定分组数 K（代替先前随意设定的 K = 20），说明不同 K 分出的是什么结构，并用软聚类找出处于两组之间的照片。",
          "Determine the number of groups K numerically (replacing the arbitrary K = 20), show what structure each K captures, and use soft clustering to find photos lying between two groups."),
        T("K-means（无监督）在完整向量空间中分别取 K = 3–100。<b>稳定性</b>（0–1）：随机换掉一部分植株后重新分组，看分出的组是否与原来一致；越高说明这些组是数据中真实存在的结构。"
          "<b>对应程度</b>（%）：把 K 个组与拍摄距离或器官标签一一配对后，照片落在配对组里的比例；标签只用于解释，不参与分组。"
          "<b>软聚类</b>：按植株重采样 50 次、每次重新分组，照片落入各组的比例即其归属；最大归属低于 0.8 记为“处于两组之间”。",
          "K-means (unsupervised) in the full embedding space for K = 3–100. <b>Stability</b> (0–1): swap part of the plants, regroup, and check whether the groups stay "
          "the same; higher means the groups are real structure. <b>Match</b> (%): after pairing the K groups one-to-one with shot-scale or organ labels, the share of "
          "photos in the paired group; labels only interpret, never form groups. <b>Soft clustering</b>: resample plants 50 times and regroup; the share of runs a "
          "photo falls in each group is its membership; a maximum below 0.8 means the photo lies between two groups."),
        T("EfficientNet、DINOv3、BioCLIP、YOLO backbone；稳定性重采样 10 次。软聚类第一版用高斯混合模型，其“几乎没有含混照片”的结果是维度造成的假象，已作废。",
          "EfficientNet, DINOv3, BioCLIP, YOLO backbone; stability from 10 resamples. A first soft-clustering version (Gaussian mixture) reported almost no ambiguous "
          "photos; this was a dimension artefact and is withdrawn."),
        T("<p class=lead>论证顺序：2.1 K 取多少时分组可靠 → 2.2 不同 K 分出的是什么 → 2.3 各组的内容 → 2.4 各组由什么决定（富集分析）→ 2.5 处于两组之间的照片</p>",
          "<p class=lead>Line of argument: 2.1 which K gives reliable groups → 2.2 what each K captures → 2.3 what the groups contain → 2.4 what drives them (enrichment) → 2.5 photos between two groups</p>")
        + panel(T("2.1 稳定性随 K 的变化", "2.1 Stability across K"), img(R / "exp9c_stability.png", "stability"),
                T(f"分组在 K = 5–7 时最可靠（DINOv3 在 K = 5 达到 {stab['DINOv3'][5]:.2f}），K 超过 10 后明显下降。",
                  f"Groups are most reliable at K = 5–7 (DINOv3 reaches {stab['DINOv3'][5]:.2f} at K = 5) and drop clearly beyond K = 10.")
                + T("<p class=ps>注：另计算了轮廓系数、Davies–Bouldin、Calinski–Harabasz、BIC 等几何指标，它们没有给出一致的最佳 K，不足以单独决定 K。</p>",
                    "<p class=ps>Note: geometric indices (silhouette, Davies–Bouldin, Calinski–Harabasz, BIC) were also computed; they give no consistent best K and cannot decide K on their own.</p>"))
        + panel(T("2.2 不同 K 分出的是什么", "2.2 What each K captures"), table(kmean_t),
                T("K = 3 分出的是拍摄远近；K = 5 分出器官大类且最稳定；K = 7 与器官一一对应；K = 20 稳定性下降，只是同一器官内部的细分。",
                  "K = 3 separates camera distance; K = 5 separates organ classes and is the most stable; K = 7 matches organs one-to-one; K = 20 loses stability "
                  "and only splits organs further."))
        + panel(T("2.3 各组的内容：EfficientNet 在 K = 3 / 5 / 7", "2.3 What the groups contain: EfficientNet at K = 3 / 5 / 7"), img(R / "exp9d_eff_k357.png", "EfficientNet K"),
                T("K = 3 时三组为：所有近拍器官（叶、花、果、枝混在一起）、远景整树（92% 为远景）、树皮，即按拍摄远近划分；K = 5 时近拍部分再分为花序、小枝、叶三类；K = 7 时进一步分出果实，并把叶分成两组。",
                  "At K = 3 the groups are all close-up organs (leaf, flower, fruit, twig together), distant whole trees (92% distant) and bark, i.e. a split by camera "
                  "distance; at K = 5 the close-ups split into inflorescence, twig and leaf; at K = 7 fruit separates and leaves split into two groups."))
        + panel(T("2.4 各组由什么决定：富集分析（五个模型，K = 5）", "2.4 What drives the groups: enrichment (five models, K = 5)"),
                img(R / "exp10_drivers.png", "drivers") + img(R / "exp9d_models_k5.png", "models K=5"),
                T("富集倍数 = 某器官在组内的比例 ÷ 它在全部照片中的比例。判定规则：单一物种占组内 30% 以上为“物种决定”；某器官富集 ≥ 2 倍、且按组内物种构成校正后仍 ≥ 2 倍为“器官决定”；"
                  "否则远景或中景比例达到全体的 2 倍以上为“距离决定”；其余为“混合”（多为混有几种器官的近景组）。"
                  f"结果：EfficientNet 与 DINOv3 分别有 {pct(drv['EfficientNet-B0']['share']['organ'])} 与 {pct(drv['DINOv3']['share']['organ'])} 的照片落在器官决定的组中，"
                  f"排除物种因素后仍然成立（DINOv3 的混合组是近拍的花、果、枝合在一起）；YOLO 的器官决定组只占 {pct(drv['YOLO11s backbone']['share']['organ'])}（backbone）与 {pct(drv['YOLO11s neck']['share']['organ'])}（neck），其余混有多种器官；"
                  f"BioCLIP 的组与器官基本无关，它按物种组织（K = 63 时与物种对应 {pct(KS4['BioCLIP 2.5'].loc[63, 'matched_acc_species'])}）。",
                  "Fold enrichment = an organ's share within a group divided by its share across all photos. Rule: a group is species-driven if one species exceeds "
                  "30% of it; organ-driven if one organ is enriched at least 2-fold, also after adjusting for the group's species mix; otherwise distance-driven if "
                  "the distant or mid-range share is at least twice the overall share; else mixed (mostly close-up groups mixing several organs). "
                  f"Result: {pct(drv['EfficientNet-B0']['share']['organ'])} (EfficientNet) and {pct(drv['DINOv3']['share']['organ'])} (DINOv3) of photos fall in organ-driven groups, "
                  f"also after controlling for species (DINOv3's mixed group combines close-up flowers, fruit and twigs); YOLO has only {pct(drv['YOLO11s backbone']['share']['organ'])} (backbone) and {pct(drv['YOLO11s neck']['share']['organ'])} (neck), its other groups mix organs; "
                  f"BioCLIP's groups are unrelated to organ because it is organised by species (match to species at K = 63: {pct(KS4['BioCLIP 2.5'].loc[63, 'matched_acc_species'])})."))
        + panel(T("2.5 处于两组之间的照片（软聚类，EfficientNet，K = 7）", "2.5 Photos between two groups (soft clustering, EfficientNet, K = 7)"), table(pairs_t, pct=True) + pairs_html,
                T(f"软聚类用 K = 7，使每组约对应一种器官（K = 5 的组是合并后的器官大类，含混照片多半落在混合组之间，无法说明是哪两种器官）。"
                  f"{pct(e7['frac_max_lt_0_8'])} 的照片处于两组之间，最常见的是果实 ↔ 小枝、花序 ↔ 叶、果实 ↔ 整树，都是视觉上相邻、在同一画面中常一起出现的器官。",
                  f"Soft clustering uses K = 7 so that each group roughly matches one organ (K = 5 groups merge organs, so ambiguity there mostly falls between mixed groups "
                  f"and cannot name two organs). {pct(e7['frac_max_lt_0_8'])} of photos lie between two groups; the most common pairs, fruit ↔ twig, inflorescence ↔ leaf "
                  "and fruit ↔ whole tree, are visually adjacent organs that often appear in the same frame.")),
        T("K 没有唯一答案，不同的 K 揭示不同层次的结构。<b>日常分组采用 K = 5</b>（最稳定，各组有清楚的器官含义）；<b>器官层面的分析与软聚类用 K = 7</b>（一组对应一种器官）；"
          "K = 20 及以上不再使用。处于两组之间的照片集中在相邻器官之间，是多标签标注的候选（第 6 节）。",
          "K has no single answer; different K reveal different layers. <b>Use K = 5 for routine grouping</b> (most stable, clear organ meaning) and <b>K = 7 for "
          "organ-level analysis and soft clustering</b> (one group per organ); drop K = 20 and above. Photos between two groups concentrate between adjacent organs "
          "and are candidates for multi-label annotation (section 6)."), "done"))

    nb = json.loads((R / "exp1d_nsc_bars.json").read_text())
    vz = json.loads((R / "exp1b_nsc_viz.json").read_text())
    pc_name = [T("PC1 · 近拍叶 ↔ 远景整树", "PC1 · close-up leaf ↔ distant whole tree"), T("PC2 · 树皮 ↔ 整树", "PC2 · bark ↔ whole tree"),
               T("PC3 · 花序 ↔ 叶与树皮", "PC3 · inflorescence ↔ leaf and bark")]
    pcs = "".join(f"<h4>{pc_name[i]}</h4>"
                  f"<div class=ends><div><small>{T('一端', 'one end')}</small>{strip(a_['low'][:3], cap_photo)}</div>"
                  f"<div><small>{T('另一端', 'other end')}</small>{strip(a_['high'][:3], cap_photo)}</div></div>" for i, a_ in enumerate(vz["axes"][:3]))
    S.append(sec(3, T("区分器官需要多少维度：收缩质心（NSC）", "How many dimensions organs need: shrunken centroids (NSC)"),
        T("确定区分器官最少需要多少个维度，以衡量器官信息是集中在少数方向上，还是分散在许多方向上。",
          "Determine how few dimensions are needed to tell organs apart, as a measure of whether organ information is concentrated in a few directions or spread over many."),
        T("NSC（最近收缩质心，有监督，只用于解释）在 64 维 PCA 空间中为每种器官计算中心，并删去与平均差别不大的维度。"
          "<b>Δ</b> 是删除门槛：Δ 越大，删去的维度越多。<b>圆点</b>：按交叉验证，在统计上与最好结果一样好的前提下能删到的最少维度。",
          "NSC (nearest shrunken centroids; supervised, used only to explain) computes a centre for each organ in 64-D PCA space and drops dimensions in which "
          "the organ hardly differs from the average. <b>Δ</b> is the dropping threshold: larger Δ drops more dimensions. <b>Dot</b>: the fewest dimensions that "
          "are statistically as good as the best result under cross-validation."),
        T("五个模型；交叉验证按个体植株分组，测试集只评估一次。另以模型自己的 K = 7 分组为目标做同样的计算，作为对照。",
          "Five models; cross-validation grouped by individual plant, test set evaluated once. The same computation with the model's own K = 7 groups as target serves as a reference."),
        T("<p class=lead>论证顺序：3.1 各模型的器官信息有多集中 → 3.2 器官与模型自身分组的对比 → 3.3 EfficientNet 用到的维度</p>",
          "<p class=lead>Line of argument: 3.1 how concentrated organ information is → 3.2 organs versus the model's own groups → 3.3 the dimensions EfficientNet uses</p>")
        + panel(T("3.1 删减维度后的器官识别准确率（五个模型）", "3.1 Organ accuracy as dimensions are dropped (five models)"), img(R / "exp1c_nsc_organ_models.png", "NSC organ"),
                T("横轴向右 = 保留的维度越少，纵轴 = 相对于全部维度的准确率。EfficientNet 一直保持平稳，每类只需约 3 个维度即保留 90% 的准确率；其他模型从一开始就下降，器官信息分散在许多维度中。",
                  "Rightwards = fewer dimensions kept; y = accuracy relative to all dimensions. EfficientNet stays flat and keeps 90% accuracy with about 3 dimensions per class; "
                  "the other models decline from the start, so their organ information is spread over many dimensions."))
        + panel(T("3.2 区分器官 vs 复现模型自己的分组：各需多少维度", "3.2 Telling organs apart vs reproducing the model's own groups: dimensions needed"),
                img(R / "exp1d_nsc_bars.png", "NSC bars"),
                T(f"两根柱都短，说明模型有清楚的结构，且这个结构就是器官（EfficientNet，{nb['EfficientNet-B0']['organ']:.0f} vs {nb['EfficientNet-B0']['own_k7']:.0f}）。"
                  f"灰柱短、橙柱长，说明模型有清楚的结构，但不是按器官组织的：YOLO backbone 复现自身分组只需 {nb['YOLO11s backbone']['own_k7']:.0f} 个维度，区分器官却需要 {nb['YOLO11s backbone']['organ']:.0f} 个。",
                  f"Both bars short: the model has clear structure and that structure is organ (EfficientNet, {nb['EfficientNet-B0']['organ']:.0f} vs {nb['EfficientNet-B0']['own_k7']:.0f}). "
                  f"Grey short and orange long: clear structure, but not organised by organ; YOLO backbone reproduces its own groups with {nb['YOLO11s backbone']['own_k7']:.0f} "
                  f"dimensions but needs {nb['YOLO11s backbone']['organ']:.0f} to tell organs apart."))
        + panel(T("3.3 EfficientNet 区分器官用到的维度", "3.3 Dimensions EfficientNet uses to tell organs apart"),
                img(R / "exp1b_nsc_heatmap.png", "heatmap"),
                T("热图：行是器官，列是保留下来的主方向；颜色越深越重要（红、蓝表示偏向方向的两端），白色表示该器官不用这个方向。"
                  "前三个主方向承担了大部分区分：PC1 区分近拍叶与远景整树，PC2 区分树皮与整树，PC3 区分花序与叶、树皮。各方向两端的真实照片见 4.2。",
                  "Heatmap: rows are organs, columns the directions kept; darker = more important (red and blue are the two ends), white = not used for that organ. "
                  "The first three directions do most of the work: PC1 separates close-up leaf from distant whole tree, PC2 bark from whole tree, PC3 inflorescence from leaf and bark. Photos at both ends of "
                  "each direction are shown in 4.2.")),
        T("只有 EfficientNet 把器官信息集中在少数维度中，适合作为“照片 → 器官”提取的起点。YOLO 的空间有清楚的结构，但这种结构不是按器官组织的。所用维度是数学方向，不等同于植物学性状。",
          "Only EfficientNet concentrates organ information in a few dimensions, making it a good starting point for photo → organ extraction. YOLO's space has "
          "clear structure, but not organised by organ. The dimensions are mathematical directions, not botanical traits."), "done"))

    AX = {m: pd.read_csv(d / "exp5_axis_table.csv").head(3) for m, d in M3.items()}
    def pc1_reads(t):
        r = t.reading[0]
        if "~" not in r:
            return T("无明显器官对应", "no clear organ")
        lo_, hi_ = r.split("low end ~ ")[1].split(", high end ~ ")
        return f"{on(lo_.strip())} ↔ {on(hi_.strip())}"
    ax_t = pd.DataFrame([[m, f"PC{int(t.loc[t.rho_scale_close.abs().idxmax(), 'pc'])}", t.rho_scale_close.abs().max(), pc1_reads(t)] for m, t in AX.items()],
                        columns=[T("模型", "Model"), T("与拍摄远近最相关的主轴", "Axis most related to camera distance"), T("相关强度 |ρ|", "Strength |ρ|"),
                                 T("PC1 区分的是", "What PC1 separates")])
    ends3 = "".join(f"<h4>{nm}</h4><div class=ends><div><small>{T('一端', 'one end')}</small>{strip(s5['ends'][str(k)]['low'][:3], cap_photo)}</div>"
                    f"<div><small>{T('另一端', 'other end')}</small>{strip(s5['ends'][str(k)]['high'][:3], cap_photo)}</div></div>"
                    for k, nm in [(1, T("PC1 · 近拍叶 ↔ 远景整树", "PC1 · close-up leaf ↔ distant whole tree")), (2, T("PC2 · 树皮 ↔ 整树", "PC2 · bark ↔ whole tree")),
                                  (3, T("PC3 · 花序 ↔ 叶与树皮", "PC3 · inflorescence ↔ leaf and bark"))])
    aer = ae.set_index(["model", "method"])
    S.append(sec(4, T("Autoencoder 与主轴的含义", "Autoencoder and the meaning of the main axes"),
        T("检验非线性降维（Autoencoder）能否比 PCA 保留更多信息，并说明 PCA 主要方向（主轴）各代表什么。",
          "Test whether non-linear reduction (autoencoder) keeps more information than PCA, and state what the main PCA directions (axes) represent."),
        T("<b>Autoencoder</b>（无监督）：一个小型神经网络，先把向量压缩到 32 维再还原（输入 → 128 → 32 → 128 → 输入，ReLU 激活），"
          "在 1,181 张训练照片上训练 40 轮（Adam，学习率 0.001），在 211 张测试照片上与同为 32 维的 PCA 比较两项：还原误差，以及近邻关系保留了多少。"
          "<b>主轴两端</b>：把所有照片按某一主方向上的得分排序，取两端的照片；并计算主轴得分与拍摄远近的相关（|ρ|，0–1）。",
          "<b>Autoencoder</b> (unsupervised): a small neural network that compresses each vector to 32 dimensions and reconstructs it (input → 128 → 32 → 128 → "
          "input, ReLU), trained for 40 epochs on 1,181 training photos (Adam, learning rate 0.001) and compared on 211 test photos with 32-D PCA on two counts: "
          "reconstruction error and neighbour relations kept. <b>Ends of an axis</b>: sort all photos by their score on one principal direction and take the "
          "photos at both ends; also correlate the score with camera distance (|ρ|, 0–1)."),
        T("AE 结果来自上一阶段，并以线性 AE 复核：线性 AE 与 PCA 一致，说明代码与数据对齐。", "AE results come from the previous stage and were checked with a linear AE, which matches PCA, so code and data line up."),
        T("<p class=lead>论证顺序：4.1 非线性降维是否更好 → 4.2 主轴代表什么</p>", "<p class=lead>Line of argument: 4.1 whether non-linear reduction is better → 4.2 what the axes represent</p>")
        + panel(T("4.1 Autoencoder 与 PCA（32 维，测试照片）", "4.1 Autoencoder vs PCA (32 dimensions, test photos)"),
                duo(table(ae.assign(knn15_retention=ae.knn15_retention.map(lambda v: f"{v:.0%}"), model=ae.model.map({"bioclip25": "BioCLIP 2.5", "dinov3": "DINOv3", "efficientnet_b0": "EfficientNet-B0"})).rename(columns={
                    "model": T("模型", "Model"), "method": T("方法", "Method"), "test_mse": T("还原误差", "Reconstruction error"),
                    "knn15_retention": T("近邻保留", "Neighbours kept")}), 6), img(R / "exp5b_ae.png", "AE")),
                T(f"三个模型中，Autoencoder 的还原误差与 PCA 几乎相同，近邻保留反而低 1–5 个百分点（EfficientNet："
                  f"{aer.loc[('efficientnet_b0', 'Autoencoder'), 'knn15_retention']:.0%} vs {aer.loc[('efficientnet_b0', 'PCA'), 'knn15_retention']:.0%}）。",
                  f"For all three models the autoencoder's reconstruction error matches PCA, and it keeps 1–5 points fewer neighbours (EfficientNet: "
                  f"{aer.loc[('efficientnet_b0', 'Autoencoder'), 'knn15_retention']:.0%} vs {aer.loc[('efficientnet_b0', 'PCA'), 'knn15_retention']:.0%})."))
        + panel(T("4.2 主轴代表什么：EfficientNet 前三个主轴两端的照片，以及各模型中与拍摄远近最相关的主轴", "4.2 What the axes represent: EfficientNet's first three axes, and the axis most related to camera distance per model"),
                ends3 + table(ax_t, 2),
                T("EfficientNet 的第一主轴就是拍摄远近（一端近拍叶，另一端远景整树），第二主轴区分树皮与整树，第三主轴一端是花序、另一端是叶与树皮。"
                  "DINOv3 同样在第一、二主轴上体现拍摄远近；BioCLIP 到第三主轴才出现，与其按物种组织的结构一致。",
                  "EfficientNet's first axis is camera distance (close-up leaf at one end, distant whole tree at the other); the second separates bark from whole tree, "
                  "the third inflorescence from leaf and bark. DINOv3 also shows camera distance on its first two axes; in BioCLIP it appears only on the third, consistent "
                  "with its species-based structure.")),
        T("非线性降维没有带来收益，<b>继续使用 PCA，Autoencoder 方向不再推进</b>。照片空间中变化最大的方向是拍摄远近，其次才是器官；比较照片时需要考虑拍摄距离的影响。",
          "Non-linear reduction brings no gain: <b>keep PCA and close the autoencoder line</b>. The largest variation in the photo space is camera distance, then "
          "organ; comparisons between photos should take shot distance into account."), "done"))

    S.append(sec(5, T("从隐空间回到图片", "Back from the latent space to images"),
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

    S.append(sec(6, T("每张图多个标签", "Multiple labels per image"),
        T("利用软聚类结果，筛选可能含有多个器官、需要多标签标注的照片。", "Use soft-clustering results to shortlist photos that may contain several organs and need multi-label annotation."),
        T("目前器官标签是单标签。软聚类中，照片落在自己器官的组、但第二候选是另一种器官，记为“两个器官都合理”，作为多标签候选。",
          "Organ labels are single-label today. In soft clustering, a photo in its own organ's group whose second candidate is another organ is marked "
          "\"both organs plausible\": a multi-label candidate."),
        T(f"EfficientNet K={MAIN_K}：{n_pl} 张候选（占含混照片 {pct(n_pl / n_amb)}）。软聚类用重采样共识得到（第一版高斯混合模型的结果为维度假象，已作废）。",
          f"EfficientNet K={MAIN_K}: {n_pl} candidates ({pct(n_pl / n_amb)} of ambiguous photos). Soft membership from bootstrap consensus (the first Gaussian-mixture version was a dimension artefact and is withdrawn)."),
        panel(T("6.1 “两个器官都合理”的候选照片", "6.1 \"Both organs plausible\" candidates"), strip(main["by_category"]["both_plausible"], cap_soft),
                T("每张照片标出自身标签与两个候选组的主要器官；画面中是否确有两种器官需人工确认。",
                  "Each photo shows its own label and the main organ of both candidate groups; whether two organs are visible needs manual confirmation.")),
        T("候选清单已经可以交给标注；要确认画面里是否真有两种器官，需要导出多标签数据（Human Gold / Gemma 多标签）后验证。",
          "The candidate list is ready for annotation; confirming two organs in the frame needs the multi-label data (Human Gold / Gemma multi-label) exported."), "next"))

    yt = pd.DataFrame([{"model": m, T("器官 F1", "organ F1"): eff_f.loc[m, "organ_f1"], T("物种 F1", "species F1"): eff_f.loc[m, "species_f1"],
                        T("NSC 器官每类坐标", "NSC organ coords / class"): nt.loc[m, "per_class"],
                        T(f"K={MAIN_K} 单一器官主导的组", f"K={MAIN_K} organ-dominated groups"): en[m]} for m in M5])
    S.append(sec(7, T("YOLO 对比（团队新方向）", "YOLO comparison (team direction)"),
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
.topic{font-size:12px;font-weight:600;letter-spacing:.04em;color:var(--acc);text-transform:uppercase;margin-bottom:4px}
.aim{color:var(--mut);margin:6px 0 16px}.meth{font-size:12.5px;color:var(--mut);margin:12px 0 0;line-height:1.55}
.abs{background:var(--soft);padding:14px 18px;border-radius:4px;margin:8px 0 10px}.ch{font-size:22px;margin:46px 0 10px;padding-bottom:6px;border-bottom:2px solid var(--fg)}
.chn{color:var(--acc);margin-right:6px}.setup{font-size:13.5px;color:var(--mut);margin:0 0 14px}.kf{margin:8px 0 0;padding-left:20px}.kf li{margin:3px 0}.kl{color:inherit;text-decoration:none}.kl:hover{text-decoration:underline}section h2{font-size:18px;line-height:1.35}
.panel{margin:4px 0 26px}.ends{display:grid;grid-template-columns:1fr 1fr;gap:14px}.ends>div{min-width:0}.ends small{color:var(--mut);font-size:11px}@media(max-width:700px){.ends{grid-template-columns:1fr}}.lead{margin:0 0 18px;font-size:14px;color:var(--mut)}.ps{font-size:12px;color:var(--mut);margin:6px 0 0}.plabel{font-size:12px;font-weight:600;letter-spacing:.02em;color:var(--acc);margin-bottom:4px}
.finding{margin:8px 0 0;padding-left:10px;border-left:2px solid var(--acc);font-size:14px}th.n,td.n{text-align:right}.duo{display:grid;grid-template-columns:auto minmax(0,300px);gap:24px;align-items:start}
.duo .fig{margin:10px 0;max-width:300px}@media(max-width:760px){.duo{grid-template-columns:1fr}}.rh{font-size:13px;font-weight:600;color:var(--mut);text-transform:uppercase;letter-spacing:.04em;margin:22px 0 8px;border-top:1px dashed var(--line);padding-top:14px}ol li{margin:4px 0}
"""
    title = T("Field Guide Vision：植物照片 embedding 的无监督结构分析", "Field Guide Vision: Unsupervised Structure of Plant-Photo Embeddings")
    models_t = pd.DataFrame([[ "EfficientNet-B0", T("ImageNet 通用图像", "ImageNet, general images"), "1,280"],
                             ["BioCLIP 2.5", T("生物图像-文本（物种）", "Biological image-text (species)"), "1,024"],
                             ["DINOv3", T("自监督通用图像", "Self-supervised, general images"), "768"],
                             ["YOLO11s backbone / neck", T("COCO 物体检测", "COCO object detection"), "512 / 896"]],
                            columns=[T("模型", "Model"), T("预训练数据", "Pre-training"), T("维度", "Dimensions")])
    U, S_, E_ = T("无监督", "unsupervised"), T("有监督", "supervised"), T("描述 / 评估", "descriptive / evaluation")
    meth_t = pd.DataFrame([["PCA", U, T("降维、主轴解读", "Reduction, axis reading"), "1, 4"],
                           ["t-SNE / UMAP", U, T("仅用于展示与检查", "Display and inspection only"), "1"],
                           ["K-means", U, T("分组（K = 3–20，推荐 K = 5）", "Grouping (K = 3–20, K = 5 recommended)"), "2"],
                           [T("重采样共识软归属", "Bootstrap-consensus soft membership"), U, T("照片在各组间的归属", "Membership across groups"), "2, 6"],
                           ["Autoencoder", U, T("非线性降维对照", "Non-linear reduction baseline"), "4"],
                           [T("最近邻检索", "Nearest-neighbour retrieval"), U, T("隐空间 → 真实照片", "Latent point → real photos"), "5"],
                           [T("富集分析", "Enrichment"), E_, T("各组构成", "Group composition"), "2"],
                           [T("近邻投票、线性探针", "Neighbour vote, linear probe"), S_, T("评估信息保留", "Evaluating information kept"), "1, 7"],
                           ["NSC", S_, T("解释哪些维度起作用", "Explaining which dimensions matter"), "3"],
                           ["Grad-CAM", T("基于模型梯度", "Model-gradient based"), T("定位依赖的图像区域", "Locating image regions used"), "5"]],
                          columns=[T("方法", "Method"), T("类型", "Type"), T("用途", "Purpose"), T("章节", "Section")])
    aims_t = pd.DataFrame([[f"4.{n}", a, c, st_lbl[st]] for n, a, b_, c, st in rows],
                          columns=[T("章节", "Section"), T("主题", "Topic"), T("主要结果", "Main result"), T("状态", "Status")])
    abstract = T(f"我们用 1,641 张 BioImages 照片（63 个物种）检验了五种预训练模型的 embedding 空间在无监督条件下如何组织植物照片。三维 PCA 图会丢失 {lo3}–{hi3} 的器官信息，"
                 f"因此分析在 64 维中进行。K = 5–7 时分组最稳定且与器官对应；通用视觉模型（EfficientNet、DINOv3）按器官分组，BioCLIP 按物种分组。约 {pct(e7['frac_max_lt_0_8'])} 的照片处于两组之间，"
                 f"其中 {n_pl} 张可作为多器官标注候选。YOLO 的现成 embedding 器官信息最弱。",
                 f"We examined how the embedding spaces of five pre-trained models organise 1,641 BioImages photos (63 species) without supervision. 3-D PCA plots "
                 f"lose {lo3}–{hi3} of organ information, so analysis runs in 64 dimensions. Groupings are most stable and organ-aligned at K = 5–7; general vision models "
                 f"(EfficientNet, DINOv3) group by organ and BioCLIP by species. About {pct(e7['frac_max_lt_0_8'])} of photos lie between two groups, {n_pl} of which are "
                 "candidates for multi-organ annotation. Off-the-shelf YOLO embeddings carry the least organ information.")
    disc = T([f"<b>意义</b>：器官结构可以在无监督条件下从通用视觉模型的 embedding 中读出，适合作为“照片 → 器官 / 性状 → 知识图谱推理”的第一步；BioCLIP 更适合物种层面。",
              "<b>可视化</b>：二维、三维图只用于展示，所有结论基于 64 维计算。",
              "<b>局限</b>：单次数据划分；器官标签为单标签；拍摄距离标签为暂定；热图未与器官掩膜比对；结论尚未在外部野外照片上复验。"],
             ["<b>Implication</b>: organ structure can be read without supervision from general vision embeddings, a suitable first step for \"photo → organ / trait → "
              "knowledge-graph reasoning\"; BioCLIP fits the species level better.", "<b>Visualisation</b>: 2-D and 3-D plots are for display only; every conclusion is computed in 64 dimensions.",
              "<b>Limitations</b>: one data split; single-label organ annotation; provisional shot-scale labels; heatmaps not compared with organ masks; "
              "no external field-photo validation yet."])
    concl = T(["分析空间统一为 64 维 PCA；展示图用 t-SNE。", "聚类数取 K = 5–7（与器官数相符、最稳定）。",
               "器官结构分析优先使用 DINOv3 / EfficientNet；物种层面使用 BioCLIP。", "YOLO 需先在器官标签上训练，再纳入同一套分析。"],
              ["Use 64-D PCA as the analysis space; t-SNE for display.", "Use K = 5–7 (matches the number of organs, most stable).",
               "Use DINOv3 / EfficientNet for organ structure and BioCLIP for species.", "Train YOLO on organ labels before including it in the same analyses."])
    keyf = "".join(f"<li><a class=kl href=#s{n}>{CLAIM[n]}</a></li>" for n in range(1, 8))
    html = (f"<!doctype html><html lang={'zh' if zh else 'en'}><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{title}</title><style>{css}</style></head><body><main>"
            f"<h1>{title}</h1><p class=sub>{T('组会进展汇报 · 2026 年 11 月', 'Group meeting update · November 2026')}</p>"
            + T("<p class=setup>数据：1,641 张 BioImages 照片，63 个物种，465 株个体；训练 / 验证 / 测试按个体植株划分（1,181 / 249 / 211）。模型：EfficientNet-B0、BioCLIP 2.5、DINOv3、YOLO11s（backbone / neck），均冻结使用。"
                "标签（器官、拍摄距离、物种）只用于解释与评估，不参与无监督分组。</p>",
                "<p class=setup>Data: 1,641 BioImages photos, 63 species, 465 individual plants; train / validation / test split by individual plant (1,181 / 249 / 211). "
                "Models: EfficientNet-B0, BioCLIP 2.5, DINOv3, YOLO11s (backbone / neck), all frozen. Labels (organ, shot scale, species) are used only to explain and "
                "evaluate, never to form the unsupervised groups.</p>")
            + f"<div class=abs><b>{T('本期主要发现', 'Key findings')}</b><ol class=kf>{keyf}</ol></div>"
            + "".join(S)
            + f"<section><div class=topic>{T('下一步', 'Next steps')}</div><h2>{T('待决事项与后续工作', 'Decisions and follow-up')}</h2>"
            + f"<p class=aim>{T('建议决定', 'Proposed decisions')}</p><ol>" + "".join(f"<li>{x}</li>" for x in concl) + "</ol>"
            + f"<p class=aim>{T('后续工作', 'Follow-up work')}</p><ol>" + "".join(f"<li>{x}</li>" for x in nxt) + "</ol></section>"
            + f"<section><div class=topic>{T('附录', 'Appendix')}</div><h2>{T('方法一览', 'Methods at a glance')}</h2>" + table(meth_t)
            + T("<p class=cap>代码、配置与全部数值见仓库 botanical-computer-vision/meeting_minimal/。</p>",
                "<p class=cap>Code, configuration and all numbers: botanical-computer-vision/meeting_minimal/.</p>")
            + "</section></main></body></html>")
    (H / f"boss_report_{lang}.html").write_text(html)
    print(lang, f"{len(html) / 1e6:.1f} MB")


for lang in ("zh", "en"):
    build(lang)
