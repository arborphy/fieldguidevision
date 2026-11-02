"""Cross-model table from results/ (EfficientNet) and results/{bioclip,dinov3}/ (vector-only experiments)."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
DIRS = {"BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3", "EfficientNet-B0": R, "YOLO": R / "yolo"}
SITE_PCA32_MSE = {"BioCLIP 2.5": 0.000284, "DINOv3": 0.000498, "EfficientNet-B0": 0.000296, "YOLO": float("nan")}  # hosted-site AE table


def rows():
    out = []
    for name, d in DIRS.items():
        try:
            J = lambda n: json.loads((d / n).read_text())
            s1, s4, s5, s6, s7 = (J(f) for f in ["exp1_nsc_summary.json", "exp4_soft_summary.json", "exp5_axes_summary.json",
                                                  "exp6_summary.json", "exp7_retrieval_summary.json"])
        except FileNotFoundError:
            continue
        t6 = {str(r["d"]): r for r in s6["table"]}
        org = s1["results"]["organ (external label)"]
        out.append({
            "model": name,
            "2D 15-NN (all 1,641)": s6["two_d_all_images"]["knn15_overlap_all1641"],
            "64D 15-NN (val)": t6["64"]["val_knn15_overlap"],
            "d*": s6["d_star"] if s6["d_star"] else ">256",
            "species probe F1 (val, full)": t6["full"]["probe_species_macroF1"],
            "organ probe F1 (val, full)": t6["full"]["probe_organ_macroF1"],
            "NSC organ coords @1-SE": org["test"]["delta_1se"]["coords_any_class"],
            "NSC organ F1 1-SE/Δ0": org["ratio_F1_1se_over_delta0"],
            "K=7 consensus max<0.8": s4["results"]["K7"]["frac_max_lt_0_8"],
            "K=7 entropy split-half ρ": s4["results"]["K7"]["entropy_split_half_spearman"],
            "K=7 entropy AUROC error": s4["results"]["K7"]["auroc_entropy_species_probe_error_test"],
            "PC1 ρ scale": s5["axes"][0]["rho_scale_close"],
            "PCA-32 test MSE (site)": f'{s5["linear_ae_sanity"]["pca32_test_mse"]:.6f} ({SITE_PCA32_MSE[name]:.6f})',
            "linear AE / PCA": s5["linear_ae_sanity"]["ratio"],
            "centroid top-6 dominant organ": s7["mean_top6_share_dominant_organ"],
        })
    return pd.DataFrame(out)


def axes(name):
    d = DIRS[name]
    return pd.read_csv(d / "exp5_axis_table.csv")


NOTE_EN = ("Reading: (1) The recovered vectors are the ones used before: PCA-32 test MSE reproduces the hosted AE table to 6 decimals for all three "
           "models, and the linear AE matches PCA (ratio ≈ 1). (2) 2-D keeps 10–14% of 15 nearest neighbours for every model; only BioCLIP reaches the "
           "d* rule, at 256-D. (3) The shot-scale axis is PC1 in EfficientNet (ρ=−0.65) and DINOv3 (ρ=−0.51) but only PC3 in BioCLIP (ρ=−0.45; PC1–2 "
           "|ρ|≤0.12), consistent with BioCLIP being species-first. (4) NSC sparsifies the organ task only in EfficientNet (22/64 coordinates); in "
           "BioCLIP and DINOv3 organ information stays spread over all 64 components. (5) Soft membership (bootstrap consensus, K=7) is ambiguous for 32–57% of "
           "images (max membership &lt;0.8) and repeatable (split-half ρ 0.84–0.90), but does not predict species errors; the earlier "
           "\"near-hard\" GMM result was a dimension artefact (item 4). (6) Retrieved centroid neighbours show the cluster's dominant organ 90–91% of the time for DINOv3 / "
           "EfficientNet but 50% for BioCLIP, whose clusters are not organ clusters. Single split, single seed per setting; descriptive where noted.")
NOTE_ZH = ("解读：(1) 恢复的向量就是之前用的那批：三个模型的 PCA-32 test MSE 与网站上 AE 表的数值在 6 位小数上一致，线性 AE 与 PCA 吻合（比值≈1）。"
           "(2) 三个模型的 2D 都只保留 10–14% 的 15 近邻；只有 BioCLIP 在 256 维时满足 d* 规则。"
           "(3) 拍摄尺度轴在 EfficientNet（ρ=−0.65）和 DINOv3（ρ=−0.51）里是 PC1，在 BioCLIP 里到 PC3 才出现（ρ=−0.45；PC1–2 的 |ρ|≤0.12），与 BioCLIP 以物种为先一致。"
           "(4) NSC 只在 EfficientNet 上能把器官任务稀疏化（22/64 个坐标）；BioCLIP 和 DINOv3 的器官信息分散在全部 64 个主成分上。"
           "(5) 软归属（bootstrap 共识，K=7）：32–57% 的照片最大归属 &lt;0.8，且可重复（两半运行 ρ 0.84–0.90），但不能预测认错物种；之前“几乎全是硬分配”的 GMM 结果是维度假象（见第 4 项）。"
           "(6) 簇中心检索出的近邻中，主导器官占比在 DINOv3 / EfficientNet 为 90–91%，在 BioCLIP 只有 50%，因为 BioCLIP 的簇不是器官簇。"
           "单次划分，每个设置单个种子；标注为描述性的部分只作描述。")


def nsc_details(lang, strip, table):
    """Collapsible block for item 1: 3-model NSC curve, EfficientNet organ fingerprint, real images per kept PC."""
    p = R / "exp1b_nsc_viz.json"
    if not p.exists():
        return ""
    v = json.loads(p.read_text())
    zh = lang == "zh"
    rows = "".join(
        f"<h4>PC{a['pc']} · {'用于' if zh else 'used by'}: {a['used_by']} · {a['reading']} · "
        f"ρ {'尺度' if zh else 'scale'} {a['rho_scale_close']:.2f}, ρ {'亮度' if zh else 'brightness'} {a['rho_brightness']:.2f}</h4>"
        f"<div class=ends><div><small>− {'端' if zh else 'end'}</small>{strip(a['low'])}</div>"
        f"<div><small>+ {'端' if zh else 'end'}</small>{strip(a['high'])}</div></div>" for a in v["axes"])
    t = (("三模型对比：器官任务的准确率—稀疏度曲线", "EfficientNet 保留的 22 个维度长什么样",
          "热图：行是器官，列是 NSC 保留的主方向；红色 = 该器官偏向这个方向的正端，蓝色 = 偏向负端，白色 = 对该器官不重要（被压成 0）。"
          "下面每行是一个方向两端各 4 张真实照片。只有前 7 个方向与器官有明显相关（|ρ|≥0.2），其余方向只对个别器官有用，含义需要看照片判断。"
          "BioCLIP / DINOv3 在 1-SE Δ 下全部 64 个维度都保留，因此只画曲线、不画热图。方向是数学方向，名称由照片和相关性推断。")
         if zh else
         ("3-model comparison: accuracy vs sparsity, organ task", "What the 22 EfficientNet coordinates look like",
          "Heatmap: rows = organs, columns = PCs kept by NSC; red = organ sits at the + end, blue = − end, white = dropped for that organ. "
          "Below, each row shows 4 real images at each end of one PC. Only the first 7 PCs correlate clearly with an organ (|ρ|≥0.2); the rest matter "
          "for one or two organs and have to be read from the images. BioCLIP / DINOv3 keep all 64 coordinates at the 1-SE Δ, so they get the curve but "
          "no heatmap. PCs are mathematical directions; names are inferred from images and correlations."))
    return (f"<details><summary>{t[0]}</summary><img src='results/exp1b_nsc_3models.png' alt='NSC 3 models'></details>"
            f"<details><summary>{t[1]}</summary><p>{t[2]}</p><img src='results/exp1b_nsc_heatmap.png' alt='NSC heatmap'>{rows}</details>")
