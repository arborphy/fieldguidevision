"""Cross-model table from results/ (EfficientNet) and results/{bioclip,dinov3}/ (vector-only experiments)."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
DIRS = {"BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3", "EfficientNet-B0": R}
SITE_PCA32_MSE = {"BioCLIP 2.5": 0.000284, "DINOv3": 0.000498, "EfficientNet-B0": 0.000296}  # hosted-site AE table


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
            "GMM max<0.6": s4["frac_images_max_membership_lt_0_6"],
            "GMM seed ARI": s4["seed_label_ari_mean"],
            "entropy AUROC mid-range": s4["auroc_entropy_vs_midrange_scale"],
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
           "BioCLIP and DINOv3 organ information stays spread over all 64 components. (5) Soft GMM membership is near-hard for all three (≤1.3% images "
           "with max membership &lt;0.6). (6) Retrieved centroid neighbours show the cluster's dominant organ 90–91% of the time for DINOv3 / "
           "EfficientNet but 50% for BioCLIP, whose clusters are not organ clusters. Single split, single seed per setting; descriptive where noted.")
NOTE_ZH = ("解读：(1) 恢复的向量就是之前用的那批：三个模型的 PCA-32 test MSE 与网站上 AE 表的数值在 6 位小数上一致，线性 AE 与 PCA 吻合（比值≈1）。"
           "(2) 三个模型的 2D 都只保留 10–14% 的 15 近邻；只有 BioCLIP 在 256 维时满足 d* 规则。"
           "(3) 拍摄尺度轴在 EfficientNet（ρ=−0.65）和 DINOv3（ρ=−0.51）里是 PC1，在 BioCLIP 里到 PC3 才出现（ρ=−0.45；PC1–2 的 |ρ|≤0.12），与 BioCLIP 以物种为先一致。"
           "(4) NSC 只在 EfficientNet 上能把器官任务稀疏化（22/64 个坐标）；BioCLIP 和 DINOv3 的器官信息分散在全部 64 个主成分上。"
           "(5) 三个模型的 GMM 软归属都接近硬分配（最大归属 &lt;0.6 的图片 ≤1.3%）。"
           "(6) 簇中心检索出的近邻中，主导器官占比在 DINOv3 / EfficientNet 为 90–91%，在 BioCLIP 只有 50%，因为 BioCLIP 的簇不是器官簇。"
           "单次划分，每个设置单个种子；标注为描述性的部分只作描述。")
