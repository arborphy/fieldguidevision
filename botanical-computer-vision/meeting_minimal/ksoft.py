"""Section 4 of both reports: K choice (a few meaningful K values) + redone soft clustering (bootstrap consensus)."""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
MODELS = {"EfficientNet-B0": R, "BioCLIP 2.5": R / "bioclip", "DINOv3": R / "dinov3", "YOLO": R / "yolo"}
SHOW_K = [3, 7, 20, 63]


def _models():
    return {n: d for n, d in MODELS.items() if (d / "exp4b_k_sweep.csv").exists() and (d / "exp4_soft_summary.json").exists()}


def section4(lang, section, table, strip):
    zh = lang == "zh"
    ms = _models()
    k = pd.concat([pd.read_csv(d / "exp4b_k_sweep.csv").assign(model=n) for n, d in ms.items()])
    k = k[k.K.isin(SHOW_K)]
    match = k.apply(lambda r: {3: f"{r.matched_acc_scale:.2f}", 7: f"{r.matched_acc_organ:.2f}",
                               63: f"{r.matched_acc_species:.2f}"}.get(r.K, "—"), axis=1)
    kt = pd.DataFrame({"model": k.model, "K": k.K,
                       ("K 的含义" if zh else "why this K"): k.K.map(
                           {3: "3 种拍摄距离" if zh else "3 shot scales", 7: "7 种器官" if zh else "7 organs",
                            20: "旧默认值" if zh else "old default", 63: "63 个物种" if zh else "63 species"}),
                       "AMI organ": k.ami_organ, "AMI scale": k.ami_scale, "AMI species": k.ami_species,
                       ("一一对应准确率" if zh else "matched accuracy"): match,
                       ("稳定性 ARI" if zh else "stability ARI"): k.stability_ari_mean})
    S = {n: json.loads((d / "exp4_soft_summary.json").read_text()) for n, d in ms.items()}
    rows = []
    for n, s in S.items():
        for kk, v in s["results"].items():
            rows.append({"model": n, "K": kk[1:], "max<0.8": v["frac_max_lt_0_8"], "max<0.6": v["frac_max_lt_0_6"],
                         ("熵可重复性" if zh else "entropy split-half ρ"): v["entropy_split_half_spearman"],
                         ("熵~边界距离 ρ" if zh else "entropy ~ boundary ρ"): v["entropy_vs_d1_d2_spearman"],
                         ("含混且两候选器官不同" if zh else "ambiguous between different-organ clusters"): v["ambiguous_between_different_organ_clusters"],
                         "AUROC error": v["auroc_entropy_species_probe_error_test"], "AUROC mid-range": v["auroc_entropy_midrange"]})
    st = pd.DataFrame(rows)
    art = pd.DataFrame(S["EfficientNet-B0"]["gmm_dimension_artifact"])
    cases = S["EfficientNet-B0"]["cases"]["K7"]
    cap = lambda r: f"{r['organ']}·{r['shot_scale']} | c{r['c1']} {r['p1']:.2f} ({r['c1_organ']}) / c{r['c2']} {r['p2']:.2f} ({r['c2_organ']})"
    ef7, ef20 = S["EfficientNet-B0"]["results"]["K7"], S["EfficientNet-B0"]["results"]["K20"]
    if zh:
        return section(4, "K 的选择 ＋ 软聚类（重做）",
            "K 应该取多少，不同 K 分别对应什么？有没有照片真的夹在两组之间？之前的软聚类哪里不对？",
            "<b>K</b>：不再固定 K=20，而是挑几个有含义的值：3（拍摄距离有 3 种）、7（器官有 7 种）、20（旧默认）、63（物种数），"
            "在每个模型的完整向量上跑 K-means，看分组和器官 / 尺度 / 物种的一致程度（AMI，已扣除随 K 增大而虚高的部分），"
            "以及 K 等于标签数时的一一对应准确率，并按个体植株重采样 10 次检验稳定性。"
            "<br><b>之前的软聚类错在哪</b>：对角高斯混合模型的归属概率是各维似然相乘，维度越高越极端。同一数据、同样 K=20，"
            "2 维时 22% 的照片含混，32 维时只剩 1.3%（见下表）。所以“几乎没有含混照片”是选了 32 维造成的，不是照片的性质；K=20 也没有依据。"
            "<br><b>重做</b>：按个体植株重采样 50 次、每次重新聚类，照片落入各组的比例就是它的软归属（不依赖维度或温度参数）。K 取 7 和 20。",
            "<p>几个 K 的结果（完整的 3–100 扫描在 results/*/exp4b_k_sweep.csv）：</p>" + table(kt, d=2) +
            "<p>旧方法的维度假象（EfficientNet，对角 GMM，K=20）：</p>" + table(art, d=3) +
            "<p>新方法：bootstrap 共识软归属</p>" + table(st, d=2) +
            "<p>EfficientNet K=7 下最含混的 8 张真实照片（两个候选组及其主导器官）：</p>" + strip(cases, cap=cap),
            "<b>K</b>：EfficientNet / DINOv3 在 K=3 时与拍摄距离一一对应约 80% / 77%，K=7 时与器官对应 68% / 58%，器官一致性在 K=5–10 最高；"
            "BioCLIP 在 K=63 时与物种对应 65%，几乎不按器官或距离分。K=20 不是特殊值：K≥30 后稳定性降到 0.4–0.5。"
            f"<br><b>软聚类</b>：含混是真实存在且可重复的。EfficientNet K=7 下 {ef7['frac_max_lt_0_8']:.0%} 的照片最大归属 &lt;0.8，"
            f"两半独立运行的熵相关 {ef7['entropy_split_half_spearman']:.2f}；其中 {ef7['ambiguous_between_different_organ_clusters']:.0%} 夹在两个不同器官的组之间"
            f"（K=20 时只有 {ef20['ambiguous_between_different_organ_clusters']:.0%}，含混变成同一器官内部的细分）。"
            "含混<b>不能</b>预测认错物种（AUROC 0.32–0.50），对中距离照片只有弱信号。<b>软归属不等于多器官</b>：多器官标签尚未导出，无法检验。",
            ["v", "e", "x"])
    return section(4, "Choice of K + soft clustering (redone)",
        "Which K, and what does each K mean? Are some photos really between two groups? What was wrong with the first soft clustering?",
        "<b>K</b>: instead of a fixed K=20, a few meaningful values: 3 (three shot scales), 7 (seven organs), 20 (old default), 63 (species). "
        "K-means on each model's full vectors; agreement with organ / scale / species (AMI, corrected for the chance level that rises with K), "
        "one-to-one matched accuracy when K equals the number of labels, and stability over 10 resamples of individual plants."
        "<br><b>What was wrong</b>: a diagonal Gaussian mixture multiplies per-dimension likelihoods, so memberships get more extreme as "
        "dimensions grow. Same data, same K=20: 22% of photos ambiguous in 2-D, 1.3% in 32-D (table below). \"Almost no ambiguous photos\" "
        "came from choosing 32-D, not from the photos; K=20 had no justification either."
        "<br><b>Redo</b>: resample individual plants 50 times and recluster each time; the share of runs in which a photo lands in each "
        "group is its soft membership (no dimension or temperature dependence). K = 7 and 20.",
        "<p>Selected K values (full 3–100 scan in results/*/exp4b_k_sweep.csv):</p>" + table(kt, d=2) +
        "<p>Dimension artefact of the old method (EfficientNet, diagonal GMM, K=20):</p>" + table(art, d=3) +
        "<p>New method: bootstrap consensus membership</p>" + table(st, d=2) +
        "<p>The 8 most ambiguous real photos, EfficientNet K=7 (two candidate groups and their dominant organ):</p>" + strip(cases, cap=cap),
        "<b>K</b>: EfficientNet / DINOv3 at K=3 match shot scale one-to-one for about 80% / 77% of photos, and at K=7 match organ for 68% / 58%; "
        "organ agreement peaks at K=5–10. BioCLIP at K=63 matches species for 65% and barely groups by organ or scale. K=20 is not special: "
        "stability drops to 0.4–0.5 for K≥30."
        f"<br><b>Soft clustering</b>: ambiguity is real and repeatable. EfficientNet K=7: {ef7['frac_max_lt_0_8']:.0%} of photos have max membership "
        f"&lt;0.8, and entropy from two independent halves of the runs correlates {ef7['entropy_split_half_spearman']:.2f}; "
        f"{ef7['ambiguous_between_different_organ_clusters']:.0%} of the ambiguous photos sit between two groups with different dominant organs "
        f"(only {ef20['ambiguous_between_different_organ_clusters']:.0%} at K=20, where ambiguity is mostly within one organ). "
        "Ambiguity does <b>not</b> predict species errors (AUROC 0.32–0.50) and is only a weak signal for mid-range shots. "
        "<b>Soft membership is not multi-organ evidence</b>: multi-organ labels are not exported yet.",
        ["v", "e", "x"])
