#!/usr/bin/env python3
"""Build the scale-conditioned representation analysis and static review page."""

from __future__ import annotations

import json
import math
from collections import Counter
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = HERE / "data"
ASSETS = HERE / "assets"

REPRESENTATIONS = ["bioclip", "dinov3", "efficientnet_b0"]
REP_LABELS = {
    "bioclip": "BioCLIP 2.5",
    "dinov3": "DINOv3",
    "efficientnet_b0": "EfficientNet-B0",
}
SCALE_ORDER = ["distant", "mid-range", "close-up", "uncertain"]
SCALE_COLORS = {
    "distant": "#376f55",
    "mid-range": "#c58b3c",
    "close-up": "#8b5d8f",
    "uncertain": "#7b8580",
}

CLOSE_NORMALIZED = {
    "flower close-up",
    "flower interior",
    "frontal flower view",
    "lateral flower view",
    "fruit close-up",
    "opened or sectioned fruit",
    "leaf upper surface",
    "leaf margin",
    "leaf scar",
    "terminal bud",
    "needle attachment",
    "petiole arrangement",
}
CLOSE_PHRASES = (
    "close-up",
    "macro view",
    "macro photo",
    "isolated leaf",
    "isolated fruit",
    "isolated seed",
    "isolated flower",
)
DISTANT_PHRASES = (
    "whole tree",
    "full tree",
    "tree crown",
    "full crown",
    "distant view",
    "wide view",
    "canopy view",
)
CLOSE_SOURCE_SUBVIEWS = {
    "whole upper surface",
    "margin of upper + lower surface",
    "lateral or general close-up",
    "lateral view of flower",
    "frontal view of flower",
    "close-up winter leaf scar/bud",
    "close-up winter terminal bud",
    "section or open",
    "entire needle",
    "showing attachment of needles",
    "ventral view of flower + perianth",
    "close-up of flower interior",
}
MID_SOURCE_SUBVIEWS = {
    "showing orientation on twig",
    "orientation of petioles",
    "as borne on the plant",
    "of a medium tree or large branch",
    "of a small tree or small branch",
    "showing leaf bases",
    "basal or on lower stem",
}
ENVIRONMENT_PHRASES = (
    "sky background",
    "blue sky",
    "woodland",
    "outdoor scene",
    "urban setting",
    "forest background",
    "natural setting",
)
ISOLATED_PHRASES = (
    "isolated leaf",
    "isolated fruit",
    "isolated seed",
    "isolated flower",
    "plain background",
    "white background",
    "neutral background",
    "studio background",
)


def as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def contains_any(tags: list[str], phrases: tuple[str, ...]) -> bool:
    return any(phrase in tag for tag in tags for phrase in phrases)


def derive_visual_groups(row: pd.Series) -> pd.Series:
    normalized = set(json.loads(row["normalized_tags"]))
    free = [tag.lower() for tag in json.loads(row["free_tags"])]
    free_joined = " | ".join(free)

    source_distant_evidence = []
    gemma_distant_evidence = []
    source_close_evidence = []
    gemma_close_evidence = []
    source_mid_evidence = []
    gemma_mid_evidence = []
    if row["group"] == "whole-tree":
        source_distant_evidence.append("BioImages group: whole-tree")
    if "whole plant" in normalized:
        gemma_distant_evidence.append("Gemma: whole plant")
    distant_matches = sorted({p for p in DISTANT_PHRASES if any(p in t for t in free)})
    gemma_distant_evidence.extend(f"Gemma free tag: {p}" for p in distant_matches)

    if row["subview"] in CLOSE_SOURCE_SUBVIEWS:
        source_close_evidence.append(f'BioImages subview: {row["subview"]}')
    normalized_close = sorted(normalized & CLOSE_NORMALIZED)
    gemma_close_evidence.extend(f"Gemma: {tag}" for tag in normalized_close)
    close_matches = sorted({p for p in CLOSE_PHRASES if any(p in t for t in free)})
    gemma_close_evidence.extend(f"Gemma free tag: {p}" for p in close_matches)

    if row["subview"] in MID_SOURCE_SUBVIEWS:
        source_mid_evidence.append(f'BioImages subview: {row["subview"]}')
    contextual_structure = bool(normalized & {"branch", "twig", "stem"})
    visible_subject = bool(
        normalized
        & {"leaf", "needle", "flower", "inflorescence", "fruit", "cone", "seed", "bud", "bark", "trunk"}
    )
    if contextual_structure and visible_subject:
        gemma_mid_evidence.append("Gemma: contextual woody structure + visible subject")

    has_distant = bool(source_distant_evidence or gemma_distant_evidence)
    has_close = bool(source_close_evidence or gemma_close_evidence)
    has_mid = bool(source_mid_evidence or gemma_mid_evidence)
    if has_distant and has_close:
        shot_scale = "uncertain"
        shot_scale_confidence = "conflict"
        scale_rationale = "Distant and close-up evidence both present; manual review required."
    elif has_distant:
        shot_scale = "distant"
        shot_scale_confidence = (
            "high" if source_distant_evidence and gemma_distant_evidence else "medium"
        )
        scale_rationale = "Whole organism or crown is visible with scene context."
    elif has_close:
        shot_scale = "close-up"
        shot_scale_confidence = "high" if source_close_evidence and gemma_close_evidence else "medium"
        scale_rationale = "A diagnostic organ, surface, or fine detail is the main subject."
    elif has_mid:
        shot_scale = "mid-range"
        shot_scale_confidence = "high" if source_mid_evidence and gemma_mid_evidence else "medium"
        scale_rationale = "A branch system or contextual plant part is visible, without the whole organism."
    else:
        shot_scale = "uncertain"
        shot_scale_confidence = "low"
        scale_rationale = "No positive distant, mid-range, or close-up evidence was found."

    has_foliage = bool(normalized & {"leaf", "needle"})
    has_leaf_off = (
        bool(normalized & {"winter habit", "winter twig"})
        or "leafless" in free_joined
        or "bare branch" in free_joined
    )
    if has_foliage and has_leaf_off:
        leaf_state = "mixed"
    elif has_foliage:
        leaf_state = "leaf-on"
    elif has_leaf_off:
        leaf_state = "leaf-off"
    else:
        leaf_state = "not-visible"

    reproductive = bool(normalized & {"flower", "inflorescence", "fruit", "cone", "seed"})
    reproductive_visibility = "visible" if reproductive else "not-visible"

    environmental = contains_any(free, ENVIRONMENT_PHRASES)
    isolated = contains_any(free, ISOLATED_PHRASES)
    if environmental and isolated:
        background_context = "mixed"
    elif environmental:
        background_context = "environmental"
    elif isolated:
        background_context = "isolated"
    else:
        background_context = "unspecified"

    return pd.Series(
        {
            "shot_scale": shot_scale,
            "shot_scale_confidence": shot_scale_confidence,
            "shot_scale_rationale": scale_rationale,
            "source_distant_evidence": " | ".join(source_distant_evidence),
            "gemma_distant_evidence": " | ".join(gemma_distant_evidence),
            "source_mid_evidence": " | ".join(source_mid_evidence),
            "gemma_mid_evidence": " | ".join(gemma_mid_evidence),
            "source_close_evidence": " | ".join(source_close_evidence),
            "gemma_close_evidence": " | ".join(gemma_close_evidence),
            "needs_scale_review": shot_scale == "uncertain" or shot_scale_confidence != "high",
            "leaf_state": leaf_state,
            "reproductive_visibility": reproductive_visibility,
            "background_context": background_context,
        }
    )


def weighted_cluster_purity(frame: pd.DataFrame) -> float:
    if frame.empty:
        return float("nan")
    correct = frame.groupby("cluster_id")["species"].agg(lambda s: s.value_counts().iloc[0]).sum()
    return float(correct / len(frame))


def normalized_entropy(values: pd.Series) -> float:
    counts = values.value_counts()
    if len(counts) <= 1:
        return 0.0
    probs = counts / counts.sum()
    return float(-(probs * np.log(probs)).sum() / math.log(len(SCALE_ORDER)))


def compact_distribution(values: pd.Series, limit: int | None = None) -> str:
    counts = values.value_counts()
    if limit:
        counts = counts.head(limit)
    return " · ".join(f"{name} {count}" for name, count in counts.items())


def load_and_label() -> pd.DataFrame:
    gemma = pd.read_csv(ROOT / "bioimages_browser/tagging/gemma_reference.csv")
    manifest = pd.read_csv(ROOT / "bioimages_browser/data/image_manifest.csv").rename(
        columns={"id": "image_id"}
    )
    keep = [
        "image_id",
        "file",
        "species",
        "common_name",
        "family",
        "organ_category",
        "subview",
        "primary_label",
        "group",
        "individual_id",
        "thumbnail_url",
        "image_url",
        "source_url",
    ]
    merged = manifest[keep].merge(gemma, on="image_id", how="left", validate="one_to_one")
    if merged["normalized_tags"].isna().any():
        missing = merged.loc[merged["normalized_tags"].isna(), "image_id"].tolist()
        raise ValueError(f"Missing Gemma tags for {len(missing)} images")
    labels = merged.apply(derive_visual_groups, axis=1)
    return pd.concat([merged, labels], axis=1)


def build_group_counts(images: pd.DataFrame, analysis_ids: set[str]) -> pd.DataFrame:
    rows = []
    axes = ["shot_scale", "leaf_state", "reproductive_visibility", "background_context"]
    for axis in axes:
        for value, group in images.groupby(axis, dropna=False):
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "all_images": len(group),
                    "analysis_images": group["image_id"].isin(analysis_ids).sum(),
                }
            )
    return pd.DataFrame(rows)


def build_representation_metrics(images: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    joined = assignments.merge(
        images[["image_id", "species", "shot_scale", "leaf_state", "reproductive_visibility", "background_context"]],
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    axes = ["shot_scale", "leaf_state", "reproductive_visibility", "background_context"]
    rows = []
    for representation, rep in joined.groupby("representation"):
        for axis in axes:
            axis_nmi = normalized_mutual_info_score(rep[axis], rep["cluster_id"])
            for value, group in rep.groupby(axis):
                rows.append(
                    {
                        "representation": representation,
                        "axis": axis,
                        "value": value,
                        "images": len(group),
                        "species_count": group["species"].nunique(),
                        "species_nmi_with_clusters": normalized_mutual_info_score(
                            group["species"], group["cluster_id"]
                        ),
                        "species_cluster_purity": weighted_cluster_purity(group),
                        "cluster_axis_nmi_all_values": axis_nmi,
                    }
                )
    return pd.DataFrame(rows)


def build_accuracy_metrics(images: pd.DataFrame) -> pd.DataFrame:
    truth = pd.read_csv(ROOT / "error_analysis/data/unified_predictions_image.csv")
    efficient = pd.read_csv(ROOT / "representation_analysis/data/efficientnet_test_predictions.csv")
    cols = ["image_id", "bioclip_correct", "dinov3_correct"]
    predictions = truth[cols].merge(
        efficient[["image_id", "correct"]].rename(columns={"correct": "efficientnet_b0_correct"}),
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    for rep in REPRESENTATIONS:
        predictions[f"{rep}_correct"] = predictions[f"{rep}_correct"].map(as_bool)
    joined = predictions.merge(
        images[["image_id", "shot_scale", "leaf_state", "reproductive_visibility", "background_context"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )
    rows = []
    for axis in ["shot_scale", "leaf_state", "reproductive_visibility", "background_context"]:
        for value, group in joined.groupby(axis):
            for rep in REPRESENTATIONS:
                rows.append(
                    {
                        "axis": axis,
                        "value": value,
                        "representation": rep,
                        "images": len(group),
                        "correct": int(group[f"{rep}_correct"].sum()),
                        "top1_accuracy": float(group[f"{rep}_correct"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def build_cluster_summaries(images: pd.DataFrame, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    joined = assignments.merge(
        images[["image_id", "species", "shot_scale", "leaf_state", "reproductive_visibility"]],
        on="image_id",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for (representation, cluster_id), group in joined.groupby(["representation", "cluster_id"]):
        species_purity = group["species"].value_counts().iloc[0] / len(group)
        scale_entropy = normalized_entropy(group["shot_scale"])
        rows.append(
            {
                "representation": representation,
                "cluster_id": int(cluster_id),
                "images": len(group),
                "species_count": group["species"].nunique(),
                "species_purity": species_purity,
                "shot_scale_entropy": scale_entropy,
                "shot_scale_distribution": compact_distribution(group["shot_scale"]),
                "top_species": compact_distribution(group["species"], limit=5),
                "review_priority_score": (1 - species_purity) * scale_entropy * math.log1p(len(group)),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["representation", "cluster_id"])
    priority = (
        summary.sort_values(["representation", "review_priority_score"], ascending=[True, False])
        .groupby("representation", as_index=False)
        .head(5)
        .reset_index(drop=True)
    )
    return summary, priority


def build_seed_explorer(
    images: pd.DataFrame, assignments: pd.DataFrame, priority: pd.DataFrame
) -> dict[str, list[dict[str, object]]]:
    """Build a compact, deterministic seed set from existing cross-observation neighbors."""
    neighbors = pd.read_csv(ROOT / "representation_analysis/data/nearest_neighbors.csv")
    neighbors = neighbors[neighbors["scope"] == "cross_observation"].copy()
    image_lookup = images.set_index("image_id")
    result: dict[str, list[dict[str, object]]] = {}
    for rep in ["bioclip", "dinov3"]:
        rep_neighbors = neighbors[neighbors["representation"] == rep]
        query_ids = set(rep_neighbors["query_image_id"])
        rep_assignments = assignments[
            (assignments["representation"] == rep) & (assignments["image_id"].isin(query_ids))
        ]
        seeds = []
        for cluster_id in priority.loc[priority["representation"] == rep, "cluster_id"]:
            candidates = rep_assignments[rep_assignments["cluster_id"] == cluster_id].sort_values(
                "distance_to_centroid"
            )
            if candidates.empty:
                continue
            seed_id = candidates.iloc[0]["image_id"]
            seed = image_lookup.loc[seed_id]
            rows = rep_neighbors[rep_neighbors["query_image_id"] == seed_id].sort_values("rank").head(10)
            neighbor_items = []
            for item in rows.itertuples():
                record = image_lookup.loc[item.neighbor_image_id]
                neighbor_items.append(
                    {
                        "rank": int(item.rank),
                        "image_id": item.neighbor_image_id,
                        "species": record.species,
                        "shot_scale": record.shot_scale,
                        "thumbnail_url": record.thumbnail_url,
                        "source_url": record.source_url,
                        "cosine_similarity": round(float(item.cosine_similarity), 4),
                        "same_species": bool(item.same_species),
                        "same_scale": record.shot_scale == seed.shot_scale,
                    }
                )
            seeds.append(
                {
                    "image_id": seed_id,
                    "cluster_id": int(cluster_id),
                    "species": seed.species,
                    "shot_scale": seed.shot_scale,
                    "primary_label": seed.primary_label,
                    "thumbnail_url": seed.thumbnail_url,
                    "source_url": seed.source_url,
                    "neighbors": neighbor_items,
                }
            )
        result[rep] = seeds
    return result


def make_projection_plots(images: pd.DataFrame) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    labels = images.set_index("image_id")["shot_scale"]
    for rep in REPRESENTATIONS:
        path = ROOT / f"representation_analysis/data/{rep}_final_projection_coordinates.npz"
        with np.load(path) as bundle:
            ids = bundle["image_ids"].astype(str)
            coords = bundle["pca"]
        fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=150)
        for scale in SCALE_ORDER:
            mask = np.array([labels.get(image_id) == scale for image_id in ids])
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=10 if scale != "mixed" else 20,
                alpha=0.58,
                linewidths=0,
                label=f"{scale} ({mask.sum():,})",
                color=SCALE_COLORS[scale],
            )
        ax.set_title(f"{REP_LABELS[rep]} · PCA colored by shot scale", loc="left", fontsize=13)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(color="#e7ebe8", linewidth=0.6)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(ASSETS / f"{rep}_pca_shot_scale.png", bbox_inches="tight")
        plt.close(fig)


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def table(headers: list[str], rows: list[list[str]], classes: str = "") -> str:
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f'<div class="table-wrap {classes}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build_html(
    images: pd.DataFrame,
    metrics: pd.DataFrame,
    accuracy: pd.DataFrame,
    priority: pd.DataFrame,
    assignments: pd.DataFrame,
    seed_data: dict[str, list[dict[str, object]]],
) -> str:
    counts = images["shot_scale"].value_counts()
    scale_cards = "".join(
        f'<div class="stat"><span class="dot" style="background:{SCALE_COLORS[name]}"></span>'
        f'<strong>{counts.get(name, 0):,}</strong><span>{escape(name)}</span></div>'
        for name in SCALE_ORDER
    )

    metric_rows = []
    for scale in ["distant", "mid-range", "close-up"]:
        for rep in REPRESENTATIONS:
            row = metrics[(metrics.axis == "shot_scale") & (metrics.value == scale) & (metrics.representation == rep)].iloc[0]
            metric_rows.append(
                [
                    escape(scale),
                    REP_LABELS[rep],
                    f'{int(row["images"]):,}',
                    pct(row["species_cluster_purity"]),
                    f'{row["species_nmi_with_clusters"]:.3f}',
                ]
            )

    accuracy_rows = []
    accuracy_findings = []
    for scale in ["distant", "mid-range", "close-up"]:
        for rep in REPRESENTATIONS:
            row = accuracy[(accuracy.axis == "shot_scale") & (accuracy.value == scale) & (accuracy.representation == rep)].iloc[0]
            accuracy_rows.append(
                [
                    escape(scale),
                    REP_LABELS[rep],
                    f'{int(row["correct"])}/{int(row["images"])}',
                    pct(row["top1_accuracy"]),
                ]
            )
    for rep in REPRESENTATIONS:
        distant_accuracy = accuracy[
            (accuracy.axis == "shot_scale")
            & (accuracy.value == "distant")
            & (accuracy.representation == rep)
        ].iloc[0]
        close_accuracy = accuracy[
            (accuracy.axis == "shot_scale")
            & (accuracy.value == "close-up")
            & (accuracy.representation == rep)
        ].iloc[0]
        gap = 100 * (close_accuracy.top1_accuracy - distant_accuracy.top1_accuracy)
        accuracy_findings.append(
            f'<article class="rule"><h3>{REP_LABELS[rep]}</h3><p>Close-up {pct(close_accuracy.top1_accuracy)} '
            f'vs distant {pct(distant_accuracy.top1_accuracy)} · <strong>{gap:+.1f} percentage points</strong>.</p></article>'
        )

    manifest = images.set_index("image_id")
    gallery_sections = []
    joined_assignments = assignments.merge(
        images[["image_id", "shot_scale"]], on="image_id", how="left", validate="many_to_one"
    )
    for rep in REPRESENTATIONS:
        rep_cards = []
        for _, row in priority[priority.representation == rep].iterrows():
            cluster = joined_assignments[
                (joined_assignments.representation == rep)
                & (joined_assignments.cluster_id == row.cluster_id)
            ].sort_values("distance_to_centroid").head(12)
            imgs = []
            for item in cluster.itertuples():
                record = manifest.loc[item.image_id]
                imgs.append(
                    f'<a class="thumb" href="{escape(record.source_url)}" target="_blank" rel="noreferrer">'
                    f'<img loading="lazy" src="{escape(record.thumbnail_url)}" alt="{escape(record.species)}">'
                    f'<span><b>{escape(record.species)}</b><i>{escape(item.shot_scale)}</i></span></a>'
                )
            rep_cards.append(
                f'<article class="cluster-card"><div class="cluster-head"><div><small>cluster</small>'
                f'<h3>{int(row.cluster_id):02d}</h3></div><div class="cluster-meta">'
                f'<b>{int(row.images)} images · {int(row.species_count)} species</b>'
                f'<span>species purity {pct(row.species_purity)} · scale entropy {row.shot_scale_entropy:.2f}</span>'
                f'<span>{escape(row.shot_scale_distribution)}</span><span>{escape(row.top_species)}</span></div></div>'
                f'<div class="thumb-grid">{"".join(imgs)}</div></article>'
            )
        gallery_sections.append(
            f'<section class="rep-panel" data-rep="{rep}"><h2>{REP_LABELS[rep]} review queue</h2>'
            f'<p class="section-note">Ranked by species impurity × shot-scale entropy × log cluster size.</p>'
            f'{"".join(rep_cards)}</section>'
        )

    plots = "".join(
        f'<figure><img src="assets/{rep}_pca_shot_scale.png" alt="{REP_LABELS[rep]} PCA by shot scale">'
        f'<figcaption>{REP_LABELS[rep]} · the saved global PCA coordinates, recolored with the new scale labels.</figcaption></figure>'
        for rep in REPRESENTATIONS
    )

    seed_json = json.dumps(seed_data, ensure_ascii=False).replace("</", "<\\/")
    css = """
:root{--ink:#17221d;--muted:#65716a;--line:#d8dfda;--paper:#fff;--soft:#f4f7f5;--green:#315f4a}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:#fbfcfb;font:15px/1.55 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:var(--green)}.wrap{max-width:1180px;margin:auto;padding:0 28px}header{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}nav{display:flex;align-items:center;gap:20px;min-height:58px;flex-wrap:wrap}.brand{font-weight:750;margin-right:auto}nav a{text-decoration:none;color:var(--ink);font-size:13px}
main{padding-bottom:80px}.hero{padding:66px 0 38px}.eyebrow{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--green);font-weight:750}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400}h1{font-size:clamp(2.5rem,6vw,5.4rem);line-height:.98;max-width:960px;margin:12px 0 20px}h2{font-size:clamp(1.7rem,3vw,2.6rem);margin:0 0 10px}.lede{font-size:17px;color:var(--muted);max-width:860px}.section{padding:42px 0;border-top:1px solid var(--line)}.section-note{color:var(--muted);max-width:820px}.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:28px}.stat{background:var(--paper);border:1px solid var(--line);padding:18px;display:grid;grid-template-columns:auto 1fr;gap:2px 10px;align-items:center}.stat strong{font:30px Georgia,serif}.stat span:last-child{grid-column:2;color:var(--muted)}.dot{width:10px;height:10px;border-radius:50%}
.rule-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.rule{background:var(--paper);border:1px solid var(--line);padding:20px}.rule h3{margin:0 0 8px}.rule p{margin:0;color:var(--muted)}code{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;background:var(--soft);padding:2px 5px}.table-wrap{overflow:auto;border:1px solid var(--line);background:var(--paper);margin-top:18px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{background:var(--soft);font-size:12px}tr:last-child td{border-bottom:0}.plot-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.plot-grid figure{margin:0;background:var(--paper);border:1px solid var(--line);padding:10px}.plot-grid img{width:100%;display:block}.plot-grid figcaption{font-size:12px;color:var(--muted);padding:8px 4px 3px}
.rep-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}.rep-tabs button{border:1px solid var(--line);background:var(--paper);padding:9px 13px;cursor:pointer;font:inherit}.rep-tabs button.active{color:white;background:var(--green);border-color:var(--green)}.rep-panel{display:none}.rep-panel.active{display:block}.cluster-card{background:var(--paper);border:1px solid var(--line);padding:18px;margin:14px 0}.cluster-head{display:grid;grid-template-columns:90px 1fr;gap:18px;margin-bottom:14px}.cluster-head small{text-transform:uppercase;letter-spacing:.1em;color:var(--muted)}.cluster-head h3{font-size:44px;margin:0;line-height:1}.cluster-meta{display:grid;gap:2px}.cluster-meta span{color:var(--muted);font-size:13px}.thumb-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}.thumb{display:block;border:1px solid var(--line);text-decoration:none;color:var(--ink);overflow:hidden}.thumb img{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;background:var(--soft)}.thumb span{display:grid;padding:6px;font-size:10px;line-height:1.3}.thumb b{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.thumb i{font-style:normal;color:var(--muted)}
.seed-controls{display:flex;gap:12px;align-items:end;flex-wrap:wrap;margin:20px 0}.seed-controls label{display:grid;gap:5px;color:var(--muted);font-size:12px}.seed-controls select{min-width:340px;padding:9px;border:1px solid var(--line);background:var(--paper);font:inherit}.seed-layout{display:grid;grid-template-columns:190px 1fr;gap:14px}.seed-card{border:2px solid var(--green);background:var(--paper);padding:8px}.seed-card img{width:100%;aspect-ratio:1/1;object-fit:cover}.seed-card span{display:grid;font-size:12px;padding:7px 2px}.neighbor-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.neighbor{border:1px solid var(--line);background:var(--paper);padding:7px}.neighbor.changed{border-color:#b96a45}.neighbor img{width:100%;aspect-ratio:1/1;object-fit:cover}.neighbor small{display:block;color:var(--muted)}.neighbor b{display:block;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.neighbor .flags{font-size:10px;color:#8f4b31}.seed-summary{margin:12px 0;padding:12px;background:var(--soft);font-size:13px}.next{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}.next div{border-left:3px solid var(--green);background:var(--soft);padding:18px}.next h3{margin:0 0 6px}.next p{margin:0;color:var(--muted)}footer{border-top:1px solid var(--line);padding:28px 0 60px;color:var(--muted);font-size:12px}
@media(max-width:900px){.stats,.plot-grid{grid-template-columns:repeat(2,1fr)}.thumb-grid{grid-template-columns:repeat(4,1fr)}.neighbor-grid{grid-template-columns:repeat(3,1fr)}}@media(max-width:620px){.wrap{padding:0 18px}.stats,.rule-grid,.plot-grid,.next,.seed-layout{grid-template-columns:1fr}.thumb-grid,.neighbor-grid{grid-template-columns:repeat(2,1fr)}.seed-controls select{min-width:0;width:100%}nav a{display:none}}
"""
    js = """
const buttons=[...document.querySelectorAll('[data-tab]')];
const panels=[...document.querySelectorAll('.rep-panel')];
function show(rep){buttons.forEach(b=>b.classList.toggle('active',b.dataset.tab===rep));panels.forEach(p=>p.classList.toggle('active',p.dataset.rep===rep));}
buttons.forEach(b=>b.addEventListener('click',()=>show(b.dataset.tab)));show('bioclip');
const seedData=JSON.parse(document.getElementById('seed-data').textContent);
const seedRep=document.getElementById('seed-rep'),seedSelect=document.getElementById('seed-select'),seedCount=document.getElementById('seed-count');
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function loadSeeds(){seedSelect.innerHTML=seedData[seedRep.value].map((s,i)=>`<option value="${i}">cluster ${String(s.cluster_id).padStart(2,'0')} · ${esc(s.species)} · ${esc(s.shot_scale)}</option>`).join('');renderSeed();}
function renderSeed(){const s=seedData[seedRep.value][Number(seedSelect.value||0)],n=Math.min(Number(seedCount.value),s.neighbors.length);document.getElementById('seed-count-label').textContent=n;document.getElementById('seed-query').innerHTML=`<a href="${esc(s.source_url)}" target="_blank" rel="noreferrer"><img src="${esc(s.thumbnail_url)}" alt="${esc(s.species)}"></a><span><b>${esc(s.species)}</b><i>${esc(s.shot_scale)} · cluster ${String(s.cluster_id).padStart(2,'0')}</i><small>${esc(s.primary_label)}</small></span>`;const visible=s.neighbors.slice(0,n);document.getElementById('seed-neighbors').innerHTML=visible.map(x=>`<article class="neighbor ${(!x.same_species||!x.same_scale)?'changed':''}"><a href="${esc(x.source_url)}" target="_blank" rel="noreferrer"><img src="${esc(x.thumbnail_url)}" alt="${esc(x.species)}"></a><small>#${x.rank} · cosine ${x.cosine_similarity}</small><b>${esc(x.species)}</b><small>${esc(x.shot_scale)}</small><div class="flags">${x.same_species?'':'species changed '}${x.same_scale?'':'scale changed'}</div></article>`).join('');const firstSpecies=s.neighbors.find(x=>!x.same_species),firstScale=s.neighbors.find(x=>!x.same_scale);document.getElementById('seed-summary').textContent=`First species change: ${firstSpecies?'rank '+firstSpecies.rank:'not in Top-10'} · first scale change: ${firstScale?'rank '+firstScale.rank:'not in Top-10'}. These are automatic flags; the semantic stopping boundary still needs human review.`;}
seedRep.addEventListener('change',loadSeeds);seedSelect.addEventListener('change',renderSeed);seedCount.addEventListener('input',renderSeed);loadSeeds();
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Scale-conditioned representations · BioImages</title><style>{css}</style></head><body>
<header><div class="wrap"><nav><span class="brand">BioImages · representation analysis</span><a href="#labels">Labels</a><a href="#projections">Projections</a><a href="#performance">Performance</a><a href="#review">Review queue</a><a href="#seed">Seed explorer</a><a href="#next">Next</a></nav></div></header>
<main class="wrap"><section class="hero"><div class="eyebrow">Scale-conditioned analysis · phase 1</div><h1>What do the embeddings group when viewing scale changes?</h1><p class="lede">An image-level visual audit over all 1,899 photographs, joined to the existing 1,641-image frozen-embedding analysis. The page separates distant, mid-range and close-up photographs before comparing species structure and classification errors.</p><div class="stats">{scale_cards}</div></section>
<section class="section" id="labels"><h2>Auditable visual groups</h2><p class="section-note">Rules combine canonical BioImages metadata with the existing image-grounded Gemma tags. Every row preserves the evidence that fired each rule in <a href="data/image_visual_groups.csv">image_visual_groups.csv</a>.</p><div class="rule-grid">
<article class="rule"><h3>Shot scale</h3><p><code>distant</code> uses whole-plant or full-crown evidence; <code>close-up</code> uses explicit close-up or fine-detail views; <code>mid-range</code> has neither signal; <code>mixed</code> has both.</p></article>
<article class="rule"><h3>Leaf state</h3><p><code>leaf-on</code> uses visible leaf or needle evidence; <code>leaf-off</code> uses winter, bare-branch or leafless evidence. Mixed and not-visible remain explicit.</p></article>
<article class="rule"><h3>Reproductive visibility</h3><p>A paired grouping: flower, inflorescence, fruit, cone or seed visible versus not visible.</p></article>
<article class="rule"><h3>Background context</h3><p>An exploratory pair: environmental context versus isolated subject. Most images remain unspecified, so this axis is descriptive.</p></article></div></section>
<section class="section" id="projections"><h2>Global PCA recolored by shot scale</h2><p class="section-note">These plots reuse the saved coordinates and provide the baseline for the later local rotation/projection explorer. Within-group NMI and purity are descriptive because group size and species composition differ.</p><div class="plot-grid">{plots}</div>{table(["Shot scale","Representation","n","Species purity","Species NMI"],metric_rows)}</section>
<section class="section" id="performance"><h2>Correct and incorrect predictions by shot scale</h2><p class="section-note">Top-1 results use the existing strict 211-image test set. The table shows whether a representation's classifier succeeds differently across photographic scales.</p><div class="rule-grid">{''.join(accuracy_findings)}</div>{table(["Shot scale","Representation","Correct / n","Top-1 accuracy"],accuracy_rows)}</section>
<section class="section" id="review"><h2>Mixed-cluster review queue</h2><p class="section-note">Each card shows the images nearest the existing k=20 centroid. The queue elevates clusters that mix species and shot scales, so visual review starts where scale is a plausible explanation for species mixing.</p><div class="rep-tabs"><button data-tab="bioclip">BioCLIP 2.5</button><button data-tab="dinov3">DINOv3</button><button data-tab="efficientnet_b0">EfficientNet-B0</button></div>{''.join(gallery_sections)}</section>
<section class="section" id="seed"><h2>Seed-guided neighborhood explorer</h2><p class="section-note">Representative strict-test seeds come from the mixed-cluster queue. Neighbors exclude the seed's own observation and are ordered by original-space cosine similarity.</p><div class="seed-controls"><label>Representation<select id="seed-rep"><option value="bioclip">BioCLIP 2.5</option><option value="dinov3">DINOv3</option></select></label><label>Seed<select id="seed-select"></select></label><label>Reveal neighbors · <span id="seed-count-label">5</span><input id="seed-count" type="range" min="1" max="10" value="5"></label></div><div class="seed-layout"><article class="seed-card" id="seed-query"></article><div><div class="neighbor-grid" id="seed-neighbors"></div><div class="seed-summary" id="seed-summary"></div></div></div></section>
<section class="section" id="next"><h2>Continuation design</h2><div class="next"><div><h3>Next · local projection</h3><p>Select a mixed cluster or species pair, recompute local PCA from the original frozen vectors, and animate rotations across selected top dimensions while coloring by scale, species, organ and correctness.</p></div><div><h3>Next · human boundary capture</h3><p>Persist the reviewer-selected neighbor rank where visual semantics change, then compare boundaries across representations and seed categories.</p></div></div></section></main>
<footer><div class="wrap">Generated from committed BioImages metadata, Gemma tags, frozen k=20 assignments, saved PCA coordinates, original-space neighbors and strict test predictions. Labels are analysis annotations and do not replace BioImages source records.</div></footer><script id="seed-data" type="application/json">{seed_json}</script><script>{js}</script></body></html>"""


def display_image_url(image_id: str, remote_url: str) -> str:
    local_name = image_id.replace("/", "__") + ".jpg"
    local_path = ROOT / "research_report/gallery_images" / local_name
    if local_path.exists():
        return f"../research_report/gallery_images/{local_name}"
    return remote_url


def build_codebook(images: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    preferred_subviews = {
        "distant": ["general", "winter", "view up trunk"],
        "mid-range": [
            "showing orientation on twig",
            "orientation of petioles",
            "as borne on the plant",
            "of a medium tree or large branch",
            "of a small tree or small branch",
            "showing leaf bases",
            "basal or on lower stem",
        ],
        "close-up": [
            "whole upper surface",
            "margin of upper + lower surface",
            "lateral or general close-up",
            "close-up winter leaf scar/bud",
            "close-up winter terminal bud",
            "close-up of flower interior",
            "frontal view of flower",
            "section or open",
        ],
    }
    result: dict[str, list[dict[str, str]]] = {}
    for scale in ["distant", "mid-range", "close-up"]:
        pool = images[
            (images["shot_scale"] == scale) & (images["shot_scale_confidence"] == "high")
        ].copy()
        pool["has_local"] = pool.apply(
            lambda r: (
                ROOT
                / "research_report/gallery_images"
                / (r["image_id"].replace("/", "__") + ".jpg")
            ).exists(),
            axis=1,
        )
        chosen: list[pd.Series] = []
        used_species: set[str] = set()
        for subview in preferred_subviews[scale]:
            candidates = pool[
                (pool["subview"] == subview) & (~pool["species"].isin(used_species))
            ].sort_values(["has_local", "species", "image_id"], ascending=[False, True, True])
            if not candidates.empty:
                row = candidates.iloc[0]
                chosen.append(row)
                used_species.add(row["species"])
            if len(chosen) == 6:
                break
        if len(chosen) < 6:
            candidates = pool[~pool["species"].isin(used_species)].sort_values(
                ["has_local", "organ_category", "species", "image_id"],
                ascending=[False, True, True, True],
            )
            for _, row in candidates.iterrows():
                chosen.append(row)
                used_species.add(row["species"])
                if len(chosen) == 6:
                    break
        result[scale] = [
            {
                "image_id": row["image_id"],
                "species": row["species"],
                "organ_category": row["organ_category"],
                "subview": row["subview"],
                "image_url": display_image_url(row["image_id"], row["image_url"]),
                "source_url": row["source_url"],
                "reason": row["shot_scale_rationale"],
            }
            for row in chosen
        ]
    uncertain = images[images["shot_scale"] == "uncertain"].sort_values(
        ["shot_scale_confidence", "image_id"]
    )
    result["uncertain"] = [
        {
            "image_id": row["image_id"],
            "species": row["species"],
            "organ_category": row["organ_category"],
            "subview": row["subview"],
            "image_url": display_image_url(row["image_id"], row["image_url"]),
            "source_url": row["source_url"],
            "reason": row["shot_scale_rationale"],
        }
        for _, row in uncertain.head(4).iterrows()
    ]
    return result


def build_scale_review_sample(images: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for scale in ["distant", "mid-range", "close-up"]:
        group = images[images["shot_scale"] == scale]
        for confidence in ["high", "medium"]:
            candidates = group[group["shot_scale_confidence"] == confidence]
            parts.append(candidates.sample(n=min(25, len(candidates)), random_state=42))
    parts.append(images[images["shot_scale"] == "uncertain"])
    review = pd.concat(parts, ignore_index=True).drop_duplicates("image_id")
    review = review.sort_values(["shot_scale", "shot_scale_confidence", "image_id"])
    columns = [
        "image_id",
        "species",
        "organ_category",
        "subview",
        "image_url",
        "source_url",
        "shot_scale",
        "shot_scale_confidence",
        "shot_scale_rationale",
        "source_distant_evidence",
        "gemma_distant_evidence",
        "source_mid_evidence",
        "gemma_mid_evidence",
        "source_close_evidence",
        "gemma_close_evidence",
    ]
    review = review[columns].copy()
    review["human_scale"] = ""
    review["human_agrees"] = ""
    review["review_note"] = ""
    return review


def make_explanatory_plots(images: pd.DataFrame, assignments: pd.DataFrame) -> None:
    labels = images.set_index("image_id")["shot_scale"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), dpi=170)
    for ax, rep in zip(axes, REPRESENTATIONS):
        with np.load(ROOT / f"representation_analysis/data/{rep}_final_projection_coordinates.npz") as bundle:
            ids = bundle["image_ids"].astype(str)
            coords = bundle["pca"]
        for scale in SCALE_ORDER:
            mask = np.array([labels.get(image_id) == scale for image_id in ids])
            ax.scatter(
                coords[mask, 0], coords[mask, 1], s=8, alpha=.58, linewidths=0,
                color=SCALE_COLORS[scale], label=scale,
            )
        ax.set_title(REP_LABELS[rep], loc="left", fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2" if rep == "bioclip" else "")
        ax.grid(color="#e7ebe8", linewidth=.5)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("The same 1,641 images, colored only by photographic scale", x=.04, ha="left", fontsize=15)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(ASSETS / "representation_by_scale.png", bbox_inches="tight")
    plt.close(fig)

    for rep in REPRESENTATIONS:
        with np.load(ROOT / f"representation_analysis/data/{rep}_final_projection_coordinates.npz") as bundle:
            ids = bundle["image_ids"].astype(str)
            coords = bundle["pca"]
        cluster_map = assignments[assignments["representation"] == rep].set_index("image_id")[
            "cluster_id"
        ]
        clusters = np.array([cluster_map.get(image_id, -1) for image_id in ids])
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), dpi=170, sharex=True, sharey=True)
        for ax, scale in zip(axes, ["distant", "mid-range", "close-up"]):
            mask = np.array([labels.get(image_id) == scale for image_id in ids])
            ax.scatter(coords[:, 0], coords[:, 1], s=6, color="#dfe4e1", alpha=.35, linewidths=0)
            ax.scatter(
                coords[mask, 0], coords[mask, 1], c=clusters[mask], cmap="tab20", vmin=0, vmax=19,
                s=13, alpha=.82, linewidths=0,
            )
            ax.set_title(f"{scale} · n={mask.sum():,}", loc="left", fontsize=11)
            ax.set_xlabel("PC1")
            ax.grid(color="#edf0ee", linewidth=.5)
        axes[0].set_ylabel("PC2")
        fig.suptitle(
            f"{REP_LABELS[rep]} · each scale shown in the same global PCA space\nColored points use the existing k=20 cluster assignment; all other images are gray.",
            x=.04, ha="left", fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, .88))
        fig.savefig(ASSETS / f"{rep}_scale_conditioned_clusters.png", bbox_inches="tight")
        plt.close(fig)

    rep = "bioclip"
    cluster_id = 4
    with np.load(ROOT / "representation_analysis/data/bioclip_final_projection_coordinates.npz") as bundle:
        ids = bundle["image_ids"].astype(str)
        coords = bundle["pca"]
    assignment_map = assignments[assignments["representation"] == rep].set_index("image_id")[
        "cluster_id"
    ]
    cluster_mask = np.array([assignment_map.get(image_id, -1) == cluster_id for image_id in ids])
    species_map = images.set_index("image_id")["species"]
    cluster_species = pd.Series([species_map.get(image_id, "Unknown") for image_id in ids[cluster_mask]])
    top_species = cluster_species.value_counts().head(4).index.tolist()
    species_colors = dict(zip(top_species, ["#315f4a", "#b36b3d", "#4f6fa8", "#9a5f85"]))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=170)
    axes[0].scatter(coords[:, 0], coords[:, 1], s=6, color="#dfe4e1", alpha=.4, linewidths=0)
    axes[0].scatter(coords[cluster_mask, 0], coords[cluster_mask, 1], s=18, color="#315f4a", alpha=.85, linewidths=0)
    axes[0].set_title("1 · Where cluster 04 sits", loc="left", fontsize=11)
    for scale in SCALE_ORDER:
        mask = cluster_mask & np.array([labels.get(image_id) == scale for image_id in ids])
        axes[1].scatter(coords[mask, 0], coords[mask, 1], s=23, alpha=.82, linewidths=0, color=SCALE_COLORS[scale], label=scale)
    axes[1].set_title("2 · The cluster colored by scale", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=7)
    for species in top_species:
        mask = cluster_mask & np.array([species_map.get(image_id) == species for image_id in ids])
        axes[2].scatter(coords[mask, 0], coords[mask, 1], s=23, alpha=.82, linewidths=0, color=species_colors[species], label=species)
    other = cluster_mask & np.array([species_map.get(image_id) not in top_species for image_id in ids])
    axes[2].scatter(coords[other, 0], coords[other, 1], s=13, alpha=.35, linewidths=0, color="#9ca6a0", label="other species")
    axes[2].set_title("3 · The same points colored by species", loc="left", fontsize=11)
    axes[2].legend(frameon=False, fontsize=6.5, loc="best")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.grid(color="#edf0ee", linewidth=.5)
    axes[0].set_ylabel("PC2")
    fig.suptitle("BioCLIP cluster 04 · one mixed cluster viewed three ways", x=.04, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(ASSETS / "bioclip_cluster_04_case.png", bbox_inches="tight")
    plt.close(fig)


def build_explanatory_html(
    images: pd.DataFrame,
    assignments: pd.DataFrame,
    codebook: dict[str, list[dict[str, str]]],
    review_sample: pd.DataFrame,
) -> str:
    counts = images["shot_scale"].value_counts()
    confidence = images["shot_scale_confidence"].value_counts()

    definitions = {
        "distant": ("Whole organism", "The entire plant, tree, shrub, or most of the crown is visible. Overall habit and scene context are readable."),
        "mid-range": ("Contextual plant part", "A branch system, trunk section, or several connected organs are visible. The whole organism is outside the frame."),
        "close-up": ("Diagnostic detail", "One organ, surface, or fine structure dominates the frame and its local texture or shape is readable."),
        "uncertain": ("Boundary or conflict", "The image contains evidence for more than one scale, or has no positive evidence for a stable label."),
    }
    codebook_sections = []
    for scale in SCALE_ORDER:
        title, definition = definitions[scale]
        cards = []
        for item in codebook[scale]:
            cards.append(
                f'<a class="example" href="{escape(item["source_url"])}" target="_blank" rel="noreferrer">'
                f'<img loading="lazy" src="{escape(item["image_url"])}" alt="{escape(item["species"])}">'
                f'<div><b>{escape(item["species"])}</b><span>{escape(item["organ_category"])} · {escape(item["subview"])}</span>'
                f'<p>{escape(item["reason"])}</p></div></a>'
            )
        codebook_sections.append(
            f'<section class="scale-block"><div class="scale-copy"><span class="scale-dot" style="background:{SCALE_COLORS[scale]}"></span>'
            f'<div><h3>{escape(scale)} <small>{escape(title)}</small></h3><p>{escape(definition)}</p>'
            f'<strong>{counts.get(scale, 0):,} provisional labels</strong></div></div><div class="example-grid">{"".join(cards)}</div></section>'
        )

    cluster = assignments[
        (assignments["representation"] == "bioclip") & (assignments["cluster_id"] == 4)
    ].merge(images, on="image_id", how="left", validate="one_to_one")
    scale_rows = []
    for scale in SCALE_ORDER:
        group = cluster[cluster["shot_scale"] == scale].sort_values("distance_to_centroid")
        cards = []
        for row in group.head(10).itertuples():
            cards.append(
                f'<a class="case-image" href="{escape(row.source_url)}" target="_blank" rel="noreferrer">'
                f'<img loading="lazy" src="{escape(row.image_url)}" alt="{escape(row.species)}">'
                f'<span><b>{escape(row.species)}</b><i>{escape(row.organ_category)} · {escape(row.subview)}</i></span></a>'
            )
        scale_rows.append(
            f'<div class="case-row"><div class="case-label"><span class="scale-dot" style="background:{SCALE_COLORS[scale]}"></span>'
            f'<b>{escape(scale)}</b><small>{len(group)} images · {group["species"].nunique()} species</small></div>'
            f'<div class="case-grid">{"".join(cards) if cards else "<em>No images</em>"}</div></div>'
        )

    rep_figures = "".join(
        f'<figure><img src="assets/{rep}_scale_conditioned_clusters.png" alt="{REP_LABELS[rep]} scale-conditioned PCA">'
        f'<figcaption>{REP_LABELS[rep]}: gray points are the other scales; colored points are the selected scale and retain their k=20 cluster colors.</figcaption></figure>'
        for rep in REPRESENTATIONS
    )

    css = """
:root{--ink:#17221d;--muted:#667069;--line:#d8ded9;--soft:#f3f6f4;--paper:#fff;--green:#315f4a}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#fbfcfb;color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--green)}.wrap{max-width:1220px;margin:auto;padding:0 28px}header{position:sticky;top:0;z-index:5;background:rgba(255,255,255,.96);border-bottom:1px solid var(--line)}nav{min-height:58px;display:flex;align-items:center;gap:20px;flex-wrap:wrap}.brand{font-weight:750;margin-right:auto}nav a{color:var(--ink);text-decoration:none;font-size:13px}
.hero{padding:66px 0 42px}.eyebrow{text-transform:uppercase;letter-spacing:.12em;color:var(--green);font-size:12px;font-weight:750}h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400}h1{font-size:clamp(2.6rem,6vw,5.3rem);line-height:1;max-width:1000px;margin:12px 0 20px}h2{font-size:clamp(1.7rem,3vw,2.6rem);margin:0 0 10px}h3{font-size:23px;margin:0}.lede,.section-note{color:var(--muted);max-width:880px}.question{margin-top:26px;padding:18px 20px;border-left:4px solid var(--green);background:var(--soft);font:21px/1.4 Georgia,serif}.section{padding:44px 0;border-top:1px solid var(--line)}
.flow{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:22px}.flow article{background:var(--paper);border:1px solid var(--line);padding:18px}.flow b{display:block;color:var(--green);font-size:12px;text-transform:uppercase;letter-spacing:.08em}.flow p{margin:8px 0 0;color:var(--muted)}
.decision{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:20px 0}.decision div{padding:16px;background:var(--soft);border-top:3px solid var(--green)}.decision b{display:block}.decision span{color:var(--muted);font-size:13px}.scale-block{padding:28px 0;border-top:1px solid var(--line)}.scale-copy{display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:start;margin-bottom:14px}.scale-dot{width:12px;height:12px;border-radius:50%;display:inline-block;margin-top:8px;flex:0 0 auto}.scale-copy h3 small{font:13px Inter,sans-serif;color:var(--muted);margin-left:8px}.scale-copy p{margin:4px 0;color:var(--muted);max-width:800px}.scale-copy strong{font-size:12px}.example-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:9px}.example{display:block;background:var(--paper);border:1px solid var(--line);color:var(--ink);text-decoration:none;overflow:hidden}.example img{display:block;width:100%;aspect-ratio:4/3;object-fit:cover;background:var(--soft)}.example div{padding:8px}.example b,.example span{display:block;font-size:11px}.example span{color:var(--muted)}.example p{font-size:10px;line-height:1.35;margin:6px 0 0}
.audit{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:20px}.stat{background:var(--paper);border:1px solid var(--line);padding:16px}.stat strong{display:block;font:31px Georgia,serif}.stat span{color:var(--muted);font-size:12px}.callout{padding:16px 18px;background:#fff8e9;border:1px solid #ead8ac;margin-top:16px}.figure{margin:20px 0;background:var(--paper);border:1px solid var(--line);padding:12px}.figure img{width:100%;display:block}.figure figcaption{padding:10px 5px 2px;color:var(--muted);font-size:12px}.rep-figures{display:grid;gap:18px}.rep-figures figure{margin:0;background:var(--paper);border:1px solid var(--line);padding:12px}.rep-figures img{width:100%;display:block}.rep-figures figcaption{color:var(--muted);font-size:12px;padding:8px 4px 0}.read-box{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin:18px 0}.read-box article{padding:17px;border-left:3px solid var(--green);background:var(--soft)}.read-box h3{font-size:18px}.read-box p{margin:5px 0 0;color:var(--muted)}
.before-after{display:grid;grid-template-columns:1fr 1.4fr;gap:18px;align-items:start}.before{background:var(--paper);border:1px solid var(--line);padding:10px}.before img{width:100%;display:block}.before p{font-size:12px;color:var(--muted)}.after{display:grid;gap:15px}.case-row{background:var(--paper);border:1px solid var(--line);padding:12px}.case-label{display:flex;align-items:center;gap:8px;margin-bottom:9px}.case-label .scale-dot{margin:0}.case-label small{color:var(--muted)}.case-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:7px}.case-image{color:var(--ink);text-decoration:none;border:1px solid var(--line);overflow:hidden}.case-image img{width:100%;aspect-ratio:1/1;object-fit:cover;display:block}.case-image span{display:grid;padding:5px;font-size:9px;line-height:1.3}.case-image b{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.case-image i{font-style:normal;color:var(--muted)}.next{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.next article{padding:18px;background:var(--soft);border-left:3px solid var(--green)}.next h3{font-size:19px}.next p{margin:5px 0 0;color:var(--muted)}footer{border-top:1px solid var(--line);padding:30px 0 60px;color:var(--muted);font-size:12px}
@media(max-width:900px){.example-grid{grid-template-columns:repeat(3,1fr)}.before-after{grid-template-columns:1fr}.audit,.flow{grid-template-columns:repeat(2,1fr)}}@media(max-width:620px){.wrap{padding:0 18px}.example-grid,.case-grid{grid-template-columns:repeat(2,1fr)}.audit,.flow,.decision,.read-box,.next{grid-template-columns:1fr}nav a{display:none}}
"""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Explaining species mixing · BioImages</title><style>{css}</style></head><body>
<header><div class="wrap"><nav><span class="brand">BioImages · scale-conditioned representation analysis</span><a href="#codebook">Scale labels</a><a href="review/">Review {len(review_sample)} images</a><a href="#representations">Representations</a><a href="#case">Cluster case</a><a href="#next">Next</a></nav></div></header>
<main class="wrap"><section class="hero"><div class="eyebrow">Phase 1 · visual definition before metrics</div><h1>Why are different species mixed in the same visual clusters?</h1><p class="lede">The current cluster atlas shows where species mix. This continuation tests whether photographic scale explains that mixing before introducing additional algorithms.</p><div class="question">Are images grouped because they show the same species, or because they were photographed at the same distance and composition?</div><div class="flow"><article><b>1 · Define</b><p>Assign each image a visible photographic scale using positive BioImages and Gemma evidence.</p></article><article><b>2 · Redraw</b><p>Color the existing embedding coordinates by scale and inspect each scale separately.</p></article><article><b>3 · Explain</b><p>Open one mixed cluster and reorganize its real images by scale and species.</p></article></div></section>
<section class="section" id="codebook"><h2>What “distant,” “mid-range,” and “close-up” mean</h2><p class="section-note">Photographic scale is independent of organ. A leaf may be a close-up, part of a branch-level mid-range view, or one small element in a whole-tree image.</p><div class="decision"><div><b>Whole organism visible?</b><span>Yes → distant</span></div><div><b>One diagnostic detail dominates?</b><span>Yes → close-up</span></div><div><b>Contextual plant part visible?</b><span>Yes → mid-range; otherwise uncertain</span></div></div>{''.join(codebook_sections)}
<div class="audit"><div class="stat"><strong>{confidence.get('high',0):,}</strong><span>BioImages + Gemma agreement</span></div><div class="stat"><strong>{confidence.get('medium',0):,}</strong><span>one-source provisional labels</span></div><div class="stat"><strong>{counts.get('uncertain',0):,}</strong><span>boundary/conflict images</span></div><div class="stat"><strong>{len(review_sample):,}</strong><span>balanced manual review rows</span></div></div><div class="callout"><strong>Label status:</strong> these labels are provisional. The review includes 25 high-confidence and 25 medium-confidence examples per main class, plus every uncertain image. <a href="review/"><strong>Open the human-friendly review interface →</strong></a> or <a href="data/scale_review_sample.csv">download the CSV</a>.</div></section>
<section class="section" id="representations"><h2>Do the frozen representations encode photographic scale?</h2><p class="section-note">First, use exactly the same 1,641 images and the same saved PCA coordinates. Change only the color to shot scale. Spatial separation of colors would indicate that scale is present in the embedding.</p><figure class="figure"><img src="assets/representation_by_scale.png" alt="Three representations colored by photographic scale"><figcaption>Figure 1. BioCLIP, DINOv3, and EfficientNet-B0. Every point is an image; color is the provisional shot-scale label.</figcaption></figure><div class="read-box"><article><h3>How to read Figure 1</h3><p>Look for large regions dominated by one color. Heavy overlap means the first two global principal components do not visibly separate scale.</p></article><article><h3>What it cannot prove</h3><p>A two-dimensional PCA view can hide structure in later dimensions. Local projection comes after the labels and case studies are validated.</p></article></div>
<h2>Where does each scale sit inside the existing clusters?</h2><p class="section-note">Each row below keeps the same global coordinate system. Colored points show one scale and use the existing k=20 cluster color; gray points provide the full-data reference.</p><div class="rep-figures">{rep_figures}</div></section>
<section class="section" id="case"><h2>Case study: BioCLIP cluster 04</h2><p class="section-note">This cluster contains lobed leaves, flowers, bark, whole trees, and several species. It is useful because both photographic scale and morphology may contribute to the mixing.</p><figure class="figure"><img src="assets/bioclip_cluster_04_case.png" alt="BioCLIP cluster 04 in PCA space"><figcaption>Figure 2. The cluster's location, then the same points colored by scale and by the four most common species. These are saved global PCA coordinates, not a new local projection.</figcaption></figure><div class="before-after"><div class="before"><h3>Before: centroid-nearest gallery</h3><img src="../representation_analysis/assets/cluster_sheets/bioclip/k20/cluster_04.jpg" alt="Original BioCLIP cluster 04 gallery"><p>The original atlas mixes every photographic scale in one gallery, so the reason for species mixing is hard to inspect.</p></div><div class="after"><h3>After: the same cluster grouped by scale</h3>{''.join(scale_rows)}</div></div><div class="callout"><strong>Question for review:</strong> after scale is separated, do the remaining cross-species groups share the same leaf shape, flower structure, bark texture, or background? This is the point where morphology can be distinguished from photographic scale.</div></section>
<section class="section" id="next"><h2>What comes after this page is reviewed</h2><div class="next"><article><h3>Scale-conditioned results</h3><p>Compute same-species/same-scale versus same-species/different-scale cosine similarities, then show accuracy bars with real correct and incorrect images.</p></article><article><h3>Local high-dimensional inspection</h3><p>For 3–5 validated mixed clusters, load the original frozen vectors, compare PC1–PC2, PC1–PC3, and PC2–PC3, then add rotation and seed-guided neighbor views.</p></article></div></section></main>
<footer><div class="wrap">Phase 1 uses existing BioImages metadata, image-grounded Gemma tags, frozen k=20 assignments, and saved global PCA coordinates. No model was retrained.</div></footer></body></html>"""


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    images = load_and_label()
    assignments = pd.read_csv(ROOT / "representation_analysis/data/cluster_assignments.csv")
    assignments = assignments[assignments["k"] == 20].copy()
    efficientnet_assignments = pd.read_csv(
        ROOT / "representation_analysis/data/efficientnet_k20_cluster_assignments.csv"
    )
    assignments = pd.concat([assignments, efficientnet_assignments], ignore_index=True)
    analysis_ids = set(assignments["image_id"])

    counts = build_group_counts(images, analysis_ids)
    image_output_columns = [
        "image_id",
        "file",
        "species",
        "organ_category",
        "subview",
        "primary_label",
        "group",
        "individual_id",
        "shot_scale",
        "shot_scale_confidence",
        "shot_scale_rationale",
        "source_distant_evidence",
        "gemma_distant_evidence",
        "source_mid_evidence",
        "gemma_mid_evidence",
        "source_close_evidence",
        "gemma_close_evidence",
        "needs_scale_review",
        "leaf_state",
        "reproductive_visibility",
        "background_context",
    ]
    images[image_output_columns].to_csv(DATA / "image_visual_groups.csv", index=False)
    counts.to_csv(DATA / "visual_group_counts.csv", index=False)
    make_explanatory_plots(images, assignments)
    codebook = build_codebook(images)
    (DATA / "scale_codebook.json").write_text(
        json.dumps(codebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    review_sample = build_scale_review_sample(images)
    review_sample.to_csv(DATA / "scale_review_sample.csv", index=False)
    html = build_explanatory_html(images, assignments, codebook, review_sample)
    (HERE / "index.html").write_text(html, encoding="utf-8")

    summary = {
        "images": len(images),
        "embedding_images": len(analysis_ids),
        "strict_test_images": 211,
        "shot_scale_counts": images["shot_scale"].value_counts().to_dict(),
        "representations": REPRESENTATIONS,
        "k": 20,
    }
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
