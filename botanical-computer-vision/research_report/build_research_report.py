#!/usr/bin/env python3
"""Build the frozen, traceable BioImages research-report bundle.

This script reads existing artifacts only.  It does not run inference, extract
embeddings, fit probes, or recompute clustering assignments.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research_report"
ASSETS = OUT / "assets"
DATA = OUT / "data"
CHARTS = ASSETS / "charts"
IMAGES = ASSETS / "images"
SHEETS = ASSETS / "cluster_sheets"

COLORS = {
    "BioCLIP": "#111111",
    "DINOv3": "#555555",
    "GPT-5.4": "#777777",
    "Gemini": "#999999",
    "EfficientNet-B0": "#aaaaaa",
}


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100 * float(x):.1f}%"


def rel_source(path: str, label: str | None = None) -> str:
    label = label or path
    return f'<a class="source" href="../{html.escape(path)}"><code>{html.escape(label)}</code></a>'


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bar_chart(path: Path, labels, series, title, subtitle, ylim=(0, 1), note=""):
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    for i, (name, values, color) in enumerate(series):
        pos = x - 0.4 + width / 2 + i * width
        bars = ax.bar(pos, values, width, label=name, color=color, edgecolor=color, linewidth=.4)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width()/2, v + 0.018, f"{v*100:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Accuracy")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=20)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#59636d")
    ax.grid(axis="y", alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=max(1, len(series)))
    if note:
        fig.text(.99, .01, note, ha="right", fontsize=8, color="#59636d")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def horizontal_bar(path: Path, labels, values, title, subtitle, color="#277a66"):
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, values, color="#444444")
    ax.set_yticks(y, labels)
    for b, v in zip(bars, values):
        ax.text(v + max(values)*.015, b.get_y()+b.get_height()/2, str(int(v)), va="center", fontsize=9)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=20)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#59636d")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def stacked_bar(path: Path, labels, parts, title, subtitle):
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    x = np.arange(len(labels)); bottom = np.zeros(len(labels))
    shades = ["#444444", "#aaaaaa"]
    for part_index, (name, values, color) in enumerate(parts):
        ax.bar(x, values, bottom=bottom, label=name, color=shades[part_index], edgecolor="#111111", linewidth=.4)
        for i, (b, v) in enumerate(zip(bottom, values)):
            if v >= .07:
                ax.text(i, b + v/2, f"{100*v:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += np.asarray(values)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of species-level errors")
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=20)
    ax.text(0, 1.01, subtitle, transform=ax.transAxes, fontsize=9, color="#59636d")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.14))
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def line_facets(path: Path, df: pd.DataFrame):
    names = {"bioclip":"BioCLIP", "dinov3":"DINOv3", "efficientnet_b0":"EfficientNet-B0"}
    metrics = [
        ("species_nmi_mean", "Species NMI"), ("organ_nmi_mean", "Organ/view NMI"),
        ("species_purity_mean", "Species purity"), ("organ_purity_mean", "Organ/view purity"),
        ("stability_pairwise_ari_mean", "Stability (pairwise ARI)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2)); axes=axes.ravel()
    for ax, (metric, title) in zip(axes, metrics):
        for key, grp in df.groupby("representation"):
            label=names[key]; xs=grp.k.to_numpy(); ys=grp[metric].to_numpy()
            std=grp[metric.replace("_mean","_std")].to_numpy()
            ax.errorbar(xs, ys, yerr=std, marker="o", lw=2, capsize=3, label=label, color=COLORS[label])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.set_xticks([10,20,50]); ax.set_xlabel("k"); ax.set_ylim(0, .9); ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center")
    fig.suptitle("Multi-seed clustering metrics", x=.07, ha="left", fontsize=16, fontweight="bold")
    fig.text(.07,.93,"Mean ± SD over seeds 42–51; n=1,641; L2-normalized embeddings; labels used after clustering only.",fontsize=9,color="#59636d")
    fig.tight_layout(rect=(0,.07,1,.91)); fig.savefig(path,dpi=170,bbox_inches="tight"); plt.close(fig)


CSS = r"""
:root{color-scheme:light;--ink:#17221d;--muted:#667069;--bg:#fff;--paper:#fff;--line:#d8ded9;--soft:#f4f6f4;--green:#315f4a;--green2:#edf4f0;--blue:#315f4a;--orange:#667069;--purple:#667069;--red:#a13a32;--correct:#18734f;--correct-bg:#e8f5ef;--wrong:#aa3832;--wrong-bg:#fbe9e7;--mono:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;font-size:15px;line-height:1.5}
a{color:var(--green);text-decoration-thickness:1px;text-underline-offset:3px}code{font-family:var(--mono);font-size:13px}.wrap{max-width:1180px;margin:auto;padding:0 28px}
header{background:rgba(255,255,255,.96);color:var(--ink);padding:14px 0;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10}.nav{display:flex;gap:22px;align-items:center;flex-wrap:wrap}.brand{font-weight:700;margin-right:auto}.nav a{color:var(--ink);text-decoration:none;font-size:13px}.nav a.active{color:var(--green);border-bottom:2px solid var(--green)}
.hero{padding:52px 0 30px}.eyebrow{text-transform:uppercase;letter-spacing:.1em;font-size:12px;color:var(--muted);font-weight:700}.hero h1,h1{font-family:Georgia,"Times New Roman",serif;font-size:clamp(2.15rem,4vw,3.7rem);font-weight:400;line-height:1.06;max-width:980px;margin:10px 0 14px}.lede{font-size:15px;color:var(--muted);max-width:880px}.meta{font-size:13px;color:var(--muted);margin-top:12px}
main{padding-bottom:70px}.section{padding:36px 0;border-top:1px solid var(--line)}h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:400}h2{font-size:clamp(1.55rem,2.5vw,2.15rem);line-height:1.2;margin:0 0 14px}h3{font-size:19px;margin:22px 0 8px}.section-intro{max-width:860px;color:var(--muted);margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:18px}.card{background:var(--paper);border:1px solid var(--line);padding:18px}.span4{grid-column:span 4}.span6{grid-column:span 6}.span8{grid-column:span 8}.span12{grid-column:1/-1}.stat .value{font-family:Georgia,"Times New Roman",serif;font-size:34px;font-weight:400;line-height:1.05;color:var(--green)}.stat .label{font-size:13px;color:var(--muted);margin-top:7px}.tag{display:inline-block;padding:3px 8px;background:var(--green2);font-size:12px;font-weight:700;color:var(--green)}.tag.warn,.tag.note{background:var(--soft);color:var(--ink)}
.result,.observation,.hypothesis,.question{border-left:3px solid var(--green);padding:10px 14px;margin:12px 0;background:var(--soft)}.hypothesis,.question{border-left-color:var(--muted)}.callout-label{font-size:11px;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:var(--muted)}
.chart{background:var(--paper);border:1px solid var(--line);padding:10px}.chart img{display:block;width:100%;height:auto}.caption{font-size:12px;color:var(--muted);margin:8px 4px 0}.table-wrap{overflow-x:auto;background:var(--paper);border:1px solid var(--line)}.table-wrap table{width:100%;border-collapse:collapse;font-size:13px}.table-wrap th,.table-wrap td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}.table-wrap th{background:var(--soft);color:var(--ink);font-size:12px;position:sticky;top:52px}.table-wrap tr:last-child td{border-bottom:0}.num{text-align:right!important;font-variant-numeric:tabular-nums}.source{font-size:11px;overflow-wrap:anywhere}.source code{background:var(--soft);padding:2px 5px}.metric-table td:nth-child(n+2){font-variant-numeric:tabular-nums}
.timeline{display:grid;grid-template-columns:130px 1fr;gap:0 20px}.time{color:var(--green);font-weight:700;padding:16px 0;border-right:2px solid var(--line)}.event{padding:16px 0 16px 20px;border-bottom:1px solid var(--line)}.event strong{display:block}.event small{color:var(--muted)}
.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;margin:18px 0}.flow .node{background:var(--paper);border:1px solid var(--line);padding:12px 16px;min-width:160px;flex:1}.arrow{align-self:center;color:var(--muted);font-weight:bold}
.motivation-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:18px 0 28px}.motivation-card{border-top:3px solid var(--green);background:var(--soft);padding:16px}.motivation-card h3{margin:0 0 7px;font-size:18px}.motivation-card p{margin:0;color:var(--muted);font-size:13px}.workflow-caption{font-size:12px;color:var(--muted);margin:8px 0}.workflow-code{margin:12px 0 30px;padding:20px 22px;border:1px solid var(--line);background:var(--paper);color:#111;font:12px/1.55 SFMono-Regular,Consolas,"Liberation Mono",Menlo,monospace;white-space:pre;overflow-x:auto}
.protocol{display:grid;grid-template-columns:190px 1fr;gap:8px 18px}.protocol dt{font-weight:700}.protocol dd{margin:0;color:var(--muted)}.prompt{white-space:pre-wrap;background:var(--soft);color:var(--ink);border:1px solid var(--line);padding:18px;font:12px/1.55 var(--mono);max-height:420px;overflow:auto}
.case-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}.case{background:var(--paper);border:1px solid var(--line);overflow:hidden}.case img{width:100%;aspect-ratio:4/3;object-fit:cover;background:var(--soft)}.case .body{padding:14px}.case .truth{font-weight:700}.case dl{font-size:12px;display:grid;grid-template-columns:80px 1fr;margin:8px 0;gap:2px 8px}.case dt{color:var(--muted)}.case dd{margin:0}.ok{color:var(--correct);font-weight:700}.bad{color:var(--wrong)}
.toc{columns:2;column-gap:32px}.toc li{margin-bottom:8px}.sheet-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.sheet{margin:0;background:var(--paper);border:1px solid var(--line);overflow:hidden}.sheet img{width:100%;display:block}.sheet figcaption{padding:8px 12px;font-size:12px;color:var(--muted)}details{background:var(--paper);border:1px solid var(--line);padding:12px 16px;margin:10px 0}summary{cursor:pointer;font-weight:700;color:var(--green)}
.footer{border-top:1px solid var(--line);padding:30px 0 60px;color:var(--muted);font-size:13px}.print-only{display:none}
.dashboard-section{display:grid;gap:18px}.control-row{display:flex;align-items:end;gap:12px;flex-wrap:wrap;margin:14px 0}.control-row label{display:grid;gap:5px;font-size:12px;color:var(--muted)}.control-row select,.control-row button{font:inherit;background:var(--paper);color:var(--ink);border:1px solid #aeb8b1;padding:8px 10px}.outcome-buttons{display:flex;gap:8px;flex-wrap:wrap}.outcome-buttons button{cursor:pointer}.outcome-buttons button.active{border-color:var(--green);box-shadow:inset 0 -3px var(--green)}.outcome-buttons .top1.active{box-shadow:inset 0 -3px var(--correct)}.outcome-buttons .top5.active{box-shadow:inset 0 -3px #a36d19}.outcome-buttons .miss.active{box-shadow:inset 0 -3px var(--wrong)}.explorer-status{display:flex;justify-content:space-between;gap:16px;align-items:center;margin:8px 0;font-size:13px}.pager{display:flex;align-items:center;gap:8px}.pager button{background:var(--paper);border:1px solid #aeb8b1;padding:5px 9px;cursor:pointer}.image-explorer-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.explorer-card{border:1px solid var(--line);min-width:0;background:var(--paper)}.explorer-card img{width:100%;height:220px;object-fit:contain;background:var(--soft);display:block;border-bottom:1px solid var(--line)}.explorer-card-body{padding:14px}.explorer-card h3{margin:3px 0 8px}.status-pill{display:inline-block;font-size:11px;font-weight:700;padding:3px 6px}.status-pill.top1{color:var(--correct);background:var(--correct-bg)}.status-pill.top5{color:#87590f;background:#fff3dc}.status-pill.miss{color:var(--wrong);background:var(--wrong-bg)}.prediction-fact{font-size:13px;margin:5px 0}.prediction-fact span{color:var(--muted);display:inline-block;width:78px}.explorer-card details{font-size:12px;margin-top:8px}.explorer-card ol{padding-left:22px}.explorer-card li.is-truth{color:var(--correct);font-weight:700}.analysis-pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.rank-row,.error-row,.organ-row-compact,.organ-compare-row{display:grid;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--line);font-size:13px}.rank-row{grid-template-columns:30px minmax(190px,1fr) 2fr 55px}.rank-bar,.error-bar,.mini-bar{height:8px;background:var(--soft)}.rank-bar span,.error-bar span,.mini-bar span{display:block;height:100%;background:var(--green)}.error-row{grid-template-columns:minmax(250px,1.7fr) 1fr 32px}.organ-row-compact{grid-template-columns:minmax(140px,1fr) 60px 2fr 60px 2fr 60px}.organ-compare-row{grid-template-columns:minmax(140px,1fr) repeat(3,1fr)}.organ-compare-cell{display:grid;grid-template-columns:1fr 46px;gap:7px;align-items:center}.subsection-head{display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:8px}.subsection-head h3{margin:0}.subsection-head button{font:inherit;background:var(--paper);color:var(--ink);border:1px solid #aeb8b1;padding:6px 9px;cursor:pointer}.legend-inline{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}.legend-inline i{display:inline-block;width:14px;height:4px;margin-right:5px;vertical-align:middle}.legend-inline .bio{background:#315f4a}.legend-inline .probe{background:#71847a}.legend-inline .dino{background:#a7b2ac}.organ-compare-cell .bio{background:#315f4a}.organ-compare-cell .probe{background:#71847a}.organ-compare-cell .dino{background:#a7b2ac}
@media(max-width:900px){.motivation-grid{grid-template-columns:1fr}}
@media(max-width:760px){.wrap{padding:0 16px}.hero{padding-top:42px}.span4,.span6,.span8{grid-column:1/-1}.nav{gap:12px}.nav a{font-size:12px}.toc{columns:1}.sheet-grid{grid-template-columns:1fr}.protocol{grid-template-columns:1fr}.protocol dd{margin-bottom:8px}.timeline{grid-template-columns:90px 1fr}.workflow-code{padding:14px;font-size:11px}.image-explorer-grid{grid-template-columns:1fr}.analysis-pair{grid-template-columns:1fr}.rank-row{grid-template-columns:28px minmax(130px,1fr) 55px}.rank-bar{display:none}.error-row{grid-template-columns:minmax(190px,1fr) 32px}.error-bar{display:none}.organ-row-compact{grid-template-columns:1fr 42px 60px 60px}.organ-row-compact .mini-bar{display:none}.organ-compare-row{grid-template-columns:1fr}.explorer-card img{height:min(72vw,360px)}}
@media print{header{position:static}.wrap{max-width:none}.section{break-inside:avoid}.card,.chart,.table-wrap{box-shadow:none}.print-only{display:block}}
"""


def nav(active: str) -> str:
    links=[("index.html","Start here"),("error_gallery.html","Full gallery"),("representation_analysis.html","Representation")]
    return '<header><div class="wrap nav"><div class="brand">BioImages Research Report</div>'+''.join(f'<a class="{"active" if href==active else ""}" href="{href}">{label}</a>' for href,label in links)+'</div></header>'


def page(title: str, active: str, body: str, description: str) -> str:
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="{html.escape(description)}"><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{nav(active)}{body}<footer class="footer"><div class="wrap">Frozen report generated from existing project artifacts on 2026-09-23. No new inference or training was run. See <a href="README.md">report index</a>.</div></footer></body></html>'''


def render_table(rows, columns, headers=None, numeric=(), raw=()):
    headers=headers or columns
    out=['<div class="table-wrap"><table><thead><tr>']
    out += [f'<th>{html.escape(str(h))}</th>' for h in headers]; out.append('</tr></thead><tbody>')
    for row in rows:
        out.append('<tr>')
        for col in columns:
            val=row.get(col,'') if isinstance(row,dict) else getattr(row,col)
            cls=' class="num"' if col in numeric else ''
            txt=str(val) if col in raw else html.escape(str(val))
            out.append(f'<td{cls}>{txt}</td>')
        out.append('</tr>')
    out.append('</tbody></table></div>')
    return ''.join(out)


def main():
    for d in (OUT,ASSETS,DATA,CHARTS,IMAGES,SHEETS): d.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","axes.labelcolor":"#37423d","xtick.color":"#4b5651","ytick.color":"#4b5651"})

    bio_full=read_json("outputs/bioclip25_baseline/summary.json")
    bio_zero=read_json("outputs/bioclip25_strict_zero_shot/summary.json")
    bio_probe=read_json("outputs/bioclip25_strict_embeddings/summary.json")
    dino=read_json("outputs/dinov3_strict_embeddings/summary.json")
    eff=read_json("outputs/efficientnet_b0_strict_embeddings/summary.json")
    eff_h=read_json("outputs/efficientnet_b0_strict_embeddings/efficientnet_control_summary.json")
    vlm=read_json("outputs/openrouter_vlm_full/summary.json")
    strict=read_json("outputs/strict_baseline_comparison/summary.json")
    audit=read_json("reports/openrouter_vlm_audit.json")
    err=read_json("error_analysis/summary.json")
    robust=read_json("representation_analysis/data/robustness_check.json")
    dashboard=read_json("app/dashboard-data.json")
    strict_dashboard=read_json("app/strict-comparison-data.json")

    # Registries
    experiments=[]
    def ex(order,eid,model,rep,task,split,candidates,protocol,metric,artifact,notes):
        experiments.append({"experiment_id":eid,"date_order":f"{order:02d}","model":model,"representation":rep,"task":task,"dataset_split":split,"candidate_set":candidates,"prompt_protocol":protocol,"metric":metric,"output_artifact":artifact,"notes_caveats":notes})
    ex(1,"bioclip_zs_full85","BioCLIP 2.5 ViT-H/14","biology-domain image-text","zero-shot species classification","all 1,899 images / 85 species","85 scientific + vernacular aliases","closed-set cosine; truth loaded after predictions","Top-1 77.5%; Top-5 93.2%","outputs/bioclip25_baseline/summary.json","Image-level; no observation-disjoint filtering")
    ex(2,"dinov3_probe_strict63","DINOv3 ViT-B/16 LVD-1689M","general self-supervised CLS","frozen linear probe","1,181 train / 249 val / 211 test; 63 species","63 species","observation-disjoint; C selected on val","Top-1 56.4%; Top-5 83.4%","outputs/dinov3_strict_embeddings/summary.json","Test labels loaded after predictions; outputs/dinov3_strict_linear_probe is a duplicate export, not a separate run")
    ex(3,"bioclip_zs_strict63","BioCLIP 2.5 ViT-H/14","biology-domain image-text","zero-shot species classification","211 strict test images / 63 observations","63 scientific + aliases","same strict test set; cosine softmax","Top-1 85.8%; Top-5 95.3%","outputs/bioclip25_strict_zero_shot/summary.json","Fair image/candidate comparison with strict probes")
    ex(4,"bioclip_probe_strict63","BioCLIP 2.5 ViT-H/14","biology-domain image embedding","frozen linear probe","1,181 / 249 / 211; individual-disjoint","63 species","L2 embedding; multinomial LR; val C grid","Top-1 87.2%; Top-5 96.2%","outputs/bioclip25_strict_embeddings/summary.json","Paired delta vs zero-shot +1.4 pp; outputs/bioclip25_strict_linear_probe is a duplicate export")
    for i,name in enumerate(["smoke","smoke_flex","smoke_minimal","smoke_cached","smoke_unique","smoke_final"],5):
        ex(i,f"openrouter_{name}","GPT-5.4 + Gemini 2.5 Flash Lite","direct VLM","implementation smoke check","3 images","85 species","historical protocol partially recoverable from summary/raw response","Not a benchmark","outputs/openrouter_vlm_%s/summary.json"%name,"Tiny operational run; retain negative/variable results; do not compare accuracy")
    ex(11,"openrouter_pilot_per_image","GPT-5.4 + Gemini 2.5 Flash Lite","direct VLM","63-species pilot","63 images; one per strict held-out observation","85 species; per-image deterministic order","temperature 0; strict JSON Top-5","GPT 16/63; Gemini 22/63 Top-1","outputs/openrouter_vlm_pilot/summary.json","GPT only 56/63 successful; denominator in summary remains 63")
    ex(12,"openrouter_pilot_fixed","GPT-5.4 + Gemini 2.5 Flash Lite","direct VLM","63-species pilot","63 images; same pilot selection","85 species; fixed deterministic order","minimal reasoning; auto detail; strict JSON","GPT 20/63; Gemini 23/63 Top-1","outputs/openrouter_vlm_pilot_final/summary.json","Gemini 61/63 successful")
    ex(13,"gpt_high_audit","GPT-5.4","direct VLM","reasoning protocol audit","63 requested","85 species; fixed order","high reasoning; original detail; 2,000-token cap","2/63 successful","outputs/openrouter_vlm_pilot_gpt_high_audit/summary.json","Negative operational result; most responses had null content")
    ex(14,"gpt_high_valid","GPT-5.4","direct VLM","reasoning protocol audit","63 requested","85 species; fixed order","high reasoning; original detail; 2,000-token cap","12/63 successful; 7/12 correct","outputs/openrouter_vlm_pilot_gpt_high_valid/summary.json","Inconclusive: completion subset is non-random")
    ex(15,"openrouter_full85","GPT-5.4 + Gemini 2.5 Flash Lite","direct VLM","full closed-set classification","all 1,899 images / 85 species","85 species; fixed deterministic order","temp 0; strict JSON; GPT minimal/auto","GPT 38.8/63.3%; Gemini 35.4/63.5%","outputs/openrouter_vlm_full/summary.json","Self-reported confidence; not calibrated")
    ex(16,"four_model_error_analysis","BioCLIP ZS + DINOv3 probe + GPT + Gemini","mixed","aligned error analysis","common strict 211-image test set","Bio/DINO 63; VLM 85","image and observation level alignment","buckets + hierarchical metrics","error_analysis/summary.json","Same images, but candidate-set mismatch prevents strict fairness")
    ex(17,"efficientnet_probe_strict63","EfficientNet-B0 ImageNet-1K","supervised CNN pooled embedding","frozen linear probe","same 1,181 / 249 / 211 split","63 species","same LR and C grid as DINO","Top-1 21.8%; Top-5 47.9%","outputs/efficientnet_b0_strict_embeddings/summary.json","Manifest validated byte-identical to DINO")
    ex(18,"apl_bioclip_dino_initial","BioCLIP + DINOv3","two frozen spaces","exploratory clustering and neighbors","all 1,641 strict-eligible images","labels not used in clustering","L2; k-means k=10/20/50; nearest neighbors","cluster/error summaries and galleries","representation_analysis/report.html","Initial APL-style analysis; exploratory and non-causal")
    ex(19,"efficientnet_k20_control","EfficientNet-B0 ImageNet-1K","supervised CNN pooled embedding","representation control + k=20 clustering","same 1,641 images","labels not used in clustering","same preprocessing family; k=20 control","probe + cluster summary + 20 sheets","outputs/efficientnet_b0_strict_embeddings/efficientnet_control_summary.json","Adds third representation; no fine-tuning")
    ex(20,"representation_robustness","BioCLIP + DINOv3 + EfficientNet","three frozen spaces","multi-seed clustering + distances","same 1,641 images","labels only post hoc","k=10/20/50; 10 seeds; cosine; PCA/UMAP","NMI/purity/stability/distances","representation_analysis/data/final_analysis_summary.json","Final robustness layer; 2D projections are visualization only")
    fields=list(experiments[0]); write_csv(DATA/"experiment_registry.csv",experiments,fields)

    # VLM prompt and candidate protocol registry from existing summaries.
    prompt_text=("Perform closed-set botanical species identification from the image. Use visible morphology only. "
                 "Select exactly five distinct candidates from the list, rank them most likely first, and return each scientific name exactly as written in ranked_species. "
                 "Put the five corresponding self-assessed scores between 0 and 1 in confidences; they need not sum to 1. "
                 "Do not use filenames, metadata, web search, or any information outside the image and candidate list.\n\nCandidates:\n- [85 scientific names, with vernacular alias in parentheses]")
    (DATA/"vlm_prompt_template.txt").write_text(prompt_text,encoding="utf-8")
    candidates=read_json("outputs/openrouter_vlm_full/candidate_species.json")
    fixed=sorted(candidates,key=lambda item:hashlib.sha256(f"42|candidate-order|{item['scientific_name']}".encode()).hexdigest())
    write_csv(DATA/"openrouter_fixed_candidate_order.csv",[{"position":i+1,**x} for i,x in enumerate(fixed)],["position","scientific_name","vernacular"])
    protocols=[]
    for run in sorted((ROOT/"outputs").glob("openrouter_vlm_*/summary.json")):
        d=json.loads(run.read_text()); run_id=run.parent.name
        for key,m in d.get("models",{}).items():
            protocols.append({"run_id":run_id,"stage":d.get("stage","unknown"),"model_key":key,"model_id":m.get("model_id",""),"requested":m.get("requested_images",""),"successful":m.get("successful_images",""),"candidate_count":d.get("candidate_species",85),"candidate_order":d.get("candidate_order_strategy","not persisted"),"reasoning_effort":d.get("gpt_reasoning_effort","not persisted" if key=="gpt_5_4" else "n/a"),"image_detail":d.get("gpt_image_detail","not persisted" if key=="gpt_5_4" else "provider default"),"temperature":"0 (runner)","output":"strict JSON Top-5 (runner)","prompt_provenance":"reconstructed from current runner; historical request prompt not stored verbatim","predictions":f"outputs/{run_id}/predictions.csv","caveat":"Summary is authoritative; fields shown as not persisted are not inferred."})
    write_csv(DATA/"prompt_candidate_protocol_registry.csv",protocols,list(protocols[0]))

    consistency=[
      {"metric":"BioCLIP zero-shot full","value":"77.5% Top-1","population":"1,899 images / 85 species","source":"outputs/bioclip25_baseline/summary.json","explanation":"Full original-image benchmark"},
      {"metric":"BioCLIP zero-shot strict","value":"85.8% Top-1","population":"211 images / 63 species","source":"outputs/bioclip25_strict_zero_shot/summary.json","explanation":"Different, observation-disjoint test subset and smaller candidate set"},
      {"metric":"BioCLIP linear probe strict","value":"87.2% Top-1","population":"211 images / 63 species","source":"outputs/bioclip25_strict_embeddings/summary.json","explanation":"Supervised probe trained on 1,430 train+validation images"},
      {"metric":"GPT full","value":"38.8% Top-1","population":"1,899 images / 85 species","source":"outputs/openrouter_vlm_full/summary.json","explanation":"Full fixed-order direct-VLM protocol"},
      {"metric":"GPT strict subset","value":"39.3% Top-1","population":"211 images but 85 candidates","source":"error_analysis/data/model_summary.csv","explanation":"Subset of the same full predictions; not rerun with 63 candidates"},
      {"metric":"Gemini full","value":"35.4% Top-1","population":"1,899 images / 85 species","source":"outputs/openrouter_vlm_full/summary.json","explanation":"Full fixed-order direct-VLM protocol"},
      {"metric":"Gemini strict subset","value":"40.3% Top-1","population":"211 images but 85 candidates","source":"error_analysis/data/model_summary.csv","explanation":"Subset of the same full predictions; not rerun with 63 candidates"},
    ]
    write_csv(DATA/"metric_consistency_audit.csv",consistency,list(consistency[0]))

    # Sensitivity examples (existing predictions only).
    per=pd.read_csv(ROOT/"outputs/openrouter_vlm_pilot/predictions.csv")
    fix=pd.read_csv(ROOT/"outputs/openrouter_vlm_pilot_final/predictions.csv")
    a=per[(per.model_key=="gpt_5_4")&(per.status=="ok")][["image_id","file","organ_category","ground_truth_species","predicted_species","correct"]]
    b=fix[(fix.model_key=="gpt_5_4")&(fix.status=="ok")][["image_id","predicted_species","correct"]]
    sens=a.merge(b,on="image_id",suffixes=("_per_image","_fixed")); sens=sens[sens.predicted_species_per_image!=sens.predicted_species_fixed].copy()
    priority=pd.concat([sens[(sens.correct_fixed)&(~sens.correct_per_image)],sens[(sens.correct_per_image)&(~sens.correct_fixed)],sens]).drop_duplicates("image_id")
    priority=priority[priority.image_id.map(lambda x:(Path("/tmp/bioimages-representation-cache")/(x.replace("/","__")+".jpg")).exists())].head(6)
    sensitivity=[]
    for _,r in priority.iterrows():
        order=sorted(candidates,key=lambda item:hashlib.sha256(f"42|{r.image_id}|{item['scientific_name']}".encode()).hexdigest())
        pos_per={x["scientific_name"]:i+1 for i,x in enumerate(order)}; pos_fix={x["scientific_name"]:i+1 for i,x in enumerate(fixed)}
        row={"image_id":r.image_id,"file":r.file,"organ_category":r.organ_category,"ground_truth_species":r.ground_truth_species,"per_image_prediction":r.predicted_species_per_image,"per_image_correct":bool(r.correct_per_image),"per_image_prediction_position":pos_per.get(r.predicted_species_per_image),"fixed_prediction":r.predicted_species_fixed,"fixed_correct":bool(r.correct_fixed),"fixed_prediction_position":pos_fix.get(r.predicted_species_fixed),"per_image_first_10":" | ".join(x["scientific_name"] for x in order[:10]),"fixed_first_10":" | ".join(x["scientific_name"] for x in fixed[:10])}
        sensitivity.append(row)
        safe=r.image_id.replace("/","__")+".jpg"; src=Path("/tmp/bioimages-representation-cache")/safe
        if src.exists(): shutil.copy2(src,IMAGES/safe)
    write_csv(DATA/"vlm_order_sensitivity_examples.csv",sensitivity,list(sensitivity[0]))

    # Copy self-contained and image assets.
    shutil.copy2(ROOT/"error_analysis/error_gallery.html",OUT/"error_gallery.html")
    # The source gallery also lives one directory above this report. Rewrite only
    # its navigation after copying so the shareable report remains self-contained.
    gallery_report = OUT / "error_gallery.html"
    gallery_html = gallery_report.read_text(encoding="utf-8").replace(
        'href="../research_report/', 'href="'
    )
    gallery_report.write_text(gallery_html, encoding="utf-8")
    gallery_images = OUT / "gallery_images"
    gallery_images.mkdir(parents=True, exist_ok=True)
    for src in (ROOT/"error_analysis/gallery_images").glob("*.jpg"):
        shutil.copy2(src,gallery_images/src.name)
    for src in (ROOT/"representation_analysis/assets/final_plots").glob("*.png"):
        shutil.copy2(src,ASSETS/src.name)
    for rep in ("bioclip","dinov3","efficientnet_b0"):
        (SHEETS/rep).mkdir(parents=True,exist_ok=True)
        for src in (ROOT/f"representation_analysis/assets/cluster_sheets/{rep}/k20").glob("*.jpg"):
            shutil.copy2(src,SHEETS/rep/src.name)
    reps=pd.read_csv(ROOT/"error_analysis/samples/representative_samples.csv")
    for _,r in reps.iterrows():
        src=ROOT/"error_analysis"/str(r.local_sample_path)
        if src.exists(): shutil.copy2(src,IMAGES/src.name)

    # Charts.
    g=vlm["models"]; bar_chart(CHARTS/"full85_accuracy.png",["BioCLIP zero-shot","GPT-5.4 direct VLM","Gemini direct VLM"],[("Top-1",[bio_full["top1_accuracy"],g["gpt_5_4"]["top1_accuracy"],g["gemini_flash_lite"]["top1_accuracy"]],"#277a66"),("Top-5",[bio_full["top5_accuracy"],g["gpt_5_4"]["top5_accuracy"],g["gemini_flash_lite"]["top5_accuracy"]],"#8eb9aa")],"Full 85-species benchmark","n=1,899 images; identical candidate taxonomy and image set",note="model families use different inference protocols")
    bar_chart(CHARTS/"strict63_accuracy.png",["BioCLIP zero-shot","BioCLIP probe","DINOv3 probe","EfficientNet-B0 probe"],[("Top-1",[bio_zero["top1_accuracy"],bio_probe["test_top1_accuracy"],dino["test_top1_accuracy"],eff["test_top1_accuracy"]],"#277a66"),("Top-5",[bio_zero["top5_accuracy"],bio_probe["test_top5_accuracy"],dino["test_top5_accuracy"],eff["test_top5_accuracy"]],"#8eb9aa")],"Strict 63-species test benchmark","n=211 test images; 63 observations; individual-disjoint; same candidate set",note="zero-shot vs trained probes are different learning protocols")
    tax_long=pd.read_csv(ROOT/"error_analysis/taxonomy_analysis/taxonomy_level_accuracy.csv")
    tax_meta=tax_long[["model_key","model","candidate_species"]].drop_duplicates()
    tax=tax_meta.merge(tax_long.pivot(index="model_key",columns="taxonomy_level",values="accuracy").reset_index(),on="model_key")
    tax=tax.rename(columns={"species":"species_top1_accuracy","genus":"genus_top1_accuracy","family":"family_top1_accuracy"})
    order=["bioclip","dinov3","gpt_5_4","gemini_flash_lite"]; t=tax.set_index("model_key").loc[order]
    bar_chart(CHARTS/"hierarchy_accuracy.png",["BioCLIP ZS\n63 candidates","DINOv3 probe\n63 candidates","GPT-5.4\n85 candidates","Gemini\n85 candidates"],[("Species",t.species_top1_accuracy.tolist(),"#277a66"),("Genus",t.genus_top1_accuracy.tolist(),"#3d6e9f"),("Family",t.family_top1_accuracy.tolist(),"#7b5aa6")],"Hierarchical Top-1 accuracy","same 211 strict images; taxonomy mapping applied post hoc",note="VLM candidate set remained 85; interpret cross-family gaps cautiously")
    br=pd.read_csv(ROOT/"error_analysis/taxonomy_analysis/species_error_taxonomy_breakdown.csv").set_index("model_key").loc[order]
    within=(br.correct_genus_wrong_species/br.species_error_count).tolist(); wrong=(br.wrong_genus/br.species_error_count).tolist()
    stacked_bar(CHARTS/"species_error_composition.png",["BioCLIP","DINOv3","GPT-5.4","Gemini"],[("Correct genus, wrong species",within,"#3d6e9f"),("Wrong genus",wrong,"#a8463b")],"Composition of species-level errors","denominators: "+", ".join(f"{m} n={int(n)}" for m,n in zip(["BioCLIP","DINO","GPT","Gemini"],br.species_error_count)))
    buckets=pd.read_csv(ROOT/"error_analysis/data/bucket_summary.csv")
    key_order=["all_four_correct","all_four_wrong","bioclip_only_correct","bioclip_wrong_vlm_rescue","vlm_embedding_consensus_conflict","embedding_disagreement"]
    bd=buckets.set_index("bucket").loc[key_order]
    horizontal_bar(CHARTS/"error_buckets.png",[x.replace("_"," ") for x in key_order],bd.image_count.tolist(),"Aligned error buckets","n=211 common images; buckets overlap unless definition is mutually exclusive","#3d6e9f")
    conf=pd.read_csv(ROOT/"error_analysis/taxonomy_analysis/within_genus_confusions.csv").head(12)
    horizontal_bar(CHARTS/"within_genus_confusions.png",[f"{r.ground_truth_species} → {r.predicted_species}" for _,r in conf.iterrows()],conf.total_occurrences.tolist(),"Most frequent within-genus confusions","summed occurrences across the four aligned baselines; directional pairs","#7b5aa6")
    ms=pd.read_csv(ROOT/"representation_analysis/data/multiseed_cluster_summary.csv"); line_facets(CHARTS/"clustering_multiseed.png",ms)
    dist=pd.read_csv(ROOT/"representation_analysis/data/distance_analysis_summary.csv")
    # compact original-space effect-size chart
    labels=["BioCLIP","DINOv3","EfficientNet-B0"]
    names={"BioCLIP":"bioclip","DINOv3":"dinov3","EfficientNet-B0":"efficientnet_b0"}
    def distance_value(rep, comparison, column):
        return float(dist[(dist.representation==rep)&(dist.comparison==comparison)].iloc[0][column])
    bar_chart(CHARTS/"distance_effects.png",labels,[("Species Cohen's d",[distance_value(names[x],"species","cohens_d") for x in labels],"#277a66"),("Organ/view Cohen's d",[distance_value(names[x],"normalized_organ","cohens_d") for x in labels],"#3d6e9f")],"Original-space cosine separation","all unordered cross-observation pairs; same masks in each representation",ylim=(0,2.1),note="Effect size compares same-label vs different-label cosine distributions")

    # Meeting summary page.
    timeline=''.join([f'<div class="time">{n}</div><div class="event"><strong>{title}</strong><small>{desc}</small></div>' for n,title,desc in [
      ("0 · Prototype","Evidence-bag species reasoning","Original image plus Gemma → GroundingDINO → SAM polygon evidence was sent to Gemini. The 85-image blind pilot reached 68.24% Top-1, but had no ablation to identify what drove the prediction."),
      ("1 · BioCLIP","Zero-shot baseline","Run directly on all 1,899 images with the full 85-species candidate set."),
      ("2 · Frozen probes","BioCLIP and DINOv3","Use an observation-disjoint split to test frozen embeddings with the same linear classifier."),
      ("3 · Direct VLMs","GPT-5.4 and Gemini","Test image-to-species prediction from the original image and a fixed candidate list."),
      ("4 · Error audit","Aligned model failures","Compare disagreement, hierarchical accuracy, and recurring species confusions."),
      ("5 · CNN control","EfficientNet-B0","Add a conventional ImageNet-supervised frozen representation on the same strict split."),
      ("6 · Structure","Clustering and distances","Compare the visual organization of BioCLIP, DINOv3, and EfficientNet embeddings.")]])
    research_flow='''<pre class="workflow-code" aria-label="Research direction and additions made during the baseline study">ORIGINAL FIVE-TRACK IDEA / PRIOR EXPLORATION

BioImages · 1,899 original images
|
+-- [1] BASELINE ................. Original image → BioCLIP → species prediction
|
+-- [2] GLOBAL REPRESENTATION .... Original image → DINOv3 embedding → classifier
|
+-- [3] SEGMENTED REPRESENTATION . SAM regions → DINOv3 embeddings → classifier
|
+-- [4] EVIDENCE REASONING ........ Original image + [Gemma → GroundingDINO → SAM polygons]
|                                  → Gemini → species prediction
|
`-- [5] WEAK SUPERVISION .......... SAM regions → DINO embeddings → clustering → prototypes
    
    :  ADDED WHILE BUILDING THE BASELINE STUDY
    :
    + - - Direct VLM controls ...... Original image + candidates → GPT-5.4 / Gemini
    |
    + - - BioCLIP linear probe ..... Frozen BioCLIP embedding → linear classifier
    |
    + - - EfficientNet control ..... Frozen ImageNet CNN embedding → linear classifier
    |
    ` - - Representation analysis . BioCLIP / DINO / EfficientNet → clustering + distances
                                      |
                                      `→ PCA / UMAP + contact sheets + error-enriched clusters

All completed baseline routes → BioImages ground truth → Top-1 / Top-5 + genus / family + error analysis</pre>'''
    organ_labels={"leaf":"Leaf","whole plant":"Whole plant / distant view","inflorescence":"Inflorescence","fruit":"Fruit","twig":"Twig","bark":"Bark","cone":"Cone","seed":"Seed","other":"Other","whole tree":"Whole tree","whole tree (or vine)":"Whole tree / vine","stem":"Stem","unspecified":"Unspecified"}
    organ_rows=''.join(f'''<div class="organ-row-compact"><strong>{html.escape(organ_labels.get(r['name'],r['name']))}</strong><span>n={r['total']}</span><div class="mini-bar"><span style="width:{100*r['top1_accuracy']:.1f}%"></span></div><strong>{pct(r['top1_accuracy'])}</strong><div class="mini-bar"><span style="width:{100*r['top5_accuracy']:.1f}%"></span></div><strong>{pct(r['top5_accuracy'])}</strong></div>''' for r in dashboard["organs"])
    max_error=max(r["count"] for r in dashboard["errors"])
    error_rows=''.join(f'''<div class="error-row"><span><i>{html.escape(r['truth'])}</i> → <i>{html.escape(r['predicted'])}</i></span><div class="error-bar"><span style="width:{100*r['count']/max_error:.1f}%"></span></div><strong>{r['count']}</strong></div>''' for r in dashboard["errors"][:10])
    strict_organ_rows=''
    for r in strict_dashboard["organs"]:
        strict_organ_rows+=f'''<div class="organ-compare-row"><strong>{html.escape(organ_labels.get(r['organ_category'],r['organ_category']))} <small>n={r['total']}</small></strong><div class="organ-compare-cell"><div class="mini-bar"><span class="bio" style="width:{100*r['bioclip_zero_top1_accuracy']:.1f}%"></span></div><span>{pct(r['bioclip_zero_top1_accuracy'])}</span></div><div class="organ-compare-cell"><div class="mini-bar"><span class="probe" style="width:{100*r['bioclip_probe_top1_accuracy']:.1f}%"></span></div><span>{pct(r['bioclip_probe_top1_accuracy'])}</span></div><div class="organ-compare-cell"><div class="mini-bar"><span class="dino" style="width:{100*r['dinov3_probe_top1_accuracy']:.1f}%"></span></div><span>{pct(r['dinov3_probe_top1_accuracy'])}</span></div></div>'''
    compact_images=[{k:r.get(k) for k in ("id","truth","vernacular","organ","predicted","confidence","top5","rank","outcome","thumbnail","image")} for r in dashboard["images"]]
    compact_species=[{k:r.get(k) for k in ("name","vernacular","total","top1_accuracy","top5_accuracy")} for r in dashboard["species"]]
    explorer_json=json.dumps({"images":compact_images,"species":compact_species,"organs":organ_labels},ensure_ascii=False,separators=(",",":" )).replace("</","<\\/")
    species_options=''.join(f'<option value="{html.escape(r["name"])}">{html.escape(r["name"])}</option>' for r in sorted(dashboard["species"],key=lambda x:x["name"]))
    organ_options=''.join(f'<option value="{html.escape(r["name"])}">{html.escape(organ_labels.get(r["name"],r["name"]))}</option>' for r in dashboard["organs"])
    index_body=f'''<div class="wrap"><section class="hero"><h1>BioImages species identification baselines</h1><p class="meta">Dataset: 1,899 original images · 85 species. Strict representation protocol: 1,641 images · 63 species · 211-image observation-disjoint test set.</p></section>
    <main><section class="section"><div class="grid"><div class="card stat span4"><div class="value">77.5%</div><div class="label">BioCLIP zero-shot Top-1 · full 85-class benchmark</div></div><div class="card stat span4"><div class="value">87.2%</div><div class="label">BioCLIP frozen probe Top-1 · strict 63-class test</div></div><div class="card stat span4"><div class="value">17 / 211</div><div class="label">Images all four aligned baselines missed</div></div></div></section>
    <section class="section"><h2>Baseline results</h2><div class="grid"><div class="chart span6"><img src="assets/charts/full85_accuracy.png" alt="Full 85 species accuracy"><p class="caption">1,899 images · 85 candidates.</p></div><div class="chart span6"><img src="assets/charts/strict63_accuracy.png" alt="Strict 63 species accuracy"><p class="caption">211 strict test images · 63 candidates.</p></div></div></section>
    <section class="section"><h2>Research question</h2><div class="flow"><div class="node"><strong>Original image</strong><br><small>image or observation</small></div><div class="arrow">→</div><div class="node"><strong>Frozen/direct model</strong><br><small>no backbone fine-tuning</small></div><div class="arrow">→</div><div class="node"><strong>Species prediction</strong><br><small>Top-1 / Top-5</small></div><div class="arrow">→</div><div class="node"><strong>Error & representation analysis</strong></div></div></section>
    <section class="section" id="research-timeline"><h2>Research timeline & motivation</h2>
      <div class="motivation-grid">
        <div class="motivation-card"><h3>Fast end-to-end prototype</h3><p>The original image and Gemma → GroundingDINO → SAM polygon evidence were both sent to Gemini. Gemini was the lowest-cost bridge from region-level evidence to a species decision before building a knowledge graph or learned decision layer.</p></div>
        <div class="motivation-card"><h3>Blind pilot result</h3><p>One image per species: Top-1 58/85 (68.24%), Top-5 71/85 (83.53%), genus 68/85 (80.00%), and family 69/85 (81.18%). This showed that the pipeline could run end to end.</p></div>
        <div class="motivation-card"><h3>Why establish baselines?</h3><p>The prototype had no original-only, evidence-only, or controlled comparison. Its result could reflect the original image, polygon crops, prompt, or language priors, so it is retained as feasibility evidence—not a reliable model comparison.</p></div>
      </div>
      <p class="caption">Historical stratified pilot: 85 images · 85 species · labels revealed only after predictions were frozen. Forced-choice Top-1 includes the best guess on abstained cases. This pilot is not part of the final 1,899-image baseline.</p>
      <p class="workflow-caption"><strong>Solid branches</strong> are the original five-track idea; <strong>dashed branches</strong> are additions made while establishing the current baseline study.</p>
      {research_flow}
      <h3>Completed baseline study</h3><div class="timeline">{timeline}</div>
    </section>
    <section class="section"><h2>Hierarchical results & error patterns</h2><div class="grid"><div class="chart span6"><img src="assets/charts/hierarchy_accuracy.png" alt="Species genus family accuracy"></div><div class="chart span6"><img src="assets/charts/error_buckets.png" alt="Error bucket counts"></div></div><div class="observation"><div class="callout-label">Observation</div>Relaxing species to genus/family improves every model. Gemini has the largest within-genus share among its species errors (37.3%); BioCLIP has 26.7%.</div><p><a href="error_gallery.html"><strong>Open the complete 6-bucket error gallery →</strong></a></p></section>
    <section class="section dashboard-section" id="image-explorer"><h2>Image-level BioCLIP predictions</h2><div class="control-row"><div class="outcome-buttons" aria-label="Prediction outcome"><button type="button" class="top1 active" data-outcome="top1">Top-1 correct · {dashboard['outcomes']['top1']}</button><button type="button" class="top5" data-outcome="top5">Top-5 rescue · {dashboard['outcomes']['top5']}</button><button type="button" class="miss" data-outcome="miss">Top-5 miss · {dashboard['outcomes']['miss']}</button></div><label>Ground-truth species<select id="species-filter"><option value="all">All species</option>{species_options}</select></label><label>Organ / view<select id="organ-filter"><option value="all">All organs</option>{organ_options}</select></label></div><div class="explorer-status"><span id="explorer-summary"></span><div class="pager"><button type="button" id="explorer-prev" aria-label="Previous page">←</button><span id="explorer-page"></span><button type="button" id="explorer-next" aria-label="Next page">→</button></div></div><div id="image-explorer-grid" class="image-explorer-grid"></div><p class="caption">Full 1,899-image BioCLIP zero-shot run. Select an outcome, species, and organ; expand any card to inspect its Top-5 ranking.</p></section>
    <section class="section"><h2>Organ difficulty varies substantially</h2><div class="card"><div class="organ-row-compact"><strong>Organ / view</strong><span>Images</span><strong></strong><strong>Top-1</strong><strong></strong><strong>Top-5</strong></div>{organ_rows}</div><p class="caption">BioCLIP zero-shot · full 1,899-image, 85-species benchmark.</p></section>
    <section class="section"><div class="analysis-pair"><div class="card"><div class="subsection-head"><div><h3>Species difficulty ranking</h3><p class="caption">Accuracy is shown with sample size.</p></div><button type="button" id="difficulty-toggle">Show easiest</button></div><div id="species-ranking"></div></div><div class="card"><div class="subsection-head"><div><h3>Most common error directions</h3><p class="caption">Ground truth → BioCLIP Top-1 prediction.</p></div></div>{error_rows}</div></div></section>
    <section class="section"><h2>Top-1 accuracy by organ</h2><div class="subsection-head"><p class="caption">Same strict 211-image test set. Small organ groups should be interpreted cautiously.</p><div class="legend-inline"><span><i class="bio"></i>BioCLIP zero-shot</span><span><i class="probe"></i>BioCLIP probe</span><span><i class="dino"></i>DINOv3 probe</span></div></div><div class="card">{strict_organ_rows}</div></section>
    <section class="section"><h2>Representation findings</h2><div class="grid"><div class="card span4"><span class="tag">Species association</span><h3>BioCLIP strongest</h3><p>Highest species NMI and purity at k=10, 20, and 50; largest original-space species separation.</p></div><div class="card span4"><span class="tag note">Organ/view association</span><h3>DINOv3 / EfficientNet stronger</h3><p>EfficientNet leads at k=10; DINOv3 at k=20/50. EfficientNet has the largest original-space organ effect size.</p></div><div class="card span4"><span class="tag warn">Scope</span><h3>Dataset-specific</h3><p>These are associations in this dataset, not causal claims.</p></div></div><p><a href="representation_analysis.html"><strong>Open the representation analysis →</strong></a></p></section>
    <section class="section"><h2>What we know</h2><div class="result"><div class="callout-label">Result</div>BioCLIP is the strongest completed species classifier in both full zero-shot and strict frozen-probe protocols.</div><div class="observation"><div class="callout-label">Observation</div>Direct-VLM predictions vary with candidate order and generation protocol; they are not pure visual-recognition measurements.</div><div class="observation"><div class="callout-label">Observation</div>BioCLIP associates more strongly with species identity; DINOv3 and EfficientNet associate more strongly with organ/view under the recorded metrics.</div></section>
    <section class="section"><h2>Within-genus confusion pairs</h2><div class="chart"><img src="assets/charts/within_genus_confusions.png" alt="Within genus confusion pairs"></div></section>
    <section class="section"><p><a href="benchmark_protocol.html">Benchmark and protocol registry</a> · <a href="README.md">Report index and source artifacts</a></p></section></main><script id="explorer-data" type="application/json">{explorer_json}</script><script>
    (() => {{
      const data=JSON.parse(document.getElementById('explorer-data').textContent);
      const grid=document.getElementById('image-explorer-grid');
      const species=document.getElementById('species-filter');
      const organ=document.getElementById('organ-filter');
      const summary=document.getElementById('explorer-summary');
      const pageLabel=document.getElementById('explorer-page');
      const prev=document.getElementById('explorer-prev');
      const next=document.getElementById('explorer-next');
      const buttons=[...document.querySelectorAll('[data-outcome]')];
      let outcome='top1', pageIndex=0, difficulty='hard';
      const esc=(value)=>String(value??'').replace(/[&<>"']/g,(c)=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
      const percent=(value)=>`${{(Number(value)*100).toFixed(1)}}%`;
      function filtered() {{
        return data.images.filter((item)=>item.outcome===outcome&&(species.value==='all'||item.truth===species.value)&&(organ.value==='all'||item.organ===organ.value)).sort((a,b)=>outcome==='top5'?((a.rank||6)-(b.rank||6)||b.confidence-a.confidence):b.confidence-a.confidence);
      }}
      function renderExplorer() {{
        const rows=filtered(); const pages=Math.max(1,Math.ceil(rows.length/9)); pageIndex=Math.min(pageIndex,pages-1);
        const shown=rows.slice(pageIndex*9,pageIndex*9+9); const labels={{top1:'Top-1 correct',top5:'Top-5 rescue',miss:'Top-5 miss'}};
        summary.innerHTML=`<strong>${{rows.length}}</strong> matching images · ${{labels[outcome]}}`;
        pageLabel.textContent=`${{pageIndex+1}} / ${{pages}}`; prev.disabled=pages===1; next.disabled=pages===1;
        grid.innerHTML=shown.map((item)=>`<article class="explorer-card"><a href="${{esc(item.image)}}" target="_blank" rel="noreferrer"><img src="${{esc(item.thumbnail)}}" alt="${{esc(item.truth)}}" loading="lazy"></a><div class="explorer-card-body"><span class="status-pill ${{item.outcome}}">${{labels[item.outcome]}}</span><h3><i>${{esc(item.truth)}}</i></h3><p class="caption">${{esc(item.id)}} · ${{esc(data.organs[item.organ]||item.organ)}}</p><p class="prediction-fact"><span>Top-1</span><strong><i>${{esc(item.predicted)}}</i></strong></p><p class="prediction-fact"><span>Confidence</span><strong>${{percent(item.confidence)}}</strong></p><details><summary>Show Top-5</summary><ol>${{item.top5.map((p,i)=>`<li class="${{p.species===item.truth?'is-truth':''}}">${{i+1}}. <i>${{esc(p.species)}}</i> · ${{percent(p.confidence)}}</li>`).join('')}}</ol></details></div></article>`).join('')||'<p>No images match this filter.</p>';
      }}
      buttons.forEach((button)=>button.addEventListener('click',()=>{{outcome=button.dataset.outcome;pageIndex=0;buttons.forEach((b)=>b.classList.toggle('active',b===button));renderExplorer();}}));
      species.addEventListener('change',()=>{{pageIndex=0;renderExplorer();}}); organ.addEventListener('change',()=>{{pageIndex=0;renderExplorer();}});
      prev.addEventListener('click',()=>{{const pages=Math.max(1,Math.ceil(filtered().length/9));pageIndex=(pageIndex-1+pages)%pages;renderExplorer();}}); next.addEventListener('click',()=>{{const pages=Math.max(1,Math.ceil(filtered().length/9));pageIndex=(pageIndex+1)%pages;renderExplorer();}});
      function renderRanking() {{ const rows=[...data.species].sort((a,b)=>difficulty==='hard'?(a.top1_accuracy-b.top1_accuracy||b.total-a.total):(b.top1_accuracy-a.top1_accuracy||b.total-a.total)).slice(0,12); document.getElementById('species-ranking').innerHTML=rows.map((r,i)=>`<div class="rank-row"><span>${{String(i+1).padStart(2,'0')}}</span><span><i>${{esc(r.name)}}</i><br><small>${{esc(r.vernacular)}} · n=${{r.total}}</small></span><div class="rank-bar"><span style="width:${{100*r.top1_accuracy}}%"></span></div><strong>${{percent(r.top1_accuracy)}}</strong></div>`).join(''); }}
      document.getElementById('difficulty-toggle').addEventListener('click',(event)=>{{difficulty=difficulty==='hard'?'easy':'hard';event.currentTarget.textContent=difficulty==='hard'?'Show easiest':'Show hardest';renderRanking();}});
      renderExplorer(); renderRanking();
    }})();
    </script></div>'''
    (OUT/"index.html").write_text(page("Start Here · BioImages Research Report","index.html",index_body,"Meeting summary and index"),encoding="utf-8")

    # Benchmark page.
    ex_rows=[]
    for r in experiments:
        q=r.copy(); q["output_artifact"]=rel_source(r["output_artifact"]); ex_rows.append(q)
    registry_table=render_table(ex_rows,["experiment_id","date_order","model","task","dataset_split","candidate_set","prompt_protocol","metric","output_artifact","notes_caveats"],["Experiment","Order","Model","Task","Dataset / split","Candidates","Prompt / protocol","Recorded metric","Source","Caveat"],raw=("output_artifact",))
    protocol_rows=[]
    for r in protocols:
        q=r.copy(); q["predictions"]=rel_source(r["predictions"]); protocol_rows.append(q)
    prompt_table=render_table(protocol_rows,["run_id","model_id","stage","requested","successful","candidate_order","reasoning_effort","image_detail","predictions","caveat"],["Run","Model","Stage","Requested","OK","Candidate order","Reasoning","Image detail","Predictions","Caveat"],numeric=("requested","successful"),raw=("predictions",))
    sens_cards=''.join(f'''<article class="case"><img src="assets/images/{html.escape(r['image_id'].replace('/','__'))}.jpg" alt="{html.escape(r['image_id'])}"><div class="body"><div class="truth"><i>{html.escape(r['ground_truth_species'])}</i></div><div class="meta">{html.escape(r['image_id'])} · {html.escape(r['organ_category'])}</div><dl><dt>Per-image</dt><dd class="{'ok' if r['per_image_correct'] else 'bad'}">{html.escape(r['per_image_prediction'])} {'✓' if r['per_image_correct'] else '✗'}</dd><dt>Fixed</dt><dd class="{'ok' if r['fixed_correct'] else 'bad'}">{html.escape(r['fixed_prediction'])} {'✓' if r['fixed_correct'] else '✗'}</dd></dl></div></article>''' for r in sensitivity)
    consistency_rows=[]
    for r in consistency:
        q=r.copy(); q["source"]=rel_source(r["source"]); consistency_rows.append(q)
    consistency_table=render_table(consistency_rows,["metric","value","population","explanation","source"],["Metric","Value","Population","Why it differs","Source"],raw=("source",))
    execution_rows=[
      {"run":"BioCLIP zero-shot full","scope":"1,899 images","model_work":"21.9 s GPU / 55.8 s wall inference","total":"98.8 s","cost":"local/cloud GPU run; API cost n/a","source":rel_source('outputs/bioclip25_baseline/summary.json')},
      {"run":"BioCLIP zero-shot strict","scope":"211 test images","model_work":"2.4 s GPU / 6.1 s wall inference","total":"39.1 s","cost":"API cost n/a","source":rel_source('outputs/bioclip25_strict_zero_shot/summary.json')},
      {"run":"BioCLIP frozen embeddings + probe","scope":"1,641 embeddings + probe","model_work":"20.0 s GPU / 53.3 s wall embedding; 2.1 s probe","total":"88.7 s","cost":"API cost n/a","source":rel_source('outputs/bioclip25_strict_embeddings/summary.json')},
      {"run":"DINOv3 frozen embeddings + probe","scope":"1,641 embeddings + probe","model_work":"2.5 s GPU / 27.7 s wall embedding; 2.1 s probe","total":"44.4 s","cost":"API cost n/a","source":rel_source('outputs/dinov3_strict_embeddings/summary.json')},
      {"run":"EfficientNet-B0 embeddings + probe","scope":"1,641 embeddings + probe","model_work":"0.6 s GPU / 23.4 s wall embedding; 2.7 s probe","total":"38.2 s","cost":"API cost n/a","source":rel_source('outputs/efficientnet_b0_strict_embeddings/summary.json')},
      {"run":"GPT-5.4 direct VLM full","scope":"1,899 requests","model_work":"6.76 s mean latency; 1,292.5 s model wall","total":"within combined 1,571.4 s run","cost":"$8.0747","source":rel_source('outputs/openrouter_vlm_full/summary.json')},
      {"run":"Gemini 2.5 Flash Lite full","scope":"1,899 requests","model_work":"1.40 s mean latency; 271.4 s model wall","total":"within combined 1,571.4 s run","cost":"$0.6042","source":rel_source('outputs/openrouter_vlm_full/summary.json')},
    ]
    execution_table=render_table(execution_rows,["run","scope","model_work","total","cost","source"],["Run","Scope","Recorded model work","Total run","API cost","Source"],raw=("source",))
    benchmark_body=f'''<div class="wrap"><section class="hero"><div class="eyebrow">HTML 1 · protocol and traceability</div><h1>Benchmark & Experimental Protocol</h1><p class="lede">All completed model runs, including smoke tests, partial failures, and protocol caveats. No result is silently substituted for another.</p></section><main>
    <section class="section"><h2>Dataset and canonical units</h2><dl class="protocol"><dt>Full corpus</dt><dd>1,899 original images, 85 species.</dd><dt>Strict eligible corpus</dt><dd>1,641 images, 63 species with ≥3 individual/observation groups.</dd><dt>Strict split</dt><dd>1,181 train, 249 validation, 211 test; one validation and one test individual per species; seed 42; no individual overlap.</dd><dt>Image-level</dt><dd>Every image is scored independently.</dd><dt>Observation-level</dt><dd>Strict-test images are grouped into 63 observations; majority vote with score-based tie breaking in the error analysis.</dd><dt>Truth isolation</dt><dd>BioCLIP zero-shot and frozen-probe scripts remove test truth before prediction, then reload it for evaluation. VLM requests contain image plus candidates, never per-image truth.</dd></dl><p>{rel_source('outputs/dinov3_strict_embeddings/split_manifest.csv')} · {rel_source('scripts/modal_dinov3_strict.py')}</p></section>
    <section class="section"><h2>Which numbers can be compared?</h2><div class="grid"><div class="card span4"><span class="tag">Directly paired</span><h3>Strict 63-class set</h3><p>BioCLIP zero-shot, BioCLIP probe, DINOv3 probe, and EfficientNet probe use the same 211 images and 63 labels. Training mode still differs.</p></div><div class="card span4"><span class="tag note">Same full inputs</span><h3>Full 85-class set</h3><p>BioCLIP zero-shot, GPT, and Gemini see the same 1,899 images and species taxonomy; their inference mechanism and score semantics differ.</p></div><div class="card span4"><span class="tag warn">Not fully fair</span><h3>Four-model error subset</h3><p>Images align exactly, but BioCLIP/DINO use 63 candidates while GPT/Gemini predictions came from the 85-candidate full run.</p></div></div></section>
    <section class="section"><h2>Headline benchmark results</h2><div class="grid"><div class="chart span6"><img src="assets/charts/full85_accuracy.png" alt="Full benchmark"></div><div class="chart span6"><img src="assets/charts/strict63_accuracy.png" alt="Strict benchmark"></div></div><div class="result"><div class="callout-label">Result</div>On the paired strict test, BioCLIP probe exceeds BioCLIP zero-shot by 1.4 percentage points; paired bootstrap 95% CI −3.3 to +6.2 pp and McNemar p=0.690. The existing evidence does not establish a reliable probe advantage.</div><p>{rel_source('outputs/strict_baseline_comparison/summary.json')}</p></section>
    <section class="section"><h2>Recorded runtime and cost</h2><p class="section-intro">These are end-to-end values preserved by each run, not standardized hardware benchmarks. GPU time, wall time, and API latency are kept distinct.</p>{execution_table}</section>
    <section class="section"><h2>Model-by-model protocol</h2>
      <details open><summary>BioCLIP 2.5 zero-shot</summary><p>Official OpenCLIP transforms; ViT-H/14 image embedding; class text is the scientific name plus vernacular alias when present; text vectors are normalized, averaged, then renormalized; image/text cosine logits use the learned logit scale and a closed-set softmax. Backbone frozen. Ground truth read only after Top-5 lists exist.</p><p>{rel_source('scripts/modal_bioclip_strict.py')} · {rel_source('outputs/bioclip25_baseline/summary.json')}</p></details>
      <details><summary>BioCLIP / DINOv3 / EfficientNet frozen probes</summary><p>BioCLIP: normalized 1,024-D image embedding. DINOv3: Hugging Face processor and normalized 768-D <code>pooler_output</code>. EfficientNet-B0: ImageNet-1K V1 transforms and normalized 1,280-D global-average-pooled penultimate feature. Each uses multinomial logistic regression (<code>lbfgs</code>, max_iter 3,000); C ∈ {{0.01, 0.1, 1, 10, 100}} selected on validation macro accuracy, then refit on train+validation. C=100 for all three.</p><p>{rel_source('scripts/modal_dinov3_strict.py')}</p></details>
      <details><summary>GPT-5.4 and Gemini 2.5 Flash Lite direct VLM</summary><p>Each input is EXIF-transposed, RGB converted, resized so longest side ≤1,536 px, and re-encoded JPEG quality 90 without metadata. The image and 85-name candidate prompt are sent once. Temperature 0; strict JSON schema requires five distinct in-list species and five self-assessed scores. GPT full run: minimal reasoning, auto detail, flex tier, 700 completion tokens. Gemini: 300 completion tokens. Parsing rejects missing/duplicate/out-of-list Top-5 outputs.</p><p>{rel_source('scripts/modal_openrouter_vlm.py')} · {rel_source('reports/openrouter_vlm_audit.md')}</p></details>
    </section>
    <section class="section"><h2>VLM protocol sensitivity & caveats</h2><p class="section-intro">Direct-VLM accuracy combines image evidence with candidate formulation, list order, forced choice, decoding, provider behavior, and parser success. The output score is self-reported and is not calibrated probability.</p><div class="grid"><div class="card stat span4"><div class="value">60.7%</div><div class="label">GPT prediction agreement: fixed vs per-image candidate order (same 56 successful images)</div></div><div class="card stat span4"><div class="value">32.1% vs 28.6%</div><div class="label">Fixed vs shuffled Top-1 on those same 56 images</div></div><div class="card stat span4"><div class="value">12 / 63</div><div class="label">High-reasoning pilot completions; result is inconclusive</div></div></div>
    <h3>Exact prompt template recoverable from the runner</h3><pre class="prompt">{html.escape(prompt_text)}</pre><p class="caption">The 85 candidate lines follow this text. Full fixed order: {rel_source('research_report/data/openrouter_fixed_candidate_order.csv','data/openrouter_fixed_candidate_order.csv')}. Historic outputs did not persist the full request prompt, so the template is reconstructed from the exact current runner and audit code rather than claimed as a verbatim request log.</p>
    <h3>Concrete fixed-order vs per-image-order changes</h3><div class="case-grid">{sens_cards}</div><p class="caption">All examples are real existing predictions. Candidate-list order—not the image, model ID, or output schema—changed between these two pilot protocols. Full rows include candidate positions and first ten names: {rel_source('research_report/data/vlm_order_sensitivity_examples.csv','data/vlm_order_sensitivity_examples.csv')}.</p>
    <div class="observation"><div class="callout-label">Observed</div>In the full GPT run, fixed-order candidate positions 1–20 account for 44.9% of predictions but only 23.5% of ground-truth labels. This is evidence of position association, not proof of a single causal mechanism.</div><div class="result"><div class="callout-label">Negative result retained</div>High reasoning frequently exhausted the 2,000-token completion budget before valid JSON: 51 of 63 requests failed in the better audit run. The 7/12 vs 5/12 accuracy difference among successful completions is not a fair efficacy comparison because completion success was non-random.</div></section>
    <section class="section"><h2>Metric consistency check</h2>{consistency_table}</section>
    <section class="section"><h2>Experiment Registry</h2><p class="section-intro">One row per completed run or analysis. Download: <a href="data/experiment_registry.csv">experiment_registry.csv</a>.</p>{registry_table}</section>
    <section class="section"><h2>Prompt / Candidate Protocol Registry</h2><p class="section-intro">Unknown historical fields are labeled “not persisted”; they are not silently inferred. Download: <a href="data/prompt_candidate_protocol_registry.csv">prompt_candidate_protocol_registry.csv</a>.</p>{prompt_table}</section>
    </main></div>'''
    (OUT/"benchmark_protocol.html").write_text(page("Benchmark & Protocol · BioImages","benchmark_protocol.html",benchmark_body,"Complete benchmark and experimental protocol"),encoding="utf-8")

    # Error page.
    taxrows=[]
    for _,r in tax.iterrows(): taxrows.append({"model":r.model,"candidates":int(r.candidate_species),"species":pct(r.species_top1_accuracy),"genus":pct(r.genus_top1_accuracy),"family":pct(r.family_top1_accuracy),"source":rel_source('error_analysis/taxonomy_analysis/taxonomy_level_accuracy.csv')})
    tax_table=render_table(taxrows,["model","candidates","species","genus","family","source"],["Model","Candidates","Species Top-1","Genus Top-1","Family Top-1","Source"],numeric=("candidates",),raw=("source",))
    model_summary=pd.read_csv(ROOT/"error_analysis/data/model_summary.csv")
    observation_rows=[]
    for _,r in model_summary.iterrows():
        observation_rows.append({"model":r.model,"candidates":int(r.candidate_species),"image_top1":f"{int(r.image_top1_correct)}/{int(r.image_count)} ({pct(r.image_top1_accuracy)})","image_top5":f"{int(r.image_top5_correct)}/{int(r.image_count)} ({pct(r.image_top5_accuracy)})","observation_majority":f"{int(r.observation_majority_correct)}/{int(r.observation_count)} ({pct(r.observation_majority_accuracy)})","any_correct":f"{int(r.observations_with_any_correct_image)}/{int(r.observation_count)}"})
    observation_table=render_table(observation_rows,["model","candidates","image_top1","image_top5","observation_majority","any_correct"],["Model","Candidates","Image Top-1","Image Top-5","Observation majority","Observations with any correct image"],numeric=("candidates",))
    sample= reps[reps.sample_bucket.isin(["all_four_wrong","bioclip_wrong_vlm_rescue","vlm_embedding_consensus_conflict"])].groupby("sample_bucket").head(2)
    sample_cards=''.join(f'''<article class="case"><img src="assets/images/{Path(str(r.local_sample_path)).name}" alt="{html.escape(r.image_id)}"><div class="body"><span class="tag {'warn' if r.sample_bucket=='all_four_wrong' else 'note'}">{html.escape(r.sample_bucket.replace('_',' '))}</span><div class="truth"><i>{html.escape(r.ground_truth_species)}</i></div><div class="meta">{html.escape(r.image_id)} · {html.escape(str(r.organ_category))}</div><dl><dt>BioCLIP</dt><dd class="{'ok' if r.bioclip_correct else 'bad'}">{html.escape(r.bioclip_prediction)} {'✓' if r.bioclip_correct else '✗'}</dd><dt>DINOv3</dt><dd class="{'ok' if r.dinov3_correct else 'bad'}">{html.escape(r.dinov3_prediction)} {'✓' if r.dinov3_correct else '✗'}</dd><dt>GPT</dt><dd class="{'ok' if r.gpt_5_4_correct else 'bad'}">{html.escape(r.gpt_5_4_prediction)} {'✓' if r.gpt_5_4_correct else '✗'}</dd><dt>Gemini</dt><dd class="{'ok' if r.gemini_flash_lite_correct else 'bad'}">{html.escape(r.gemini_flash_lite_prediction)} {'✓' if r.gemini_flash_lite_correct else '✗'}</dd></dl></div></article>''' for _,r in sample.iterrows())
    error_body=f'''<div class="wrap"><section class="hero"><div class="eyebrow">HTML 2 · aligned predictions</div><h1>Error Analysis & Hierarchical Evaluation</h1><p class="lede">Four baselines aligned on the same 211 strict test images and 63 observations. Facts are separated from unannotated hypotheses.</p></section><main>
    <section class="section"><h2>Protocol guardrail</h2><div class="observation"><div class="callout-label">Observed protocol difference</div>BioCLIP zero-shot and DINOv3 probe used 63 candidates. GPT and Gemini rows are a subset of their already-completed 85-candidate full run. Image IDs align exactly; candidate difficulty does not.</div><p>{rel_source('error_analysis/data/unified_predictions_image.csv')} · {rel_source('error_analysis/data/observation_predictions.csv')}</p></section>
    <section class="section"><h2>Species → genus → family</h2><div class="chart"><img src="assets/charts/hierarchy_accuracy.png" alt="Hierarchical accuracy"><p class="caption">Taxonomy is applied after prediction; original species benchmark remains unchanged.</p></div>{tax_table}</section>
    <section class="section"><h2>Image-level and observation-level scoring</h2><p class="section-intro">Observation prediction uses majority vote across its images, with mean model score to break ties. Observations contain 1–16 images.</p>{observation_table}<p>{rel_source('error_analysis/data/model_summary.csv')} · {rel_source('error_analysis/data/observation_predictions.csv')}</p></section>
    <section class="section"><h2>What kind of species error?</h2><div class="chart"><img src="assets/charts/species_error_composition.png" alt="Species error composition"></div><div class="observation"><div class="callout-label">Observation</div>Most species errors are wrong-genus errors for every model. Gemini is the exception in degree, not direction: 47 of 126 species errors retain the correct genus.</div><p>{rel_source('error_analysis/taxonomy_analysis/species_error_taxonomy_breakdown.csv')}</p></section>
    <section class="section"><h2>Agreement and disagreement</h2><div class="grid"><div class="chart span6"><img src="assets/charts/error_buckets.png" alt="Error buckets"></div><div class="card span6"><h3>Key counts</h3><ul><li>All four correct: 43 / 211</li><li>All four wrong: 17 / 211</li><li>BioCLIP alone correct: 40 / 211</li><li>BioCLIP wrong, either VLM correct: 4 / 211</li><li>GPT/Gemini consensus vs BioCLIP/DINO consensus: 10 / 211</li><li>At least one model correct: 194 / 211</li></ul><p class="caption">Buckets can overlap. Definitions and every row are preserved in CSV.</p></div></div><p>{rel_source('error_analysis/data/bucket_summary.csv')} · {rel_source('error_analysis/data/correctness_patterns.csv')}</p></section>
    <section class="section"><h2>Within-genus confusion pairs</h2><div class="chart"><img src="assets/charts/within_genus_confusions.png" alt="Within genus confusion pairs"></div><p>{rel_source('error_analysis/taxonomy_analysis/within_genus_confusions.csv')} · {rel_source('error_analysis/data/shared_confusion_pairs.csv')}</p></section>
    <section class="section"><h2>Representative aligned cases</h2><p class="section-intro">These are display samples, not automatically interpreted failure causes. Human annotation fields remain blank.</p><div class="case-grid">{sample_cards}</div><p><a href="error_gallery.html">Open the complete priority-bucket gallery →</a></p><p>{rel_source('error_analysis/annotation/human_error_annotation.csv')} · {rel_source('error_analysis/annotation/annotation_codebook.csv')}</p></section>
    <section class="section"><h2>Observed facts vs hypotheses</h2><div class="result"><div class="callout-label">Result</div>All four models fail together on 17 images from 10 species and 10 observations.</div><div class="observation"><div class="callout-label">Observation</div>Existing cluster analysis finds several BioCLIP/DINO clusters enriched for BioCLIP errors and all-four-wrong cases; their dominant recorded view is whole-tree/whole-plant.</div><div class="hypothesis"><div class="callout-label">Hypothesis—not annotated</div>View scale, missing diagnostic organs, or within-genus morphology may contribute to these errors. The current project does not yet contain human labels that establish those causes.</div></section>
    </main></div>'''
    (OUT/"error_hierarchical.html").write_text(page("Errors & Hierarchy · BioImages","error_hierarchical.html",error_body,"Aligned error analysis and hierarchical evaluation"),encoding="utf-8")

    # Representation page.
    fp=pd.read_csv(ROOT/"representation_analysis/data/frozen_probe_comparison.csv")
    fp_rows=[]
    for _,r in fp.iterrows(): fp_rows.append({"representation":r.representation,"dimension":int(r.embedding_dimension),"top1":pct(r.species_top1_accuracy),"top5":pct(r.species_top5_accuracy),"genus":pct(r.genus_top1_accuracy),"family":pct(r.family_top1_accuracy),"C":r.selected_C})
    fp_table=render_table(fp_rows,["representation","dimension","top1","top5","genus","family","C"],["Representation","Dim","Species Top-1","Species Top-5","Genus","Family","Selected C"],numeric=("dimension","C"))
    ms_rows=[]
    for _,r in ms.iterrows(): ms_rows.append({"representation":r.representation,"k":int(r.k),"species_nmi":f"{r.species_nmi_mean:.3f} ± {r.species_nmi_std:.3f}","organ_nmi":f"{r.organ_nmi_mean:.3f} ± {r.organ_nmi_std:.3f}","species_purity":f"{r.species_purity_mean:.3f} ± {r.species_purity_std:.3f}","organ_purity":f"{r.organ_purity_mean:.3f} ± {r.organ_purity_std:.3f}","stability":f"{r.stability_pairwise_ari_mean:.3f} ± {r.stability_pairwise_ari_std:.3f}"})
    ms_table=render_table(ms_rows,["representation","k","species_nmi","organ_nmi","species_purity","organ_purity","stability"],["Representation","k","Species NMI","Organ NMI","Species purity","Organ purity","Stability ARI"],numeric=("k",))
    dist_rows=[]
    for rep in ("bioclip","dinov3","efficientnet_b0"):
        s=dist[(dist.representation==rep)&(dist.comparison=="species")].iloc[0]
        o=dist[(dist.representation==rep)&(dist.comparison=="normalized_organ")].iloc[0]
        dist_rows.append({"representation":rep,"species_same":f"{s.same_mean:.3f}","species_diff":f"{s.different_mean:.3f}","species_delta":f"{s.mean_difference_same_minus_different:.3f}","species_d":f"{s.cohens_d:.3f}","organ_same":f"{o.same_mean:.3f}","organ_diff":f"{o.different_mean:.3f}","organ_delta":f"{o.mean_difference_same_minus_different:.3f}","organ_d":f"{o.cohens_d:.3f}"})
    dist_table=render_table(dist_rows,["representation","species_same","species_diff","species_delta","species_d","organ_same","organ_diff","organ_delta","organ_d"],["Representation","Species same","Species diff","Δ","d","Organ same","Organ diff","Δ","d"])
    projection_html=''.join(f'<div class="chart span6"><img src="assets/{rep}_{kind}.png" alt="{rep} {kind}"><p class="caption">{rep.replace("_"," ")} · {kind.replace("pca_umap_","").replace("-"," ")}</p></div>' for rep in ["bioclip","dinov3","efficientnet_b0"] for kind in ["pca_umap_species","pca_umap_normalized-organ","pca_umap_k20-cluster"])
    contact=''
    for rep in ("bioclip","dinov3","efficientnet_b0"):
        figs=[]
        for p in sorted((SHEETS/rep).glob("*.jpg")):
            figs.append(f'<figure class="sheet"><img src="assets/cluster_sheets/{rep}/{p.name}" alt="{rep} {p.stem}"><figcaption>{html.escape(rep)} · {html.escape(p.stem)}</figcaption></figure>')
        contact+=f'<details><summary>{rep.replace("_"," ")} · 20 centroid contact sheets</summary><div class="sheet-grid">{"".join(figs)}</div></details>'
    clusters=pd.read_csv(ROOT/"representation_analysis/data/cluster_summary.csv")
    enriched=clusters[(clusters.fdr_error_enriched==True)|(clusters.descriptive_error_enriched==True)].sort_values(["all_four_wrong_enrichment","bioclip_error_enrichment"],ascending=False).head(10)
    erows=[]
    for _,r in enriched.iterrows(): erows.append({"representation":r.representation,"k":int(r.k),"cluster":int(r.cluster_id),"test_n":int(r.size_test),"dominant_view":r.dominant_organ,"bioclip_error":f"{int(r.bioclip_error_count)} ({r.bioclip_error_enrichment:.2f}×)","all_wrong":f"{int(r.all_four_wrong_count)} ({r.all_four_wrong_enrichment:.2f}×)","flag":"FDR" if r.fdr_error_enriched else "descriptive"})
    enriched_table=render_table(erows,["representation","k","cluster","test_n","dominant_view","bioclip_error","all_wrong","flag"],["Representation","k","Cluster","Test n","Dominant raw view","BioCLIP errors","All-four-wrong","Flag"],numeric=("k","cluster","test_n"))
    rep_body=f'''<div class="wrap"><section class="hero"><div class="eyebrow">1,641 images · 63 species · frozen embeddings</div><h1>Representation & Clustering Analysis</h1></section><main>
    <section class="section"><h2>PCA / UMAP projections</h2><p class="section-intro">Visualization only: two-dimensional distance is not used as evidence about the original embedding space.</p><div class="grid">{projection_html}</div></section>
    <section class="section"><h2>Comparison first</h2><div class="grid"><div class="card span6"><span class="tag">Species identity</span><h3>BioCLIP has the stronger dataset-specific association</h3><p>It leads species NMI and purity for every k, and has the largest original-space cosine separation between same- and different-species pairs (Cohen’s d 1.874).</p></div><div class="card span6"><span class="tag note">Organ / view</span><h3>DINOv3 and EfficientNet have the stronger association</h3><p>EfficientNet leads k=10 and original-space effect size (d 1.286); DINOv3 leads organ NMI/purity at k=20 and k=50. There is no single clustering winner across k.</p></div></div><div class="hypothesis"><div class="callout-label">Scope</div>These statements describe this dataset and protocol. They do not establish that any representation “understands” taxonomy or that organ/view causes classification error.</div></section>
    <section class="section"><h2>Frozen linear-probe control</h2><div class="chart"><img src="assets/charts/strict63_accuracy.png" alt="Probe accuracy"></div>{fp_table}<p>{rel_source('representation_analysis/data/frozen_probe_comparison.csv')}</p></section>
    <section class="section"><h2>Clustering protocol</h2><dl class="protocol"><dt>Inputs</dt><dd>All 1,641 L2-normalized frozen embeddings.</dd><dt>Labels</dt><dd>Not used in K-means. Species and normalized organ/view labels are attached only afterward for NMI and purity.</dd><dt>Grid</dt><dd>k=10, 20, 50; seeds 42–51; one k-means++ initialization per seed; no “best-looking” seed selection.</dd><dt>Stability</dt><dd>Mean/std pairwise adjusted Rand index across all 45 seed pairs.</dd><dt>Organ normalization</dt><dd>12 raw values mapped to 9 normalized values; raw field retained.</dd></dl><p>{rel_source('representation_analysis/data/organ_normalization_mapping.csv')} · {rel_source('representation_analysis/data/multiseed_cluster_runs.csv')}</p></section>
    <section class="section"><h2>Multi-seed results</h2><div class="chart"><img src="assets/charts/clustering_multiseed.png" alt="Multi-seed clustering metrics"></div>{ms_table}<p>{rel_source('representation_analysis/data/multiseed_cluster_summary.csv')}</p></section>
    <section class="section"><h2>Original-space cosine evidence</h2><div class="grid"><div class="chart span6"><img src="assets/charts/distance_effects.png" alt="Distance effect sizes"></div><div class="chart span6"><img src="assets/distance_similarity_distributions.png" alt="Cosine similarity distributions"></div></div>{dist_table}<p>{rel_source('representation_analysis/data/distance_analysis_summary.csv')} · {rel_source('representation_analysis/data/distance_similarity_histograms.csv')}</p></section>
    <section class="section"><h2>Error- and view-enriched clusters</h2><p class="section-intro">Enrichment was calculated after clustering. “Dominant view” is a composition label, not a causal explanation.</p>{enriched_table}<p>{rel_source('representation_analysis/data/cluster_summary.csv')}</p></section>
    <section class="section"><h2>k=20 centroid contact sheets</h2><p class="section-intro">Each sheet shows images nearest the cluster centroid. No LLM-generated cluster names or morphology labels were added.</p>{contact}</section>
    <section class="section"><h2>Robustness conclusion</h2><div class="result"><div class="callout-label">Consistent result</div>BioCLIP leads both species clustering metrics at every k and both original-space species separation metrics.</div><div class="observation"><div class="callout-label">Mixed result</div>Organ/view association is consistently stronger in DINOv3/EfficientNet than BioCLIP, but the winner changes by k and metric.</div><div class="question"><div class="callout-label">Open question</div>The qualitative meaning of specific clusters still requires human inspection; current sheets deliberately avoid automatic morphology explanations.</div><p>{rel_source('representation_analysis/data/robustness_check.json')}</p></section>
    </main></div>'''
    (OUT/"representation_analysis.html").write_text(page("Representations · BioImages","representation_analysis.html",rep_body,"Frozen representation and clustering analysis"),encoding="utf-8")

    # Index / audit README.
    inventory=[]
    for folder in ("outputs","error_analysis","representation_analysis","reports","scripts","work","app"):
        inventory.extend(str(p.relative_to(ROOT)) for p in (ROOT/folder).rglob("*") if p.is_file())
    (DATA/"source_inventory.txt").write_text("\n".join(sorted(inventory))+"\n",encoding="utf-8")
    readme=f'''# BioImages research report bundle

Generated 2026-09-23 from existing project artifacts only. No model inference, embedding extraction, probe training, or clustering assignment was rerun.

## Recommended advisor-meeting order

1. `index.html` — compact meeting summary, headline results, hierarchy, and confusion pairs.
2. `error_gallery.html` — complete six-bucket visual review page.
3. `representation_analysis.html` — PCA/UMAP first, followed by frozen probes, clustering, distances, and contact sheets.

Secondary traceability pages:

- `benchmark_protocol.html` — exact model, split, candidate, prompt, parsing, runtime, and comparison caveats.
- `error_hierarchical.html` — detailed hierarchical tables and aligned error counts retained for reference.

## Traceability files

- `data/experiment_registry.csv` — one row per completed experiment/analysis.
- `data/prompt_candidate_protocol_registry.csv` — VLM run/protocol registry, including missing historical fields.
- `data/openrouter_fixed_candidate_order.csv` — reconstructed fixed order from the recorded seed and runner algorithm.
- `data/vlm_prompt_template.txt` — prompt template from the current benchmark runner.
- `data/vlm_order_sensitivity_examples.csv` — real fixed-vs-per-image order prediction changes.
- `data/metric_consistency_audit.csv` — explains repeated model names with different populations/protocols.
- `data/source_inventory.txt` — scanned source artifact inventory.

## Primary source families

- Full and strict predictions: `../outputs/`
- Four-model alignment, taxonomy, gallery, annotation template: `../error_analysis/`
- Embeddings, clustering, distances, projections, sheets: `../representation_analysis/`
- VLM integrity audit: `../reports/openrouter_vlm_audit.md`
- Executable protocol definitions: `../scripts/`

## Interpretation guardrails

- BioCLIP/DINO strict error rows use 63 candidates; VLM rows use 85.
- VLM confidences are self-assessed and not calibrated.
- BioCLIP cosine-softmax and probe probabilities are also method-specific; score magnitudes should not be compared across models.
- Labels were not used by K-means; they were attached after clustering.
- PCA/UMAP are visualization only; original-space cosine analysis provides the distance evidence.
- Human failure annotation fields are currently blank. The report does not assign image-level failure causes.
'''
    (OUT/"README.md").write_text(readme,encoding="utf-8")
    print(json.dumps({"pages":["index.html","benchmark_protocol.html","error_hierarchical.html","representation_analysis.html","error_gallery.html"],"experiments":len(experiments),"protocol_rows":len(protocols),"sensitivity_examples":len(sensitivity),"inventory_files":len(inventory)},indent=2))


if __name__ == "__main__":
    main()
