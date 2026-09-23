#!/usr/bin/env python3
"""Build a self-contained static HTML gallery from existing error buckets."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
BUCKETS = [
    (
        "all_four_wrong",
        "All four models are wrong",
        "All four Top-1 predictions are wrong. Many examples are also difficult for a person to identify from the image alone.",
        "Product note: guide users toward close-up capture—for example with a translucent leaf-shaped frame—and treat distant whole-plant views as a separate mode.",
    ),
    (
        "bioclip_wrong_vlm_rescue",
        "BioCLIP wrong, VLM rescue",
        "BioCLIP is wrong, while GPT-5.4 or Gemini predicts the ground-truth species. All 4 images are shown.",
        "",
    ),
    (
        "vlm_embedding_consensus_conflict",
        "VLM vs. embedding consensus conflict",
        "BioCLIP and DINO agree with each other, GPT and Gemini agree with each other, and the two groups disagree. All 10 images are shown.",
        "",
    ),
    (
        "bioclip_only_correct",
        "BioCLIP only correct",
        "BioCLIP predicts the ground-truth species; DINOv3, GPT-5.4, and Gemini are wrong. All 40 images are shown.",
        "",
    ),
    (
        "embedding_disagreement",
        "Embedding disagreement",
        "BioCLIP and DINOv3 predict different species. All 101 images are shown; this bucket can overlap other sections.",
        "",
    ),
    (
        "all_four_correct",
        "All four models are correct",
        "All four Top-1 predictions match the ground truth. All 43 images are shown.",
        "",
    ),
]
MODEL_ROWS = [
    ("BioCLIP", "bioclip_prediction", "bioclip_correct"),
    ("DINO", "dinov3_prediction", "dinov3_correct"),
    ("GPT", "gpt_5_4_prediction", "gpt_5_4_correct"),
    ("Gemini", "gemini_flash_lite_prediction", "gemini_flash_lite_correct"),
]


def escape(value: object) -> str:
    if pd.isna(value):
        return "—"
    value = str(value).strip()
    return html.escape(value) if value else "—"


def local_name(image_id: str) -> str:
    return image_id.replace("/", "__").replace(" ", "_") + ".jpg"


def ensure_local_image(row: pd.Series) -> Path | None:
    image_dir = HERE / "gallery_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / local_name(row.image_id)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    existing_sample = HERE / "samples" / "images" / local_name(row.image_id)
    if existing_sample.exists() and existing_sample.stat().st_size > 0:
        destination.write_bytes(existing_sample.read_bytes())
        return destination

    representation_cache = Path("/tmp/bioimages-representation-cache") / local_name(row.image_id)
    if representation_cache.exists() and representation_cache.stat().st_size > 0:
        destination.write_bytes(representation_cache.read_bytes())
        return destination

    return None


def prediction_row(label: str, prediction: object, correct: bool) -> str:
    state = "correct" if bool(correct) else "wrong"
    symbol = "✓" if bool(correct) else "✗"
    accessible = "correct" if bool(correct) else "incorrect"
    return f"""
      <div class="prediction {state}">
        <span class="model">{label}</span>
        <span class="species"><i>{escape(prediction)}</i></span>
        <span class="result" aria-label="{accessible}">{symbol}</span>
      </div>"""


def card(row: pd.Series, image_src: str) -> str:
    predictions = "".join(
        prediction_row(label, row[prediction_col], row[correct_col])
        for label, prediction_col, correct_col in MODEL_ROWS
    )
    organ = escape(row.organ_detail or row.organ_category)
    subview = escape(row.subview)
    return f"""
    <article class="case-card">
      <figure class="case-image">
        <img src="{image_src}" alt="BioImages specimen {escape(row.image_id)}, ground truth {escape(row.ground_truth_species)}" loading="lazy">
      </figure>
      <div class="case-body">
        <div class="truth-label">Ground truth</div>
        <h3><i>{escape(row.ground_truth_species)}</i></h3>
        <div class="predictions" aria-label="Model predictions">
          {predictions}
        </div>
        <dl class="metadata">
          <div><dt>Observation</dt><dd>{escape(row.observation_id)}</dd></div>
          <div><dt>Image</dt><dd>{escape(row.image_id)}</dd></div>
          <div><dt>Organ</dt><dd>{organ}</dd></div>
          <div><dt>View</dt><dd>{subview}</dd></div>
        </dl>
      </div>
    </article>"""


def build() -> Path:
    frames: dict[str, pd.DataFrame] = {}
    image_sources: dict[str, str] = {}

    for key, _, _, _ in BUCKETS:
        frame = pd.read_csv(HERE / "buckets" / f"{key}.csv", keep_default_na=False)
        frames[key] = frame
        for _, row in frame.iterrows():
            if row.image_id not in image_sources:
                local_path = ensure_local_image(row)
                image_sources[row.image_id] = (
                    f"gallery_images/{local_path.name}" if local_path else escape(row.image_url)
                )

    section_html = []
    for index, (key, title, description, note) in enumerate(BUCKETS, start=1):
        frame = frames[key]
        cards = "".join(card(row, image_sources[row.image_id]) for _, row in frame.iterrows())
        section_html.append(
            f"""
    <section class="bucket" id="{key}">
      <header class="bucket-header">
        <div class="bucket-index">0{index}</div>
        <div>
          <h2>{escape(title)} <span>{len(frame)} images</span></h2>
          <p>{escape(description)}</p>
          {f'<p class="product-note">{escape(note)}</p>' if note else ''}
        </div>
      </header>
      <div class="case-grid">{cards}</div>
    </section>"""
        )

    total_cards = sum(len(frame) for frame in frames.values())
    unique_images = len(image_sources)
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BioImages Four-Baseline Error Gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17221d;
      --muted: #667069;
      --line: #d8ded9;
      --soft: #f4f6f4;
      --paper: #ffffff;
      --correct: #18734f;
      --correct-bg: #e8f5ef;
      --wrong: #aa3832;
      --wrong-bg: #fbe9e7;
      --accent: #315f4a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 15px;
      line-height: 1.45;
    }}
    .top-nav {{ background: var(--paper); color: var(--ink); border-bottom: 1px solid var(--line); padding: 14px 20px; display: flex; gap: 22px; align-items: center; flex-wrap: wrap; }}
    .top-nav strong {{ margin-right: auto; }}
    .top-nav a {{ color: var(--ink); text-decoration: none; font-size: 13px; }}
    .top-nav a.current {{ color: var(--accent); border-bottom: 2px solid var(--accent); }}
    main {{ width: min(1520px, calc(100% - 40px)); margin: 0 auto; padding: 52px 0 80px; }}
    .page-header {{ border-bottom: 1px solid var(--line); padding-bottom: 28px; }}
    .eyebrow, .truth-label, dt {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 8px 0 10px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(2rem, 4vw, 3.7rem); font-weight: 400; line-height: 1.05; }}
    .page-header p {{ max-width: 780px; margin: 0; color: var(--muted); }}
    .page-meta {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 18px; font-size: 13px; color: var(--muted); }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; align-items: center; margin-top: 14px; font-size: 13px; }}
    .legend .correct {{ color: var(--correct); }}
    .legend .wrong {{ color: var(--wrong); }}
    .bucket-links {{ display: flex; gap: 8px 16px; flex-wrap: wrap; margin-top: 16px; }}
    .bucket-links a {{ color: var(--ink); font-size: 13px; }}
    .bucket {{ padding-top: 54px; }}
    .bucket-header {{ display: grid; grid-template-columns: 44px 1fr; gap: 16px; align-items: start; margin-bottom: 20px; }}
    .bucket-index {{ color: var(--accent); font: 400 1.1rem/1 Georgia, serif; padding-top: 7px; }}
    h2 {{ margin: 0; font-family: Georgia, "Times New Roman", serif; font-size: clamp(1.5rem, 2.5vw, 2.15rem); font-weight: 400; }}
    h2 span {{ color: var(--muted); font-size: 13px; font-weight: 400; letter-spacing: 0.02em; margin-left: 8px; white-space: nowrap; }}
    .bucket-header p {{ margin: 6px 0 0; color: var(--muted); max-width: 820px; }}
    .bucket-header .product-note {{ color: var(--ink); font-weight: 700; max-width: 980px; }}
    .case-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 22px; }}
    .case-card {{ border: 1px solid var(--line); background: var(--paper); min-width: 0; }}
    .case-image {{ margin: 0; height: 330px; display: grid; place-items: center; overflow: hidden; background: var(--soft); border-bottom: 1px solid var(--line); }}
    .case-image img {{ display: block; width: 100%; height: 100%; object-fit: contain; }}
    .case-body {{ padding: 17px 18px 18px; }}
    h3 {{ margin: 3px 0 14px; font-family: Georgia, "Times New Roman", serif; font-size: 1.35rem; font-weight: 400; overflow-wrap: anywhere; }}
    .predictions {{ border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
    .prediction {{ display: grid; grid-template-columns: 68px 1fr 22px; gap: 8px; align-items: baseline; padding: 8px 0; font-size: 13px; }}
    .prediction + .prediction {{ border-top: 1px solid var(--line); }}
    .prediction .model {{ color: var(--muted); font-weight: 700; }}
    .prediction .species {{ min-width: 0; overflow-wrap: anywhere; }}
    .prediction .result {{ width: 20px; height: 20px; display: grid; place-items: center; border-radius: 50%; font-weight: 800; }}
    .prediction.correct .result {{ color: var(--correct); background: var(--correct-bg); }}
    .prediction.wrong .result {{ color: var(--wrong); background: var(--wrong-bg); }}
    .metadata {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px 18px; margin: 16px 0 0; }}
    .metadata div {{ min-width: 0; }}
    dd {{ margin: 3px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    footer {{ margin-top: 64px; padding-top: 18px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }}
    @media (max-width: 1050px) {{ .case-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
    @media (max-width: 680px) {{
      main {{ width: min(100% - 24px, 1520px); padding-top: 30px; }}
      .case-grid {{ grid-template-columns: 1fr; }}
      .case-image {{ height: min(78vw, 430px); }}
      .bucket-header {{ grid-template-columns: 34px 1fr; }}
    }}
    @media print {{
      main {{ width: 100%; padding: 0; }}
      .case-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
      .case-card {{ break-inside: avoid; }}
      .case-image {{ height: 250px; }}
      .bucket {{ break-before: page; }}
      .bucket:first-of-type {{ break-before: auto; }}
    }}
  </style>
</head>
<body>
  <nav class="top-nav"><strong>BioImages Research Report</strong><a href="../research_report/index.html">Start here</a><a class="current" href="../research_report/error_gallery.html">Full gallery</a><a href="../research_report/representation_analysis.html">Representation</a></nav>
  <main>
    <header class="page-header">
      <div class="eyebrow">BioImages · manual review</div>
      <h1>Complete error gallery</h1>
      <p>Existing predictions only. No automatic failure explanations or morphology labels.</p>
      <div class="page-meta"><span>{total_cards} bucket entries</span><span>{unique_images} unique images</span><span>BioCLIP · DINOv3 · GPT-5.4 · Gemini 2.5 Flash Lite</span></div>
      <div class="legend" aria-label="Prediction result legend"><span class="correct">✓ correct Top-1</span><span class="wrong">✗ incorrect Top-1</span></div>
      <nav class="bucket-links" aria-label="Gallery sections">{''.join(f'<a href="#{key}">{escape(title)}</a>' for key, title, _, _ in BUCKETS)}</nav>
    </header>
    {''.join(section_html)}
    <footer>Generated from the existing <code>error_analysis/buckets</code> tables. Local images are bundled once for portability; 11 uncached images use their recorded BioImages URLs. No new inference or automated failure annotation was performed.</footer>
  </main>
</body>
</html>
"""
    output = HERE / "error_gallery.html"
    output.write_text(document, encoding="utf-8")
    return output


if __name__ == "__main__":
    path = build()
    print(f"Wrote {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")
