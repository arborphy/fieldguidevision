"""Render a static HTML review gallery from an agentic-tagging run.

Combines the iteration trace (Gemma's reasoning) with the notice-batch artifact
(the grounded polygons) into one scrollable page, so you can eyeball every photo
with its grounded organ overlaid, read the agent's per-turn reasoning, and compare
the emitted tags against the DeVo ground truth — the fast loop for iterating on
the statement and prompts.

Usage::

    uv run python scripts/render_review_gallery.py \
        --run-dir scratch/p17-strict \
        --output scratch/p17-strict/review.html

Then open the HTML file in a browser (no server needed).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from io import BytesIO
from pathlib import Path


def _load_image_thumb(url: str, max_px: int = 360) -> str | None:
    """Fetch an image and return a base64 data-URI thumbnail, or None."""
    try:
        from urllib.request import Request, urlopen

        from PIL import Image

        req = Request(url, headers={"User-Agent": "ArborphyFGV/0.1"})
        with urlopen(req, timeout=30) as resp:
            img = Image.open(BytesIO(resp.read())).convert("RGB")
        img.thumbnail((max_px, max_px))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=82)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}", img.size
    except Exception:
        return None, None


def _polygon_overlay_svg(polygon, img_w: int, img_h: int, view_px: int) -> str:
    """Render the normalized polygon as an SVG overlay sized to the thumbnail."""
    if not img_w or not img_h:
        return ""
    scale = min(view_px / img_w, view_px / img_h)
    w, h = img_w * scale, img_h * scale
    pts = " ".join(f"{x * w:.1f},{y * h:.1f}" for x, y in polygon)
    return (
        f'<svg class="overlay" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}">'
        f'<polygon points="{pts}" fill="rgba(56,189,248,0.25)" '
        f'stroke="rgba(56,189,248,0.95)" stroke-width="2"/></svg>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an agentic-tagging review gallery.")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Dir containing iteration_trace.jsonl + notice_batch.json + feature_tags.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-px", type=int, default=360)
    args = parser.parse_args()

    run = args.run_dir
    trace = [json.loads(l) for l in (run / "iteration_trace.jsonl").open()]
    tags = json.loads((run / "feature_tags.json").read_text())["tags"]
    batch = json.loads((run / "notice_batch.json").read_text())

    # Polygons by observation_id from the artifact. Normalize the "inat-" prefix
    # (the finalizer prefixes the artifact; the trace keeps raw ids).
    def _raw(oid: str) -> str:
        return oid.removeprefix("inat-")

    polys: dict[str, list] = {}
    for obs in batch.get("observations", []):
        for ann in obs.get("annotations", []):
            geom = ann.get("geometry", {})
            if geom.get("type") == "Polygon":
                polys.setdefault(_raw(obs["observation_id"]), []).append(geom["coordinates"][0])
    tags_by_obs: dict[str, list] = {}
    for t in tags:
        tags_by_obs.setdefault(_raw(t["observation_id"]), []).append(t)

    statement = html.escape(json.loads((run / "feature_tags.json").read_text()).get("statement", ""))

    cards = []
    n_imgs = 0
    for r in trace:
        oid = r["observation_id"]
        species = html.escape(r.get("species_name") or "")
        outcome = r["outcome"]
        thumb, dims = _load_image_thumb(r["image_url"], args.max_px)
        overlay = ""
        if thumb and dims and _raw(oid) in polys:
            overlay = _polygon_overlay_svg(polys[_raw(oid)][0], dims[0], dims[1], args.max_px)
            n_imgs += 1
        img_html = (
            f'<div class="imgwrap"><img src="{thumb}" alt="{species}">{overlay}</div>'
            if thumb else '<div class="imgwrap noimg">image unavailable</div>'
        )
        turn_rows = "".join(
            f'<tr><td class="turn">{t["turn"]}</td><td class="phase">{html.escape(t["phase"])}</td>'
            f'<td class="rec">{html.escape(str(t["rec_phrase"] or "—"))}</td>'
            f'<td>{html.escape(", ".join(t["grounded_organs"]) or "—")}</td>'
            f'<td class="rat">{html.escape(t["gemma_rationale"])}</td></tr>'
            for t in r["turns"]
        )
        # tags file may carry either prefixed or raw ids; normalize to raw.
        obs_tags = tags_by_obs.get(_raw(oid), [])
        tag_html = "".join(
            f'<span class="tag">{html.escape(t["value_label"])} '
            f'<em>{t["confidence"]:.2f}</em></span>'
            for t in obs_tags
        ) or '<span class="none">no tags (abstained)</span>'
        cards.append(f"""
        <section class="card outcome-{outcome}">
          <header>
            <span class="species">{species}</span>
            <span class="badge">{outcome} · {r["turns_used"]} turns</span>
          </header>
          <div class="body">
            {img_html}
            <div class="reason">
              <table>{turn_rows}</table>
              <div class="final"><strong>Final:</strong> {html.escape(r["rationale"])}</div>
              <div class="tags">{tag_html}</div>
            </div>
          </div>
        </section>""")

    outcomes = {}
    for r in trace:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    summary = " · ".join(f"{k} {v}" for k, v in sorted(outcomes.items()))

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>P17 review — {html.escape(run.name)}</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 0; background: #0f1410; color: #e7e5df; }}
  header.top {{ position: sticky; top: 0; background: #1a231c; padding: 14px 20px; border-bottom: 1px solid #2c3a2e; z-index: 10; }}
  header.top h1 {{ margin: 0 0 4px; font-size: 18px; }}
  header.top .meta {{ color: #9db3a2; font-size: 13px; }}
  main {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
  .card {{ background: #1a231c; border: 1px solid #2c3a2e; border-radius: 12px; margin-bottom: 18px; overflow: hidden; }}
  .card > header {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #202b22; }}
  .species {{ font-style: italic; font-weight: 600; }}
  .badge {{ font-size: 12px; padding: 2px 10px; border-radius: 999px; background: #2c3a2e; }}
  .outcome-resolved .badge {{ background: #2f5d3a; }}
  .outcome-abstain .badge {{ background: #6b5a2f; }}
  .outcome-discarded .badge {{ background: #5d2f2f; }}
  .body {{ display: flex; gap: 16px; padding: 14px; }}
  .imgwrap {{ position: relative; flex: 0 0 auto; }}
  .imgwrap img {{ display: block; border-radius: 8px; max-width: 360px; }}
  .imgwrap .overlay {{ position: absolute; inset: 0; }}
  .noimg {{ color: #9db3a2; padding: 40px 20px; }}
  .reason {{ flex: 1; min-width: 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  td {{ padding: 4px 6px; border-top: 1px solid #243024; vertical-align: top; }}
  td.turn, td.phase {{ white-space: nowrap; color: #9db3a2; }}
  td.rec {{ font-family: ui-monospace, monospace; color: #a5d6a7; }}
  td.rat {{ color: #cfcabf; }}
  .final {{ margin-top: 10px; font-size: 13px; }}
  .tags {{ margin-top: 8px; }}
  .tag {{ display: inline-block; background: #2f5d3a; padding: 3px 10px; border-radius: 999px; font-size: 12px; margin-right: 6px; }}
  .tag em {{ opacity: 0.7; font-style: normal; }}
  .none {{ color: #b59a5f; font-size: 12px; }}
</style></head><body>
<header class="top">
  <h1>Agentic tagging review — {html.escape(run.name)}</h1>
  <div class="meta">{html.escape(statement)}<br>{len(trace)} photos · {summary}</div>
</header>
<main>{"".join(cards)}</main>
</body></html>"""

    args.output.write_text(page)
    print(f"Wrote {args.output} ({len(trace)} photos, {n_imgs} with polygon overlays).")
    print(f"Open it:  open {args.output}")


if __name__ == "__main__":
    main()
