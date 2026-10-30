"""Make a self-contained copy of a report: figures and thumbnails inlined as data URIs, so it opens anywhere
(also offline) and can be published online (the hosted viewer blocks external images).

Usage: python3 build_artifact.py [--artifact] [IN.html] OUT.html
  IN defaults to index.html. --artifact strips the document skeleton for the hosted viewer and adds its theme hooks.
"""
import base64
import re
import sys
from pathlib import Path

from common import IMG_DIR, load

H = Path(__file__).resolve().parent
args = [a for a in sys.argv[1:] if a != "--artifact"]
artifact = "--artifact" in sys.argv
src, dst = (H / args[0], Path(args[1])) if len(args) == 2 else (H / "index.html", Path(args[0]))
meta, _ = load()
thumb = dict(zip(meta.thumbnail_url, meta.thumb))
html = src.read_text()


def b64(p, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode()


html = re.sub(r"src='(results/[^']+\.png)'", lambda m: f"src='{b64(H / m.group(1), 'image/png')}'", html)
html = re.sub(r"<img loading=lazy src='(https://bioimages[^']+)'",
              lambda m: f"<img src='{b64(IMG_DIR / 'tn' / thumb[m.group(1)], 'image/jpeg')}'", html)
assert "src='results/" not in html and "src='https://bioimages" not in html
if artifact:
    html = re.sub(r'<!doctype html><html lang=\w+><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">',
                  '<meta name="claude:viewport-fit-sides" content="cover">', html)
    html = html.replace("</style></head><body>", "</style>").replace("</main></body></html>", "</main>")
    dark = "--bg:#151714;--fg:#e8e9e4;--mut:#a3a89f;--line:#33362f;--card:#1d201b;--v:#6fcf97;--e:#f2c94c;--x:#f28b82;--r:#8ab4f8"
    html = html.replace(f"@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{{dark}}}}}",
                        f"@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{{dark};color-scheme:dark}}}}"
                        f":root[data-theme=dark]{{{dark};color-scheme:dark}}")
    html = html.replace("margin:0;padding:0 16px}",
                        "margin:0;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}")
    html += "<style>pre{overflow-x:auto}</style>"
dst.write_text(html)
print(dst, f"{len(html) / 1e6:.1f} MB")
