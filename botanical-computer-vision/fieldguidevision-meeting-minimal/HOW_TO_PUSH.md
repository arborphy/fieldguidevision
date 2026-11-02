# Push the changes yourself

Branch: `meeting-minimal-2026-10-30`, based on `lindsey-cleanup` @ 7c08b18 (6 commits).

Option A — git bundle (keeps the commits as-is):
    cd fieldguidevision
    git fetch origin lindsey-cleanup
    git fetch /path/to/meeting-minimal.bundle meeting-minimal-2026-10-30:meeting-minimal-2026-10-30
    git push -u origin meeting-minimal-2026-10-30

Option B — patches:
    git checkout -b meeting-minimal-2026-10-30 origin/lindsey-cleanup
    git am --keep-cr /path/to/patches/*.patch   # --keep-cr: EXPERIMENT_REGISTRY.csv uses CRLF
    git push -u origin meeting-minimal-2026-10-30

Option C — changed_files.zip: unzip at the repo root (paths are repo-relative), then commit.

Reports:
- report_zh_standalone.html / report_en_standalone.html: open directly, anywhere, offline (all figures and thumbnails embedded).
- In the repo, botanical-computer-vision/meeting_minimal/index.html (zh) and index_en.html (en) reference results/*.png and
  load thumbnails from bioimages.vanderbilt.edu; open them from inside that folder.
- Rebuild the standalone copies: python3 build_artifact.py index.html report_zh_standalone.html
