# Report style guide (biweekly group-meeting update)

Applies to `build_boss_report.py` (boss_report_zh.html / boss_report_en.html). Every new or revised section follows it.

## Audience and scope
- Biweekly group meeting, presented live. Report only the work of this period; no long background or history.
- Every request from the last review gets a visible answer, but the report never quotes or addresses the reviewer ("your request", "you suggested").
- Organ is the main target; shot scale and species are secondary.

## Tone
- Academic, declarative. No questions in headings, aims or text.
- Plain words first; technical terms get a one-sentence explanation without formulas
  (e.g. how η², information kept or a clustering metric is computed).
- State what each method is for and whether it is unsupervised or supervised (supervised methods only explain / evaluate).
- Report failures honestly in one sentence (e.g. the withdrawn GMM soft clustering) - it shows rigour.

## Page structure
1. Title + one line of setup (data, split by individual plant, models, labels used only for evaluation).
2. **Key findings**: one sentence per section, each linking to its section.
3. Sections 1…n, then **Next steps** (proposed decisions + follow-up), then **Appendix: methods at a glance**.

## Section structure (same order every time)
1. Small topic label (`n · topic`).
2. **Heading = the conclusion as one full sentence** with the key number (assertion-evidence).
3. One-line aim (declarative).
4. One-line **line of argument** when the section has several panels (e.g. loss → cause → consistency → choice → display).
   Panels are ordered by this argument, not by figure type.
5. Numbered panels `n.1, n.2 …`; each panel = label, figure or table, and **one-sentence finding** underneath.
   Definitions sit in the first panel that uses them; later panels refer back.
6. **Interpretation** (2–3 sentences, decision-oriented).
7. **Method / Procedure** in small grey text at the end.

## Figures and tables
- Every table gets a small companion chart beside it (heatmap, dumbbell, grouped bars); the table stays the main element,
  the chart is at most ~300 px wide. Stack on narrow screens.
- One-glance figures: few colours, consistent colours across the report (organ #dd8452, shot scale #4c72b0,
  species #55a868, neighbours grey), labels at line ends instead of legends, no overlapping text.
- Numbers as percentages where they are shares, right-aligned; translate column headers in the Chinese version.
- Example photos: always say what the photo's own label is and what each candidate group is (group number + main organ and
  its share). Never show bare IDs like "c6 / c3".
- 2-D / 3-D plots are for display only; say so. Use t-SNE for display and state its concrete use
  (finding photos that sit in another organ's group). Conclusions come from 64-D computations.

## Visual style
- Minimal: white background, dark grey text, one accent colour (dark blue), no coloured cards or status badges.
- Full-width results; figures capped at ~820 px; consistent spacing.
- Chinese and English versions are generated from the same code and numbers.

## Working rhythm
- One section at a time: run its experiment, update the report, send both versions, wait for review.
- Few, well-chosen numbers; every number read from `results/`, never typed by hand. Check numbers in text against the data.
- Keep token use low: no long waits in the foreground, no duplicate runs.
