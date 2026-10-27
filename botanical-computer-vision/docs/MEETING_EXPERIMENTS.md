# The seven meeting experiments: what each one is for, and how to run it properly

Status: design document, 2026-10-27. The first-round runs exist only on the hosted site
(`scale/meeting/`, `nsc/`, `cluster_activation_mapping/`) and are **not in Git**; recovering
that source is task P0-2 in [PROJECT_STATUS.md](PROJECT_STATUS.md). Numbers below for the
first round are quoted from the handoff note, not recomputed.

## One question that ties them together

The seven items were raised as separate method names. They are tools for the question the
scale-conditioned analysis already opened:

> **Species mix in embedding space. How much of that is shot scale, how much is organ,
> how much is real morphological similarity — and which of it explains errors?**

What is already established (README §2–3): BioCLIP is species-first; DINOv3 and
EfficientNet are organ-first; scale moves EfficientNet images more than species does.
What is not yet established: whether that survives species/individual confounding and
human-checked labels, and what the residual mixing is. Each method below is judged by
whether it moves that question forward.

| Method | Its job in this question |
|---|---|
| 2a PCA d* | a faithful low-dim space in which the scale/organ/species axes can be measured |
| 4 soft clustering, K by numbers | clusters that are not an artefact of K=20; images "between" clusters (e.g. a leaf on a distant crown) |
| 2b enrichment | which clusters are scale / organ clusters **after controlling species** |
| 1 NSC | which few components still separate species *within* a scale |
| 5 axis reading | is PC1 "distance of the camera"? which PCs are organ? |
| 3 activation mapping | does a "distant" cluster look at the crown or at the sky? |
| 7 inversion | show what a cluster / axis point looks like (nearest real images) |
| 6 fidelity rule | none of this read off 2-D pictures |

```
frozen embeddings (1,641 strict images; later + Pound Ridge external photos)
  │
  ├─ [6] never read structure off a 2-D picture; every plot carries a fidelity score
  ├─ [2a] PCA to d* dimensions, d* chosen so neighbourhoods are preserved
  │
  ├─ [4] choose K and soft / overlapping memberships on the d*-space
  ├─ [2b] enrichment: which organ / scale / family labels each cluster over-represents
  ├─ [1] shrunken centroids: the few PCA components that define each cluster or species
  ├─ [5] axis reading: what images sit at each end of a PC (and AE unit)
  │
  ├─ [3] map cluster membership back onto image pixels (is it the leaf or the sky?)
  └─ [7] go back from a latent point to pictures (retrieval first, generation later)
```

Shared rules for every experiment below:

- **Fixed inputs.** Strict 1,641-image set, individual-disjoint split
  (`organ_benchmark/split_manifests/individual_disjoint.csv`), embeddings with recorded
  SHA-256. Anything fit on data (PCA, K-means, NSC, AE) is fit on **train only** and
  applied to val/test, unless the experiment is explicitly descriptive on all 1,641.
- **Leakage unit is the individual plant**, not the image. Permutation tests, bootstraps
  and CV folds resample individuals.
- **Labels are provisional.** Organ comes from BioImages `organ_category` (single label)
  plus Gemma multi-labels; scale is BioImages/Gemma-derived. Report which label version was
  used; validate on the 100-image Human Gold set once exported.
- **Species confound.** Clusters of 1,641 images from 465 individuals can be "about" one
  plant. Every enrichment or organ claim is repeated *within species* (stratified test)
  or the claim is downgraded.
- **Each experiment has a stop condition.** If the stated check fails, record the negative
  result in the registry and stop; do not tune until it passes.

---

## 1. Nearest shrunken centroids (NSC) × K-means

**What it is.** NSC (Tibshirani et al., 2002) computes a centroid per class, standardises
each coordinate by within-class spread, and soft-thresholds every class-vs-overall
difference by Δ. Coordinates that shrink to zero for a class stop mattering for that class.
As Δ grows you trade accuracy for a **sparse, per-class description** ("species X differs
from the average only on components 3, 7 and 19").

**Why it was raised.** K-means already gives centroids. NSC asks the follow-up: *which
dimensions actually make this centroid special?* The value is interpretability and
feature selection, not beating the linear probe.

**What was wrong with round 1.** NSC was run as a species classifier on raw 768/1024-d
coordinates and judged by accuracy (57.8 %, macro-F1 0.429; CV chose Δ = 0). Two problems:
(a) embedding coordinates are not individually meaningful — any rotation gives the same
model — so per-coordinate shrinkage is arbitrary in the raw basis; (b) a centroid classifier
on ~19 images per class will lose to logistic regression (87.2 %), so "Δ = 0 wins" was the
expected outcome and says nothing.

**Proper design.**

1. Work in the PCA basis from experiment 2a (fit on train), d* components, whitened.
   Components are ordered and comparable, so "component 7 survives" means something.
2. Three label sets: (a) species (63), (b) K-means cluster IDs from experiment 4,
   (c) species **within each shot scale** — do different components separate species at
   distance than at close-up?
3. Sweep Δ on a grid; at each Δ record CV accuracy / macro-F1 **and** the number of
   surviving components per class. Deliverable is the **accuracy–sparsity curve**, plus
   the smallest Δ within 1 SE of the best score (one-standard-error rule), not the argmax.
4. Baselines on the same split: plain nearest centroid (Δ = 0), logistic regression probe.
5. For clusters: list surviving components per cluster and pass them to experiment 5 to
   read what those axes look like.
6. Stability: the surviving set per class across the 5 CV folds (Jaccard).

**Success / stop.** Useful if a 1-SE Δ keeps ≥ 90 % of Δ=0 macro-F1 with clearly fewer
components *and* the surviving sets are stable (Jaccard ≥ 0.5). If accuracy collapses as
soon as Δ > 0, record "species information is spread across many components" — itself a
finding — and stop.

**Cost:** low (CPU, minutes). **Priority:** P2, after 2a and 4.

---

## 2. Enrichment score, with PCA used only for dimension reduction

Two separate things were bundled here.

### 2a. PCA as a lossless-enough reduction (not as a picture)

**Aim.** Pick the smallest d* such that clustering and neighbour structure in d* match the
full embedding, then run everything else in d*. This removes noise dimensions and makes
NSC/GMM feasible.

**Already done.** BioCLIP 2-D keeps 13.5 % variance and 9.6 % of 15-NN; 64-D keeps 74.5 %
and 84.8 % (all 1,641 images).

**What to add.** For d ∈ {8, 16, 32, 64, 128, 256}: explained variance, 15-NN overlap,
trustworthiness, linear-probe accuracy (species and organ) on val, and K-means ARI vs the
full-dimensional K-means. Choose d* as the smallest d with 15-NN ≥ 0.85 and probe accuracy
within 1 pp of full. Fit PCA on train, report on val.

### 2b. Enrichment

**Aim.** For each cluster (hard or soft), ask whether a label (organ, scale, family) is
over-represented relative to its base rate. Same logic as gene-set enrichment.

**What was wrong with round 1.** Fisher/BH on images treats images as independent. Images
of one individual are not, and clusters can be a single species, so "flower-enriched"
may just mean "the Cornus florida cluster".

**Proper design.**

- Report per cluster × label: count, base rate, fold enrichment, and a **CI from an
  individual-level bootstrap**.
- p-values from an **individual-level permutation** (shuffle labels among individuals'
  image sets, or within species), then BH across all cluster × label tests.
- Repeat with a **species-stratified** test (Cochran–Mantel–Haenszel across species)
  for organ and scale. Only claims that survive stratification count as "organ structure".
- For soft clusters, use membership-weighted counts.
- Sort output by effect size, not p-value; flag any cluster with < 5 individuals.

**Success / stop.** Report which organ/scale enrichments survive species stratification.
If none do, the clusters are species/individual clusters and the organ story stops there.

**Cost:** low. **Priority:** P1 (cheap, and it decides whether 3/4/5 are worth doing).

---

## 3. Activation mapping back to the image, using the cluster as the tag

**What it is.** Treat "belongs to cluster c" as a label, then ask *which pixels* push an
image towards c. CAM / Grad-CAM need a spatial feature map (ResNet / EfficientNet); for
ViTs (BioCLIP, DINOv3) use gradient × patch-token or attention-rollout, and occlusion for
any model.

**Why.** It is the direct test of "is this cluster about the leaf, or about the sky,
the hand, the ruler?"

**What was wrong with round 1.** 12 hand-picked images; 11/12 kept their nearest centroid,
9/12 occlusions dropped the score more than random. That is a demo, not evidence: the cases
were selected, there is no ground truth for where the organ is, and fp16/fp32 differences
flipped a boundary image.

**Proper design.**

1. Target score = cosine similarity of the image embedding to cluster centroid c
   (or the logit of a linear probe trained to predict cluster ID). fp32 throughout.
2. Sample ~20 images per cluster stratified by distance to the centroid (core / boundary),
   fixed seed, for all K clusters — not hand-picked.
3. Faithfulness: **deletion / insertion curves** (remove top-k % pixels by saliency vs.
   random vs. centre-crop) with AUC, aggregated per cluster.
4. Localisation: organ masks from the existing `segmentation/grounded_sam2.py` pipeline on
   the same images (spot-check 50 masks by hand). Metric: fraction of saliency mass inside
   the organ mask vs inside background. That is the number that answers "background or plant".
5. Sanity check (Adebayo et al., 2018): saliency from a randomly re-initialised last block
   must look different; otherwise the method is not reading the model.
6. Compare BioCLIP vs DINOv3 vs EfficientNet on the same images.

**Success / stop.** Report per-cluster "on-organ mass" with a CI. Stop if the sanity check
fails for a method — switch method, do not interpret the maps.

**Cost:** medium (GPU hours; masks). **Priority:** P2, run on clusters that survived 2b.

---

## 4. Soft clustering, and choosing K and dimension by numbers

**What it is.** Hard K-means gives each image one cluster. A photo with leaves *and* fruit
may legitimately belong to two. Soft methods (Gaussian / von Mises–Fisher mixtures, fuzzy
c-means) give a membership vector; overlapping methods (NEO-K-means) allow several
memberships. "Numeric methods" means choosing K (and d) from criteria instead of fixing
K = 20.

**What was wrong with round 1.** NEO-K-means at fixed K = 20 with hand-set α, β: 245
multi-assigned images, but only 19.2 % of the multi-assigned set agreed across three
seeds. α, β set the overlap budget, so the overlap is produced by the parameters, not
found in the data. K was never chosen.

**Proper design.**

1. Space: PCA d* from 2a (embeddings are L2-normalised, so also try spherical K-means /
   vMF mixture on the unit sphere).
2. Choose K over {5, 10, 15, 20, 30, 40, 60, 80} with: GMM BIC, silhouette, gap statistic,
   and **bootstrap stability** (ARI between runs on individual-level resamples). Report
   all four; choose the K where stability is high and BIC flattens. Choose d jointly from
   2a's grid.
3. Fit a GMM (diagonal covariance) at the chosen K; per image record max membership and
   **membership entropy**.
4. Test the one hypothesis that makes soft clustering worth it:
   high-entropy images are (a) multi-organ photos — check against Gemma multi-labels now,
   Human Gold later — and/or (b) the images the species classifier gets wrong — check
   against existing BioCLIP/DINOv3 test predictions. Report AUROC of entropy for each.
5. NEO-K-means only as a comparison, with α, β chosen from the data (its paper's
   heuristic) and seed stability reported.

**Success / stop.** If entropy predicts multi-organ (AUROC ≥ 0.65) or errors, soft
membership is a real signal → feed into 2b and 3. If AUROC ≈ 0.5 and stability is low,
report "no stable overlapping structure at this sample size" and stop.

**Cost:** low–medium (CPU). **Priority:** P1, together with 2.

---

## 5. Autoencoder, and "extending" the axes of PCA

**What it is.** (a) An autoencoder is a nonlinear PCA: compress the embedding to d units
and reconstruct it. (b) "Extension of axes" = walk along one direction (a PC or an AE
unit) and see what changes — which images sit at each end.

**What was wrong with round 1.** A 32-d AE vs PCA: reconstruction MSE barely better
(.000280 vs .000284), neighbour preservation worse (61.7 % vs 66.7 %). With 1,181 training
vectors a nonlinear AE has little to learn that PCA misses. This answered the AE question:
**no evidence of nonlinear structure worth the complexity.** Don't search architectures.

**Proper design (small).**

1. Keep the AE result as a negative control. One extra check only: a linear AE should
   match PCA (sanity of the training code). Stop the AE line there.
2. Axis reading, which is the useful part: for the first ~20 PCs (and any component that
   survives NSC in experiment 1) show the 12 images at each extreme and compute the
   rank correlation of the PC score with organ, scale, family and image brightness /
   background fraction. Output: one table "PC k ≈ close-up vs whole tree", "PC m ≈ bark".
3. Consistency: compare axes across BioCLIP / DINOv3 via CCA between their PCA spaces.

**Success / stop.** The deliverable is a labelled table of axes. No further AE work
unless new data (Pound Ridge) changes the picture.

**Cost:** low. **Priority:** P2 (axis reading); AE itself P3.

---

## 6. "Simple visualisation is not accurate"

This is a rule, not an experiment, and it is already demonstrated (2-D keeps ~10 % of
neighbours). Adopt it as a reporting standard:

- No conclusion is drawn from a 2-D/3-D PCA or UMAP picture alone; the claim must be
  backed by a full-space or d*-space number (distance, kNN, enrichment, probe).
- Every projection figure carries its fidelity: variance retained, 15-NN overlap,
  trustworthiness.
- Stop producing new projection pages. One fidelity table per model is enough.

**Cost:** none. **Priority:** done; enforce in review.

---

## 7. Going back from latent space to an image

**What it is.** Given a point in the embedding space (a cluster centroid, the end of a PC
axis, an interpolation between two species), show what image it corresponds to.

**What was wrong with round 1.** Pixel optimisation from random noise, 12 steps, no image
prior. Distances fell but the output was coloured noise — the expected result: many noise
images map to the same embedding. It was a negative result by construction.

**Proper design, in order of cost.**

1. **Retrieval inversion (do this).** For any latent point, return the nearest real images
   from the 1,641 set (later, plus Pound Ridge). This is honest, cheap, and answers
   "what does centroid c look like" for experiments 1, 4 and 5.
2. **Prior-constrained feature visualisation (optional).** Optimise with a strong image
   prior: jitter / frequency-penalised parameterisation (Olah et al., 2017), hundreds to
   thousands of steps, several seeds. Re-encode the result and report cosine to the target.
3. **Generative decoding (only if 1–2 motivate it).** Learn a small mapping from BioCLIP
   embedding to a CLIP space that a pretrained image-variation diffusion model (unCLIP-type)
   is conditioned on, fit on the 1,181 train images, and decode. Evaluate by re-encoding
   and by blind human judgement of "which organ is visible".

**Success / stop.** Step 1 is the deliverable. Step 2/3 only if there is a specific
question a real nearest image cannot answer.

**Cost:** 1 low; 2 medium; 3 high. **Priority:** 1 → P2; 2–3 → P3.

---

## Order of work

| Order | Item | Why first | Depends on |
|---|---|---|---|
| 1 | 2a PCA d* | every other method runs in this space | embeddings recovered (P0) |
| 2 | 4 choose K + soft membership | defines the clusters | 2a |
| 3 | 2b enrichment, species-stratified | decides which clusters are about organs/scale at all | 4 |
| 4 | 5 axis reading + 7.1 retrieval | cheap interpretation of what survived | 2a, 4 |
| 5 | 1 NSC on PCA basis | sparse description of species / clusters | 2a, 4 |
| 6 | 3 activation mapping, sampled + masked | expensive; only for clusters that survived 2b | 2b, masks |
| — | 6 fidelity rule | apply everywhere | — |
| P3 | AE search, 7.2/7.3 generation | only with a concrete question | — |

All of it should be re-run on the Pound Ridge sealed-test photos (see
[DATA_EXPANSION_PLAN.md](DATA_EXPANSION_PLAN.md)) once those exist: a structure that only
appears in BioImages is a property of that collection, not of plants.
