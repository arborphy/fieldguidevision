# YOLO organ detection / segmentation comparison

Status: experiment plan only; no YOLO accuracy or embedding result has been measured here. The repository snapshot has image-level organ labels and an unfilled multi-label audit template, but no supplied YOLO checkpoint or usable organ bounding-box / mask ground truth.

## What should change

A detector/segmenter localizes one or more organ instances. It does not replace an image embedding model automatically. Separate the localization question from the representation question:

1. Keep the original individual-plant-disjoint train/validation/test manifest. Every object, crop and augmentation from an image inherits its original split. Also audit duplicate images and species-disjoint transfer where appropriate.
2. Define the organ ontology and annotation policy before training: leaf, twig, bark, flower/inflorescence, fruit, cone, whole plant; handle seeds, occlusion, small objects, crowded branches, overlapping organs and unlabeled regions explicitly. A single image-level canonical organ label cannot generate ground-truth boxes or masks. Manually review a stratified subset containing mixed-organ and difficult views.
3. Freeze a named YOLO architecture, task, input size, pretrained checkpoint and hash; fine-tune only on train. Select thresholds and training checkpoints on validation. Standard COCO weights are an initialization, not a validated plant-organ recognizer.
4. First report per-organ detection precision/recall, box mAP50 and mAP50–95, small-object recall, missed-image rate, false positives, and speed at a named device/input size. For instance segmentation also report mask mAP50–95 and matched-instance IoU/Dice with an explicit matching rule. Report both conditional matched-mask quality and end-to-end recall; high IoU on matched objects can hide missed organs.
5. Run a paired downstream ablation on the same images with the same frozen BioCLIP/DINOv3/EfficientNet encoder: (A) whole image, (B) predicted organ-box crops, (C) predicted organ masks with a fixed background fill. Add ground-truth boxes/masks as an upper-bound control on annotated images. This distinguishes localization errors from feature quality and from merely changing encoder architecture.
6. Specify how multiple instances combine into an image-level prediction (fixed mean/attention pooling or a fixed per-organ aggregation), and retain a fallback for images with no detections. Count missed detections in the downstream denominator; do not remove difficult images. Compare crop-only results only as a separate conditional analysis.
7. Then repeat the relevant existing analyses: organ/species probes and absolute macro-F1; PCA retention; K sensitivity at 5/10/15/20/30/50 with geometry, seed stability and chance-adjusted alignment; train-only soft-clustering diagnostics. Fit every transformation on train. Compare embeddings at a fixed extraction layer and pooling rule if YOLO's own backbone is also evaluated. Its feature maps and object confidences are not directly interchangeable with an image-level embedding or GMM membership.
8. Assess uncertainty against actual validation errors, with individuals resampled as groups. Evaluate mixed-organ presence only where independent multi-label/instance annotations exist. A YOLO detection confidence and a GMM component posterior answer different questions; neither is evidence of simultaneous organ presence by itself.

## Minimum inputs needed for an actual run

- Exact YOLO version/model and checkpoint (or authorization to train a specified model), plus its class list.
- Accessible original images with their stable image IDs and split mapping.
- Organ bounding boxes or instance masks and an annotation policy; distinguish real human ground truth from proposed/pseudo-labels.
- Available training/inference hardware and a fixed runtime budget.

Start with a validation pilot that covers organs, species, individual plants, shooting scales and mixed-organ cases. Choose sample size and training budget after inspecting annotation support; do not claim statistical equivalence from an arbitrary small subset.

Official task and metric references:
- https://docs.ultralytics.com/tasks/detect/
- https://docs.ultralytics.com/tasks/segment/
