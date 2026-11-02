# YOLO11s BioImages embeddings

Frozen, generic COCO-pretrained YOLO11s detection model. No training or fine-tuning.

The cohort contains the same 1,641 original images, in exactly the same order as the reference BioCLIP NPZ. The `splits` array is copied unchanged. Original image bytes are used, rather than report thumbnails.

## Files

| File | Shape | Representation |
| --- | --- | --- |
| yolo11s_backbone_embeddings.npz | 1641 × 512 | Final backbone layer 10, global average pooling, L2 normalization |
| yolo11s_neck_multiscale_embeddings.npz | 1641 × 896 | Neck layers 16/19/22, global average pooling, per-layer L2 normalization, concatenation, final L2 normalization |

Start with the backbone file when comparing image representations. The neck file is a separate optional representation; do not silently pool its results with the backbone.

Each NPZ contains `image_ids`, `embeddings`, `splits`, `model`, `normalized`, `layers`, and `feature_type`. Load with `allow_pickle=False`.

```python
import numpy as np

data = np.load('yolo11s_backbone_embeddings.npz', allow_pickle=False)
X = data['embeddings']
image_ids = data['image_ids']
print(X.shape)
# For unit-normalized vectors, cosine similarity:
similarity = X @ X.T
```

## Interpretation

One row represents one whole image, not an individual detected object. Predictions, bounding boxes and COCO class probabilities are not used as embeddings. The original species and organ labels remain external evaluation labels: this model has not been trained to predict the project species or organ categories.

Compare all models on matching image IDs and consistent evaluation splits. Fit a separate PCA for each model; raw coordinates and dimensions from different encoders are not directly comparable. These files support representation analysis and do not establish improved species/organ accuracy on their own.

## Reproducibility

Input size 640 × 640 with Ultralytics square letterboxing (`rect=False`), float32 inference, batch 32, augmentation disabled. Software, GPU name, SHA-256 hashes, layer dimensions and runtime are recorded in `summary.json`. `image_manifest.csv` maps rows to image IDs and available source labels. `extract_yolo_embeddings.py` contains the extraction code.
