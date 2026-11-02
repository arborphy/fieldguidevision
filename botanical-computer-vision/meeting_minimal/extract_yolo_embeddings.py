"""Extract frozen YOLO image embeddings for the strict 1,641-image set in the same npz layout as the other models
(image_ids, embeddings [L2-normalised float32], splits, model, normalized), so every meeting_minimal script can use
them with FGV_MODEL=yolo.

  python3 extract_yolo_embeddings.py --weights yolo11s-cls.pt            # default output data/embeddings/yolo_strict/
  FGV_MODEL=yolo python3 exp4b_k_sweep.py                                  # then any experiment

Embedding = Ultralytics `model.embed()` (global-average-pooled features of the layer before the head).
If you already ran your own extraction, convert it to this layout instead (same keys, same image_id order is not
required; ids and splits are checked against the manifest by common.load()).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from ultralytics import YOLO

# Two BioImages originals are truncated at the source (baskauf/10213, baskauf/19189; re-download gives the same bytes).
# Load what is there instead of failing, and record which images needed it.
ImageFile.LOAD_TRUNCATED_IMAGES = True

BCV = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(Path.home() / ".cache/yolo/yolo11s-cls.pt"))  # downloaded on first use
ap.add_argument("--images", default="/home/user/imgcache/gq")
ap.add_argument("--out", default=str(BCV / "data/embeddings/yolo_strict/embeddings.npz"))
ap.add_argument("--imgsz", type=int, default=224)
ap.add_argument("--limit", type=int, default=0, help="first N images only (smoke test)")
a = ap.parse_args()

man = pd.read_csv(BCV / "organ_benchmark/split_manifests/individual_disjoint.csv")
if a.limit:
    man = man.head(a.limit)
Path(a.weights).parent.mkdir(parents=True, exist_ok=True)
model = YOLO(a.weights)
truncated = []
for i in man.image_id:
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        Image.open(Path(a.images) / (i.replace("/", "_") + ".jpg")).load()
    except OSError:
        truncated.append(i)
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
feats = []
for start in range(0, len(man), 32):
    paths = [str(Path(a.images) / i.replace("/", "_")) + ".jpg" for i in man.image_id[start:start + 32]]
    feats += [e.cpu().numpy().ravel() for e in model.embed(paths, imgsz=a.imgsz, verbose=False)]
E = np.stack(feats).astype(np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, image_ids=np.array(man.image_id.tolist(), dtype=str), embeddings=E, splits=np.array(man.split.tolist(), dtype=str),
         model=np.array(f"ultralytics/{Path(a.weights).name}"), normalized=np.array(True),
         truncated_source_images=np.array(truncated, dtype=str))
print(a.out, E.shape, "truncated sources:", truncated)
