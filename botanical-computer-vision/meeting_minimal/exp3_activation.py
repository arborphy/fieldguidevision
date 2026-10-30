"""Exp 3: Grad-CAM + occlusion control on real images, EfficientNet-B0 (the only model whose raw
vectors AND weights are available here). Target score = cosine(image embedding, K=20 cluster centroid),
fp32 on CPU. BioCLIP / DINOv3: not run (no raw vectors in Git; recorded as blocked).

Steps
 0. Re-embed 40 random images and compare with the stored vectors (checks preprocessing/model).
 1. Sample: for every K=20 cluster, 2 core + 2 boundary images (by distance rank), seed 42 -> 80 images.
 2. Grad-CAM on model.features[-1] w.r.t. target score.
 3. Occlusion control: grey out the top-25% Grad-CAM cells (7x7 grid) vs 10 random sets of equal size;
    record score drop. Faithful if top-CAM drop > mean random drop.
 4. Sanity (Adebayo): re-initialise the last feature block, recompute CAM, Spearman vs original.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from common import IMG_DIR, OUT, SEED, load, save_json, input_record

torch.manual_seed(SEED)
torch.set_num_threads(4)
meta, X = load()
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights).eval()
for p in model.parameters():
    p.requires_grad_(False)
tf = weights.transforms()
rng = np.random.default_rng(SEED)


def img_tensor(i):
    with Image.open(IMG_DIR / "gq" / meta.thumb[i]) as im:
        return tf(im.convert("RGB")).unsqueeze(0)


def embed(x, m=model):
    return F.normalize(torch.flatten(m.avgpool(m.features(x)), 1), dim=-1)


# 0. reproduction check
chk = rng.choice(len(meta), 40, replace=False)
with torch.no_grad():
    E = torch.cat([embed(img_tensor(i)) for i in chk]).numpy()
repro = (E * X[chk]).sum(1)
print("re-embed cosine vs stored: median %.4f min %.4f" % (np.median(repro), repro.min()))

cent = np.stack([X[meta.eff_k20 == c].mean(0) for c in range(20)])
cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
cent_t = torch.tensor(cent)


def cam_and_score(x, c, m=model):
    feats = {}
    h = m.features[-1].register_forward_hook(lambda mod, i, o: feats.__setitem__("a", o))
    x = x.clone().requires_grad_(True)
    e = embed(x, m)
    s = (e * cent_t[c]).sum()
    a = feats["a"]
    g, = torch.autograd.grad(s, a)
    h.remove()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1))[0].detach().numpy()  # 7x7
    return cam, float(s)


MEAN = torch.zeros(1, 3, 1, 1)  # normalised-space zero == ImageNet mean grey


def occlude(x, cells):
    y = x.clone()
    for r, c in cells:
        y[:, :, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = MEAN
    return y


def score(x, c):
    with torch.no_grad():
        return float((embed(x) * cent_t[c]).sum())


# 1. sample
dist = 1 - (X * cent[meta.eff_k20.values]).sum(1)
rows = []
for c in range(20):
    idx = np.where(meta.eff_k20 == c)[0]
    order = idx[np.argsort(dist[idx])]
    q = max(1, len(order) // 4)
    core = rng.choice(order[:q], min(2, q), replace=False)
    bnd = rng.choice(order[-q:], min(2, q), replace=False)
    rows += [(i, c, "core") for i in core] + [(i, c, "boundary") for i in bnd]

K = 12  # top 25% of 49 cells
res = []
cams = {}
for i, c, kind in rows:
    x = img_tensor(i)
    cam, s0 = cam_and_score(x, c)
    flat = np.argsort(cam.ravel())[::-1]
    top = [divmod(int(j), 7) for j in flat[:K]]
    d_top = s0 - score(occlude(x, top), c)
    d_rand = [s0 - score(occlude(x, [divmod(int(j), 7) for j in rng.choice(49, K, replace=False)]), c) for _ in range(10)]
    own = int(np.argmax(cent @ X[i]))
    res.append(dict(image_id=meta.image_id[i], cluster=c, kind=kind, organ=meta.organ[i], scale=meta.shot_scale[i],
                    species=meta.species[i], score=s0, nearest_centroid_is_cluster=own == c,
                    drop_top_cam=d_top, drop_random_mean=float(np.mean(d_rand)), drop_random_sd=float(np.std(d_rand)),
                    top_beats_random=d_top > np.mean(d_rand),
                    cam_center_mass=float(cam[2:5, 2:5].sum() / (cam.sum() + 1e-12))))
    cams[meta.image_id[i]] = cam
df = pd.DataFrame(res)
df.to_csv(OUT / "exp3_activation_cases.csv", index=False)

# 4. sanity: randomise last block
import copy
rmodel = copy.deepcopy(model)
for mod in rmodel.features[-1].modules():
    if hasattr(mod, "reset_parameters"):
        mod.reset_parameters()
san = []
for i, c, kind in rows[:24]:
    x = img_tensor(i)
    cr, _ = cam_and_score(x, c, rmodel)
    san.append(spearmanr(cams[meta.image_id[i]].ravel(), cr.ravel())[0])
san = np.nan_to_num(np.array(san))

summary = dict(inputs=input_record(), model=str(weights), n_cases=len(df),
               reembed_cosine_median=float(np.median(repro)), reembed_cosine_min=float(repro.min()),
               frac_top_cam_beats_random=float(df.top_beats_random.mean()),
               frac_by_kind=df.groupby("kind").top_beats_random.mean().to_dict(),
               median_drop_top=float(df.drop_top_cam.median()), median_drop_random=float(df.drop_random_mean.median()),
               sanity_spearman_vs_randomised_last_block_median=float(np.median(san)),
               mean_cam_center_mass_3x3=float(df.cam_center_mass.mean()), center_mass_uniform_expectation=9 / 49,
               note="Sampled 2 core + 2 boundary per cluster, seed 42, not hand-picked. No organ masks: "
                    "says whether the map is faithful to the score, not whether it is on the organ.")
save_json("exp3_activation_summary.json", summary)
np.savez_compressed(OUT / "exp3_cams.npz", ids=np.array(list(cams)), cams=np.stack(list(cams.values())))
print(summary)

# figure: 8 showcase cases (first core + first boundary of 4 clusters with most images)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
big = meta.eff_k20.value_counts().index[:4]
show = pd.concat([df[(df.cluster == c)].groupby("kind").head(1) for c in big])
fig, ax = plt.subplots(2, len(show), figsize=(2.3 * len(show), 5))
for j, r in enumerate(show.itertuples()):
    i = int(np.where(meta.image_id == r.image_id)[0][0])
    im = Image.open(IMG_DIR / "gq" / meta.thumb[i]).convert("RGB").resize((224, 224))
    ax[0, j].imshow(im); ax[0, j].set_title(f"c{r.cluster} {r.kind}\n{r.organ}, {r.scale}", fontsize=8)
    ax[1, j].imshow(im); ax[1, j].imshow(np.kron(cams[r.image_id], np.ones((32, 32))), alpha=.5, cmap="jet")
    ax[1, j].set_title(f"drop top {r.drop_top_cam:.3f}\nrandom {r.drop_random_mean:.3f}", fontsize=8)
for a in ax.ravel():
    a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "exp3_gradcam_cases.png", dpi=110)
