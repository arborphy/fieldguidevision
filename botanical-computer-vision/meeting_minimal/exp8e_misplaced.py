"""Exp 8e: photos that sit inside another organ's group on the t-SNE map (EfficientNet).
A photo is 'misplaced' when the majority organ of its 5 nearest neighbours on the 2-D t-SNE map differs from its own
organ label ('other' excluded). Same t-SNE as Figure 1.2 (full embedding, seed 42). Output: count + list for review."""
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

from common import OUT, SEED, load, save_json

meta, X = load()
Z = TSNE(2, random_state=SEED, init="pca").fit_transform(X)
nb = NearestNeighbors(n_neighbors=6).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]
org = meta.organ.values
maj = np.array([pd.Series(org[n]).value_counts().index[0] for n in nb])
ok = org != "other"
mis = ok & (maj != org)
df = meta.loc[mis, ["image_id", "species", "organ", "shot_scale"]].assign(neighbours_organ=maj[mis])
df.to_csv(OUT / "exp8_misplaced.csv", index=False)
pairs = df.groupby(["organ", "neighbours_organ"]).size().sort_values(ascending=False).head(5)
save_json("exp8_misplaced.json", dict(n=int(mis.sum()), share=float(mis.sum() / ok.sum()),
                                      top_pairs=[dict(label=a, sits_among=b, n=int(n)) for (a, b), n in pairs.items()]))
print(int(mis.sum()), round(mis.sum() / ok.sum(), 3)); print(pairs)
