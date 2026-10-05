"""Capture-bias probe: how much does the *background* alone reveal about the class?

Usage::

    python -m grapevine.bias_check

PlantVillage-style datasets are known to carry per-class capture conditions (background, light).
This probe keeps only background-like pixels in a 12-pixel frame around each image (low
saturation, not dark), summarises their colour, and fits a logistic regression on the training
split. Its accuracy on val/test (chance = 25%) measures how much class signal the background
carries, independently of any CNN. Results go to ``artifacts/data/background_bias.json``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import CLASS_NAMES, DATA_DIR, DATA_ROOT, SEED
from .dataset import NeutralizeBackground, load_splits
from .utils import save_json

FRAME = 12


def border_features(path) -> np.ndarray:
    with Image.open(path) as im:
        img = im.convert("RGB").resize((224, 224))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    border = np.zeros(rgb.shape[:2], bool)
    border[:FRAME], border[-FRAME:], border[:, :FRAME], border[:, -FRAME:] = True, True, True, True
    background = border & (hsv[..., 1] < 0.25) & (hsv[..., 2] > 0.2)
    sel = background if background.sum() >= 50 else border
    px = np.concatenate([rgb[sel], hsv[sel]], axis=1)  # (n, 6)
    return np.concatenate([px.mean(0), px.std(0), np.percentile(px, [10, 50, 90], axis=0).ravel(),
                           [background.mean() / border.mean()]])


def main() -> None:
    splits = load_splits()
    X = np.stack([border_features(DATA_ROOT / p) for p in splits["path"]])
    y = splits["label_idx"].to_numpy()
    part = splits["split"].to_numpy()
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0, random_state=SEED))
    probe.fit(X[part == "train"], y[part == "train"])
    out = {"description": __doc__.strip().splitlines()[0], "features": "colour statistics of background-like border pixels",
           "chance_accuracy": 1 / len(CLASS_NAMES)}
    for split in ("val", "test"):
        for view, mask in (("all", part == split), ("original_photos", (part == split) & splits["is_original"].to_numpy())):
            pred = probe.predict(X[mask])
            out[f"{split}_{view}"] = {
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y[mask], pred)),
                "macro_f1": float(f1_score(y[mask], pred, average="macro")),
                "confusion_matrix": confusion_matrix(y[mask], pred, labels=list(range(len(CLASS_NAMES)))).tolist(),
            }
    save_json(out, DATA_DIR / "background_bias.json")

    # Examples of the background-neutralised images used by evaluate's capture-bias check.
    test = splits[splits["split"] == "test"]
    picks = pd.concat([g.sample(3, random_state=SEED) for _, g in test.groupby("label")])
    neutral = NeutralizeBackground()
    tiles = []
    for p in picks["path"]:
        with Image.open(DATA_ROOT / p) as im:
            img = im.convert("RGB").resize((160, 160))
        tiles.append(np.concatenate([np.asarray(img), np.asarray(neutral(img))], axis=1))
    rows = [np.concatenate(tiles[k : k + 3], axis=1) for k in range(0, len(tiles), 3)]
    Image.fromarray(np.concatenate(rows, axis=0)).save(DATA_DIR / "background_neutralized_examples.jpg", quality=90)
    t = out["test_all"]
    print(f"Background-only probe: test accuracy {t['accuracy']:.4f}, macro-F1 {t['macro_f1']:.4f} "
          f"(chance {out['chance_accuracy']:.2f}); originals-only test accuracy {out['test_original_photos']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
