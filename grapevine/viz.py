"""Matplotlib/PIL figures for the data audit, training curves and evaluation reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from .config import DATA_ROOT  # noqa: E402


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def embedding_similarity_histogram(within: np.ndarray, nearest_other: np.ndarray, source_types: pd.Series, path: Path) -> None:
    """Shows why a global-embedding threshold alone cannot find same-leaf pairs here."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0.6, 1.0, 161)
    st = source_types.to_numpy()
    for kind, color in (("jpg_photo", "tab:blue"), ("png_keras_aug", "tab:orange")):
        vals = within[(st == kind) & ~np.isnan(within)]
        ax.hist(vals, bins=bins, alpha=0.55, color=color, label=f"best match in same filename group ({kind}, n={len(vals)})")
    ax.hist(nearest_other, bins=bins, histtype="step", lw=1.6, color="black",
            label=f"best match in any other group (n={len(nearest_other)})")
    ax.set_xlabel("cosine similarity (flip-averaged ImageNet MobileNetV2 embedding)")
    ax.set_ylabel("images")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Embedding similarity overlaps: used only to shortlist candidate pairs")
    _save(fig, path)


def inlier_histogram(samples: dict[str, np.ndarray], threshold: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.concatenate([np.arange(0, 30), np.geomspace(30, 700, 25)])
    for name, vals in samples.items():
        ax.hist(np.clip(vals, 0, 699), bins=bins, histtype="step", lw=1.5, label=f"{name} (n={len(vals)})")
    ax.axvline(threshold - 0.5, color="red", ls="--", lw=1.2, label=f"link if inliers >= {threshold}")
    ax.set_xscale("symlog", linthresh=30)
    ax.set_yscale("log")
    ax.set_xlabel("SIFT/RANSAC affine inliers (best of normal / mirrored)")
    ax.set_ylabel("pairs")
    ax.legend(fontsize=8)
    ax.set_title("Geometric verification: known same-leaf pairs vs random pairs")
    _save(fig, path)


def pair_sheet(pairs: list[tuple[str, str, str]], path: Path, tile: int = 112, cols: int = 4) -> None:
    """Contact sheet of image pairs; each entry is (path_a, path_b, caption)."""
    if not pairs:
        return
    rows = (len(pairs) + cols - 1) // cols
    cap_h = 26
    sheet = Image.new("RGB", (cols * (2 * tile + 12), rows * (tile + cap_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (a, b, caption) in enumerate(pairs):
        r, c = divmod(k, cols)
        x0, y0 = c * (2 * tile + 12), r * (tile + cap_h)
        for off, p in ((0, a), (tile, b)):
            with Image.open(DATA_ROOT / p) as im:
                sheet.paste(im.convert("RGB").resize((tile, tile)), (x0 + off, y0))
        draw.text((x0 + 2, y0 + tile + 2), caption[:44], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def split_distribution(frame: pd.DataFrame, path: Path) -> None:
    counts = frame.pivot_table(index="label", columns="split", values="path", aggfunc="count").fillna(0)
    counts = counts[[c for c in ("train", "val", "test") if c in counts.columns]]
    ax = counts.plot(kind="bar", figsize=(7, 4), rot=0, color=["tab:blue", "tab:orange", "tab:green"])
    for container in ax.containers:
        ax.bar_label(container, fontsize=7)
    ax.set_ylabel("images")
    ax.set_title("Images per class and split (group-aware split)")
    _save(ax.figure, path)


def training_curves(history: pd.DataFrame, path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    x = history["epoch"]
    axes[0].plot(x, history["train_loss"], "o-", label="train")
    axes[0].plot(x, history["val_loss"], "o-", label="val")
    axes[0].set_title("loss (label-smoothed CE)")
    axes[1].plot(x, history["train_acc"], "o-", label="train")
    axes[1].plot(x, history["val_acc"], "o-", label="val")
    axes[1].set_title("accuracy")
    axes[2].plot(x, history["val_macro_f1"], "o-", color="tab:green", label="val macro-F1")
    axes[2].set_title("validation macro-F1")
    head_end = history.loc[history["phase"] == "head", "epoch"].max()
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if pd.notna(head_end):
            ax.axvline(head_end + 0.5, color="grey", ls=":", lw=1)
    fig.suptitle(f"{title}  (dotted line: end of head-only phase)", y=1.03)
    _save(fig, path)


def confusion_matrix_plot(cm: np.ndarray, class_names: list[str], path: Path, title: str) -> None:
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def comparison_bars(table: pd.DataFrame, metrics: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 3.6))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        vals = table[metric]
        bars = ax.bar(table["model"], vals, color=["tab:blue", "tab:orange", "tab:green"][: len(table)])
        ax.bar_label(bars, labels=[f"{v:.4f}" for v in vals], fontsize=8)
        lo = max(0.0, float(vals.min()) - 0.05)
        ax.set_ylim(lo, min(1.0, float(vals.max()) + 0.02) if vals.max() <= 1 else None)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
    _save(fig, path)


def gradcam_grid(items: list[tuple[Image.Image, Image.Image, str]], path: Path, cols: int = 4) -> None:
    """Grid of (original, overlay, caption) triples."""
    if not items:
        return
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 1.9, rows * 2.15))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (orig, overlay, caption) in enumerate(items):
        r, c = divmod(k, cols)
        axes[r, 2 * c].imshow(orig)
        axes[r, 2 * c + 1].imshow(overlay)
        axes[r, 2 * c].set_title(caption, fontsize=6.5, loc="left")
    fig.tight_layout()
    _save(fig, path)
