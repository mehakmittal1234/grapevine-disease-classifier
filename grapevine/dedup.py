"""Content-based near-duplicate detection, used to merge source groups before splitting.

Filename groups already tie each augmented file to its source leaf. They cannot reveal
(a) a Keras PNG group whose source leaf is also one of the JPG photos, or (b) the same leaf
photographed twice and stored under two different UUIDs. Both occur in this dataset and
both would put one physical leaf on both sides of a split.

A global CNN embedding alone cannot separate "same leaf" from "similar-looking leaf" here
(the similarity distributions overlap heavily), so detection is two-stage:

1. **Shortlist** - every image is embedded with a frozen ImageNet MobileNetV2 (mean of the
   image and its mirror); its ``k`` most similar images from *other* filename groups become
   candidate pairs.
2. **Verify** - each candidate pair is matched with SIFT keypoints (Lowe ratio test plus
   mutual nearest neighbours) and an affine transform is fitted with RANSAC; the mirrored
   image is tried too. Augmentations and re-shoots of one leaf yield many geometrically
   consistent inliers, different leaves almost none. Degenerate fits (collapsed or strongly
   anisotropic transforms) are rejected.

Pairs with at least ``min_inliers`` inliers are linked, and linked groups are merged with
union-find. Neither stage uses labels, so this step cannot leak label information. The
verification itself lives in :mod:`grapevine.geometric`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- shortlist


def nearest_other_groups(
    emb: np.ndarray, groups: list[str], k: int, chunk: int = 1024, query: np.ndarray | None = None,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` most similar images from a different filename group, for each query image.

    ``query`` restricts which rows are searched for (default: all); ``allowed`` restricts which
    images may be returned as neighbours (default: all). Returns (indices, similarities).
    """
    gid = pd.factorize(pd.Series(groups))[0]
    rows_all = np.arange(len(emb)) if query is None else np.asarray(query)
    out_idx = np.empty((len(rows_all), k), np.int64)
    out_sim = np.empty((len(rows_all), k), np.float32)
    for s in range(0, len(rows_all), chunk):
        rows = rows_all[s : s + chunk]
        sims = emb[rows] @ emb.T
        sims[gid[rows][:, None] == gid[None, :]] = -np.inf  # also removes self-matches
        if allowed is not None:
            sims[:, ~allowed] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, 1)
        order = np.argsort(-top_sims, axis=1)
        out_idx[s : s + chunk] = np.take_along_axis(top, order, 1)
        out_sim[s : s + chunk] = np.take_along_axis(top_sims, order, 1)
    return out_idx, out_sim


def within_group_similarity(emb: np.ndarray, groups: list[str], chunk: int = 1024) -> np.ndarray:
    """Best similarity of each image to another member of its own filename group (NaN if alone)."""
    gid = pd.factorize(pd.Series(groups))[0]
    best = np.full(len(emb), np.nan, np.float32)
    for s in range(0, len(emb), chunk):
        sims = emb[s : s + chunk] @ emb.T
        r = np.arange(sims.shape[0])
        sims[r, s + r] = -np.inf
        sims[gid[s : s + chunk][:, None] != gid[None, :]] = -np.inf
        m = sims.max(1)
        best[s : s + chunk] = np.where(np.isfinite(m), m, np.nan)
    return best


def sequence_pairs(frame: pd.DataFrame, window: int) -> np.ndarray:
    """Pairs of JPG photos (different filename groups) with the same class and photographer
    tag whose photo numbers differ by at most ``window`` - typical re-shoots of one leaf."""
    jpg = frame[frame["photo_num"] >= 0]
    out = []
    for _, part in jpg.groupby(["label", "photo_tag"]):
        part = part.sort_values("photo_num")
        idx, num, grp = part.index.to_numpy(), part["photo_num"].to_numpy(), part["file_group"].to_numpy()
        for a in range(len(part)):
            b = a + 1
            while b < len(part) and num[b] - num[a] <= window:
                if grp[a] != grp[b]:
                    out.append((min(idx[a], idx[b]), max(idx[a], idx[b])))
                b += 1
    return np.unique(np.array(out, dtype=np.int64).reshape(-1, 2), axis=0)


def is_sequence_pair(frame: pd.DataFrame, pairs: np.ndarray, window: int) -> np.ndarray:
    num, tag, lab = frame["photo_num"].to_numpy(), frame["photo_tag"].to_numpy(), frame["label"].to_numpy()
    i, j = pairs[:, 0], pairs[:, 1]
    return (num[i] >= 0) & (num[j] >= 0) & (tag[i] == tag[j]) & (lab[i] == lab[j]) & (np.abs(num[i] - num[j]) <= window)


def link_mask(frame: pd.DataFrame, pairs: np.ndarray, inliers: np.ndarray, cfg) -> np.ndarray:
    """Same-leaf decision: same class and enough inliers (fewer for consecutive photos).

    Cross-class matches are never merged - a leaf cannot carry two labels - and are reported
    separately for review instead.
    """
    if len(pairs) == 0:
        return np.zeros(0, bool)
    lab = frame["label"].to_numpy()
    same_class = lab[pairs[:, 0]] == lab[pairs[:, 1]]
    seq = is_sequence_pair(frame, pairs, cfg.sequence_window)
    return same_class & ((inliers >= cfg.min_inliers) | (seq & (inliers >= cfg.min_inliers_sequence)))


def candidate_pairs(nn_idx: np.ndarray, query: np.ndarray | None = None) -> np.ndarray:
    """Unique unordered (i, j) pairs from a neighbour table, sorted for feature reuse."""
    rows = np.arange(len(nn_idx)) if query is None else np.asarray(query)
    i = np.repeat(rows, nn_idx.shape[1])
    j = nn_idx.ravel()
    pairs = np.unique(np.sort(np.stack([i, j], 1), axis=1), axis=0)
    return pairs[pairs[:, 0] != pairs[:, 1]]


# --------------------------------------------------------------------------- merging


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic representative: the lexicographically smaller root.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def merge_groups(groups: list[str], links: np.ndarray) -> list[str]:
    uf = UnionFind(set(groups))
    for i, j in links:
        uf.union(groups[i], groups[j])
    return [uf.find(g) for g in groups]
