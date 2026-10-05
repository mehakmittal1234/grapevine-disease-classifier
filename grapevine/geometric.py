"""SIFT keypoint matching with RANSAC-affine verification (OpenCV, CPU, multi-process).

Kept free of PyTorch imports so that spawned worker processes start fast and stay small.
Features (normal and mirrored image) are extracted once into on-disk memory maps, so each
pair verification is pure matching.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

_STATE: dict = {}
_BLAS_VARS = ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _open_store(store: Path, mode: str):
    meta = np.load(store / "meta.npy")
    n, f = int(meta[0]), int(meta[1])
    return {
        "xy": np.lib.format.open_memmap(store / "xy.npy", mode=mode, dtype=np.float32, shape=(n, 2, f, 2)),
        "desc": np.lib.format.open_memmap(store / "desc.npy", mode=mode, dtype=np.uint8, shape=(n, 2, f, 128)),
        "count": np.lib.format.open_memmap(store / "count.npy", mode=mode, dtype=np.int16, shape=(n, 2)),
    }


def _init_extract(store: Path, n_features: int) -> None:
    cv2.setNumThreads(1)
    _STATE.update(_open_store(store, "r+"), sift=cv2.SIFT_create(nfeatures=n_features), f=n_features)


def artificial_border_mask(bgr: np.ndarray) -> np.ndarray:
    """255 where keypoints may be detected: excludes near-black fill (and a margin around it)
    left by offline rotations, whose identical corner geometry would match across leaves."""
    black = (bgr.max(axis=2) < 16).astype(np.uint8)
    black = cv2.dilate(black, np.ones((11, 11), np.uint8))
    return np.where(black > 0, 0, 255).astype(np.uint8)


def _extract_chunk(items: list[tuple[int, str]]) -> int:
    for i, path in items:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            _STATE["count"][i] = 0
            continue
        gray, mask = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), artificial_border_mask(bgr)
        for m, (img, msk) in enumerate(((gray, mask), (cv2.flip(gray, 1), cv2.flip(mask, 1)))):
            kp, desc = _STATE["sift"].detectAndCompute(img, msk)
            k = 0 if desc is None else min(len(kp), _STATE["f"])
            if k:
                _STATE["xy"][i, m, :k] = np.float32([p.pt for p in kp[:k]])
                _STATE["desc"][i, m, :k] = np.clip(np.rint(desc[:k]), 0, 255).astype(np.uint8)
            _STATE["count"][i, m] = k
    for arr in ("xy", "desc", "count"):
        _STATE[arr].flush()
    return len(items)


def extract_features(paths: list[str], store: Path, n_features: int, workers: int) -> Path:
    """Write SIFT keypoints/descriptors for every image (normal + mirrored) to ``store``.

    An existing store built from the same paths and feature budget is reused as is.
    """
    key = hashlib.sha1(("\n".join(paths) + f"\n{n_features}\nmask-v1").encode()).hexdigest()
    if (store / "key.txt").exists() and (store / "key.txt").read_text() == key:
        print(f"Reusing SIFT feature cache: {store}")
        return store
    if store.exists():
        shutil.rmtree(store)
    store.mkdir(parents=True)
    n = len(paths)
    np.save(store / "meta.npy", np.array([n, n_features]))
    np.lib.format.open_memmap(store / "xy.npy", mode="w+", dtype=np.float32, shape=(n, 2, n_features, 2))
    np.lib.format.open_memmap(store / "desc.npy", mode="w+", dtype=np.uint8, shape=(n, 2, n_features, 128))
    np.lib.format.open_memmap(store / "count.npy", mode="w+", dtype=np.int16, shape=(n, 2))
    items = list(enumerate(paths))
    chunks = [items[s : s + 128] for s in range(0, n, 128)]
    with _pool(workers, _init_extract, (store, n_features)) as ex:
        for _ in tqdm(ex.map(_extract_chunk, chunks), total=len(chunks), desc="SIFT features", unit="chunk"):
            pass
    (store / "key.txt").write_text(key)  # written last: marks the cache as complete
    return store


def _init_verify(store: Path, ratio: float, ransac_px: float) -> None:
    cv2.setNumThreads(1)
    _STATE.update(_open_store(store, "r"), ratio=ratio, ransac_px=ransac_px)


def _features(i: int, mirror: int):
    k = int(_STATE["count"][i, mirror])
    desc = np.asarray(_STATE["desc"][i, mirror, :k], dtype=np.float32)
    return np.asarray(_STATE["xy"][i, mirror, :k]), desc, (desc * desc).sum(1)


def mutual_ratio_matches(da, na, db, nb, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """Lowe-ratio-tested mutual nearest neighbours from one squared-distance matrix.

    ``na``/``nb`` are the squared norms of the descriptors. Equivalent to an OpenCV
    BFMatcher knnMatch(k=2) + reverse match, but one BLAS matmul is ~10x faster here.
    """
    d2 = np.maximum(na[:, None] + nb[None, :] - 2.0 * (da @ db.T), 0.0)
    best_b = d2.argmin(1)
    two = np.partition(d2, 1, axis=1)[:, :2]
    passes_ratio = two[:, 0] < (ratio * ratio) * two[:, 1]
    mutual = d2.argmin(0)[best_b] == np.arange(len(da))
    keep = np.nonzero(passes_ratio & mutual)[0]
    return keep, best_b[keep]


def affine_inliers(pa, da, na, pb, db, nb, ratio: float, ransac_px: float) -> int:
    """Inliers of a RANSAC affine fit on ratio-tested, mutual-nearest-neighbour SIFT matches."""
    if len(da) < 4 or len(db) < 4:
        return 0
    ia, ib = mutual_ratio_matches(da, na, db, nb, ratio)
    if len(ia) < 4:
        return 0
    M, mask = cv2.estimateAffine2D(pa[ia], pb[ib], method=cv2.RANSAC, ransacReprojThreshold=ransac_px,
                                   maxIters=2000, confidence=0.995)
    if M is None or mask is None:
        return 0
    s_max, s_min = np.linalg.svd(M[:, :2], compute_uv=False)
    # Reject degenerate fits: collapsed, exploding or strongly sheared transforms.
    if not (s_min > 0.4 and s_max < 2.5 and s_max / s_min < 1.6):
        return 0
    return int(mask.sum())


def _verify_chunk(pairs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(pairs), np.int32)
    for n, (i, j) in enumerate(pairs):
        b = _features(int(j), 0)
        best = 0
        for mirror in (0, 1):  # SIFT descriptors are not mirror-invariant, so try both
            best = max(best, affine_inliers(*_features(int(i), mirror), *b, _STATE["ratio"], _STATE["ransac_px"]))
        out[n] = best
    return out


def verify_pairs(pairs: np.ndarray, store: Path, ratio: float, ransac_px: float, workers: int,
                 desc: str = "verifying") -> np.ndarray:
    """Best (normal or mirrored) RANSAC-affine inlier count for each (i, j) pair."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return np.zeros(0, np.int32)
    chunks = [pairs[s : s + 512] for s in range(0, len(pairs), 512)]
    with _pool(workers, _init_verify, (store, ratio, ransac_px)) as ex:
        results = list(tqdm(ex.map(_verify_chunk, chunks), total=len(chunks), desc=desc, unit="chunk"))
    return np.concatenate(results)


def _workers(requested: int) -> int:
    return max(1, min(requested, os.cpu_count() or 1))


@contextmanager
def _pool(workers: int, initializer, initargs):
    """Spawn-based process pool whose workers each use a single BLAS thread."""
    saved = {v: os.environ.get(v) for v in _BLAS_VARS}
    os.environ.update({v: "1" for v in _BLAS_VARS})
    try:
        with ProcessPoolExecutor(_workers(workers), mp_context=get_context("spawn"),
                                 initializer=initializer, initargs=initargs) as ex:
            yield ex
    finally:
        for v, old in saved.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old
