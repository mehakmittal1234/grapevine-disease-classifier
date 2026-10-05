"""Audit the dataset, merge same-leaf source groups and write a leakage-free split.

Usage::

    python -m grapevine.prepare_data            # reuses cached embeddings / SIFT features when valid
    python -m grapevine.prepare_data --recompute-embeddings

Steps: manifest + MD5 dedup -> candidate pairs (embedding shortlist + consecutive photo
numbers) -> SIFT/RANSAC verification -> union-find merge into source groups -> stratified
group split -> independent deeper leakage audit (re-split if it finds a leak) -> hard
leakage checks. Outputs go to ``artifacts/data``; the script exits non-zero, and writes no
``splits.csv``, if any check fails.
"""

from __future__ import annotations

import argparse
import shutil
import time
from functools import partial

import numpy as np
import pandas as pd

# Keep top-level imports free of PyTorch: spawned verification workers re-import this module.
from .config import (
    ARTIFACTS_DIR,
    CLASS_NAMES,
    DATA_DIR,
    DATA_REPORT_JSON,
    DATA_ROOT,
    LARGE_GROUP_MIN_FILES,
    MANIFEST_CSV,
    SEED,
    SPLIT_FRACTIONS,
    SPLITS_CSV,
    DedupConfig,
)
from .dedup import (
    candidate_pairs,
    is_sequence_pair,
    link_mask,
    merge_groups,
    nearest_other_groups,
    sequence_pairs,
    within_group_similarity,
)
from .geometric import extract_features, verify_pairs
from .manifest import build_manifest, summarize_manifest
from .split import assign_splits, group_table, leakage_checks, split_summary

EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
SIFT_STORE = ARTIFACTS_DIR / "cache" / "sift"
MAX_AUDIT_ROUNDS = 4
SPLIT_COLUMNS = [
    "path", "label", "label_idx", "split", "source_group", "file_group", "source_type",
    "aug", "is_original", "photo_tag", "photo_num", "max_inliers_to_train", "width", "height", "md5",
]
SHORT_TYPE = {"jpg_photo": "jpg", "png_keras_aug": "png", "unknown": "other"}


def load_or_compute_embeddings(paths: list[str], cfg: DedupConfig, device_name: str | None, recompute: bool) -> np.ndarray:
    if EMBEDDINGS_NPZ.exists() and not recompute:
        cached = np.load(EMBEDDINGS_NPZ, allow_pickle=False)
        if cached["paths"].tolist() == paths:
            print(f"Using cached embeddings: {EMBEDDINGS_NPZ}")
            return cached["emb"]
        print("Cached embeddings do not match the manifest; recomputing.")
    from .embeddings import compute_embeddings
    from .utils import get_device, set_seed

    set_seed(SEED)
    t0 = time.time()
    emb = compute_embeddings([DATA_ROOT / p for p in paths], get_device(device_name), cfg.embed_batch_size, cfg.num_workers)
    print(f"Embedded {len(paths)} images in {time.time() - t0:.0f}s")
    np.savez_compressed(EMBEDDINGS_NPZ, paths=np.array(paths), emb=emb)
    return emb


def _stats(values) -> dict:
    s = pd.Series(values, dtype=float).dropna()
    if s.empty:
        return {}
    out = {"n": int(len(s)), "min": s.min(), "p01": s.quantile(0.01), "p05": s.quantile(0.05),
           "median": s.median(), "p95": s.quantile(0.95), "p99": s.quantile(0.99), "max": s.max()}
    return {k: (v if k == "n" else round(float(v), 4)) for k, v in out.items()}


def calibration_pairs(frame: pd.DataFrame, n_pos: int, n_neg: int, seed: int) -> dict[str, np.ndarray]:
    """Known positives (two files of one filename group) and random different-group pairs."""
    rng = np.random.default_rng(seed)
    groups, labels = frame["file_group"].to_numpy(), frame["label"].to_numpy()
    members = frame.groupby("file_group").indices
    out = {}
    for kind in ("jpg_photo", "png_keras_aug"):
        multi = [ix for ix in members.values() if len(ix) > 1 and frame.at[ix[0], "source_type"] == kind]
        out[f"same_group_{SHORT_TYPE[kind]}"] = np.array(
            [rng.choice(multi[rng.integers(len(multi))], 2, replace=False) for _ in range(n_pos)])
    neg = []
    while len(neg) < n_neg:
        a, b = rng.integers(len(frame), size=2)
        if labels[a] == labels[b] and groups[a] != groups[b]:
            neg.append((min(a, b), max(a, b)))
    out["random_same_class_other_group"] = np.array(neg)
    return out


def hard_negative_rates(frame: pd.DataFrame, pairs: np.ndarray, inliers: np.ndarray, thresholds=(4, 5, 6, 7, 8, 10)) -> dict:
    """Share of *cross-class* shortlisted pairs (ground-truth different leaves) per inlier level."""
    lab, st = frame["label"].to_numpy(), frame["source_type"].to_numpy()
    i, j = pairs[:, 0], pairs[:, 1]
    cross = lab[i] != lab[j]
    has_png = (st[i] == "png_keras_aug") | (st[j] == "png_keras_aug")
    out = {}
    for name, mask in (("jpg-jpg", ~has_png), ("with_png", has_png)):
        sel = cross & mask
        out[name] = {"cross_class_candidates": int(sel.sum()),
                     **{f"rate_inliers_ge_{t}": round(float((inliers[sel] >= t).mean()), 5) if sel.any() else None
                        for t in thresholds}}
    return out


def describe_pairs(frame: pd.DataFrame, pairs: np.ndarray, inliers: np.ndarray, emb: np.ndarray, cfg: DedupConfig) -> pd.DataFrame:
    a = frame.iloc[pairs[:, 0]].reset_index(drop=True)
    b = frame.iloc[pairs[:, 1]].reset_index(drop=True)
    kind = ["-".join(sorted(k)) for k in zip(a["source_type"].map(SHORT_TYPE), b["source_type"].map(SHORT_TYPE))]
    return pd.DataFrame({
        "path_a": a["path"], "path_b": b["path"], "inliers": inliers,
        "linked": link_mask(frame, pairs, inliers, cfg),
        "sequence_pair": is_sequence_pair(frame, pairs, cfg.sequence_window),
        "embedding_sim": np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]]).round(4),
        "kind": kind, "label_a": a["label"], "label_b": b["label"],
        "cross_class": (a["label"] != b["label"]).to_numpy(),
        "file_group_a": a["file_group"], "file_group_b": b["file_group"],
    })


class PairCache:
    """Verifies each image pair once; later requests for the same pair are served from memory."""

    def __init__(self, verify):
        self.verify = verify
        self.inliers: dict[tuple[int, int], int] = {}

    def __call__(self, pairs: np.ndarray, desc: str = "verifying") -> np.ndarray:
        keys = [(int(i), int(j)) for i, j in pairs]
        new = sorted({k for k in keys if k not in self.inliers})
        if new:
            self.inliers.update(zip(new, self.verify(np.array(new), desc=desc).tolist()))
        return np.array([self.inliers[k] for k in keys], dtype=np.int32)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        keys = sorted(self.inliers)
        return np.array(keys, dtype=np.int64).reshape(-1, 2), np.array([self.inliers[k] for k in keys], dtype=np.int32)


def audit_split(frame, emb, cache: PairCache, cfg: DedupConfig, round_no: int) -> dict:
    """Check every val/test image against its ``audit_k`` most similar *training* images
    (a deeper search than the shortlist) with the same link rule."""
    is_train = frame["split"].to_numpy() == "train"
    held = np.nonzero(~is_train)[0]
    nn_idx, _ = nearest_other_groups(emb, frame["file_group"].tolist(), cfg.audit_k, query=held, allowed=is_train)
    pairs = candidate_pairs(nn_idx, query=held)
    before = len(cache.inliers)
    inliers = cache(pairs, desc=f"audit round {round_no}")
    found = pairs[link_mask(frame, pairs, inliers, cfg)]
    return {
        "round": round_no,
        "heldout_images_checked": int(len(held)),
        "train_neighbours_per_image": cfg.audit_k,
        "pairs_checked": int(len(pairs)),
        "pairs_newly_verified": int(len(cache.inliers) - before),
        "heldout_images_with_train_match": len({int(i) if not is_train[i] else int(j) for i, j in found}),
    }


def max_inliers_to_train(frame: pd.DataFrame, pairs: np.ndarray, inliers: np.ndarray) -> np.ndarray:
    """Per image: best same-class inlier count to any training image (-1 for training images)."""
    split, lab = frame["split"].to_numpy(), frame["label"].to_numpy()
    out = np.where(split == "train", -1, 0)
    i, j = pairs[:, 0], pairs[:, 1]
    keep = (lab[i] == lab[j]) & ((split[i] == "train") != (split[j] == "train"))
    held = np.where(split[i[keep]] == "train", j[keep], i[keep])
    np.maximum.at(out, held, inliers[keep])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=None)
    parser.add_argument("--recompute-embeddings", action="store_true")
    parser.add_argument("--keep-cache", action="store_true", help="keep the ~2 GB SIFT feature cache for re-runs")
    args = parser.parse_args()

    from . import viz
    from .utils import project_relative, save_json

    cfg = DedupConfig()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"Scanning {DATA_ROOT} for classes {CLASS_NAMES}")
    manifest = build_manifest(DATA_ROOT, CLASS_NAMES)
    manifest.to_csv(MANIFEST_CSV, index=False)
    manifest_summary = summarize_manifest(manifest)
    print(f"{manifest_summary['files_scanned']} files scanned, "
          f"{manifest_summary['exact_duplicates_removed']} exact duplicates removed, "
          f"{len(manifest_summary['unreadable_files'])} unreadable")

    frame = manifest[manifest["usable"]].reset_index(drop=True)
    paths = [str(DATA_ROOT / p) for p in frame["path"]]
    file_groups = frame["file_group"].tolist()
    emb = load_or_compute_embeddings(frame["path"].tolist(), cfg, args.device, args.recompute_embeddings)

    t0 = time.time()
    extract_features(paths, SIFT_STORE, cfg.sift_features, cfg.verify_workers)
    print(f"SIFT features ready in {time.time() - t0:.0f}s")
    cache = PairCache(partial(verify_pairs, store=SIFT_STORE, ratio=cfg.ratio, ransac_px=cfg.ransac_px,
                              workers=cfg.verify_workers))

    # 1) Calibrate the verifier on pairs whose answer is known from the filenames.
    calib = {name: cache(p, desc=f"calibrate {name}")
             for name, p in calibration_pairs(frame, 300, cfg.calibration_pairs, args.seed).items()}

    # 2) Candidate pairs: embedding shortlist + consecutive photo numbers; verify each one.
    nn_idx, _ = nearest_other_groups(emb, file_groups, cfg.shortlist_k)
    shortlist = candidate_pairs(nn_idx)
    seq = sequence_pairs(frame, cfg.sequence_window)
    pairs = np.unique(np.concatenate([shortlist, seq]), axis=0)
    print(f"Verifying {len(pairs)} candidate pairs ({len(shortlist)} from top-{cfg.shortlist_k} embedding "
          f"neighbours, {len(seq)} consecutive-photo pairs)")
    t0 = time.time()
    inliers = cache(pairs)
    print(f"Verified in {time.time() - t0:.0f}s")
    hard_neg = hard_negative_rates(frame, shortlist, cache(shortlist))

    # 3) Merge -> split -> audit; fold any leak the audit finds back in and split again.
    audit_rounds = []
    for round_no in range(1, MAX_AUDIT_ROUNDS + 1):
        all_pairs, all_inliers = cache.arrays()
        links = all_pairs[link_mask(frame, all_pairs, all_inliers, cfg)]
        frame["source_group"] = merge_groups(file_groups, links)
        groups = group_table(frame)
        frame["split"] = frame["source_group"].map(assign_splits(groups, SPLIT_FRACTIONS, LARGE_GROUP_MIN_FILES, args.seed))
        audit_rounds.append(audit_split(frame, emb, cache, cfg, round_no))
        print(f"Audit: {audit_rounds[-1]}", flush=True)
        if audit_rounds[-1]["heldout_images_with_train_match"] == 0:
            break
    all_pairs, all_inliers = cache.arrays()
    links = all_pairs[link_mask(frame, all_pairs, all_inliers, cfg)]
    frame["max_inliers_to_train"] = max_inliers_to_train(frame, all_pairs, all_inliers)

    # 4) Hard checks. Nothing is written as splits.csv unless they all pass.
    checks = leakage_checks(frame, links, audit_rounds[-1])
    split_path = SPLITS_CSV if checks["passed"] else SPLITS_CSV.with_name("splits_FAILED_CHECKS.csv")
    if not checks["passed"]:
        SPLITS_CSV.unlink(missing_ok=True)
    frame[SPLIT_COLUMNS].to_csv(split_path, index=False)

    pair_table = describe_pairs(frame, all_pairs, all_inliers, emb, cfg)
    pair_table[pair_table["inliers"] >= 3].sort_values("inliers", ascending=False).to_csv(
        DATA_DIR / "verified_pairs.csv", index=False)
    link_table = pair_table[pair_table["linked"]]
    cross_matches = pair_table[pair_table["cross_class"] & (pair_table["inliers"] >= cfg.min_inliers)]
    sizes = groups["n_files"]
    merged = groups[groups["source_group"].isin(frame.loc[frame["source_group"] != frame["file_group"], "source_group"])]
    group_has_jpg = frame.groupby("source_group")["source_type"].agg(lambda s: (s == "jpg_photo").any())
    png_sources = frame[frame["source_type"] == "png_keras_aug"].groupby("file_group")["source_group"].first()
    within = within_group_similarity(emb, file_groups)
    nearest_other = np.einsum("ij,ij->i", emb, emb[nn_idx[:, 0]])
    report = {
        "data_root": project_relative(DATA_ROOT),
        "seed": args.seed,
        "manifest": manifest_summary,
        "near_duplicates": {
            "method": "candidates = MobileNetV2 embedding top-k + consecutive photo numbers; "
                      "verified by SIFT (ratio + mutual NN) and RANSAC affine fit, mirror-aware",
            "config": cfg.__dict__,
            "embedding_similarity": {
                "within_filename_group_jpg": _stats(within[frame["source_type"] == "jpg_photo"]),
                "within_filename_group_png": _stats(within[frame["source_type"] == "png_keras_aug"]),
                "nearest_other_group": _stats(nearest_other),
            },
            "calibration_inliers": {k: _stats(v) for k, v in calib.items()},
            "calibration_rate_ge_min_inliers": {k: round(float((v >= cfg.min_inliers).mean()), 4) for k, v in calib.items()},
            "hard_negative_rates_cross_class": hard_neg,
            "candidate_pairs_verified": int(len(all_pairs)),
            "links": int(len(link_table)),
            "links_by_kind": link_table["kind"].value_counts().to_dict(),
            "links_from_sequence_rule_only": int((link_table["sequence_pair"] & (link_table["inliers"] < cfg.min_inliers)).sum()),
            "cross_class_matches_not_merged": cross_matches[["path_a", "path_b", "inliers"]].to_dict("records"),
            "png_groups": int(len(png_sources)),
            "png_groups_whose_source_leaf_is_also_a_jpg": int(group_has_jpg.loc[png_sources].sum()),
            "filename_groups": int(frame["file_group"].nunique()),
            "source_groups_after_merge": int(frame["source_group"].nunique()),
            "merged_source_groups": int(len(merged)),
            "largest_source_group_files": int(sizes.max()),
            "source_group_size_quantiles": _stats(sizes),
        },
        "split_fractions_of_groups": SPLIT_FRACTIONS,
        "splits": split_summary(frame),
        "audit_rounds": audit_rounds,
        "heldout_weak_matches": {
            split: {
                "images": int((frame["split"] == split).sum()),
                f"with_train_match_ge_{cfg.weak_match_inliers}_inliers": int(
                    ((frame["split"] == split) & (frame["max_inliers_to_train"] >= cfg.weak_match_inliers)).sum()),
            }
            for split in ("val", "test")
        },
        "leakage_checks": checks,
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    save_json(report, DATA_REPORT_JSON)

    # Figures for visual verification of the decisions.
    viz.embedding_similarity_histogram(within, nearest_other, frame["source_type"], DATA_DIR / "embedding_similarity.png")
    viz.inlier_histogram({**calib, "candidate_pairs": inliers}, cfg.min_inliers, DATA_DIR / "verification_calibration.png")
    viz.split_distribution(frame, DATA_DIR / "split_distribution.png")

    def sheet(rows: pd.DataFrame, name: str, tag: str) -> None:
        viz.pair_sheet([(r.path_a, r.path_b, f"{r.inliers} inl {r.kind} {tag}") for r in rows.itertuples()], DATA_DIR / name)

    ranked = pair_table.sort_values(["inliers", "path_a"])
    sheet(ranked[ranked["linked"] & ~ranked["sequence_pair"]].head(24), "linked_pairs_weakest.jpg", "linked")
    sheet(ranked[ranked["linked"] & ranked["sequence_pair"]].head(24), "linked_sequence_pairs_weakest.jpg", "linked(seq)")
    sheet(ranked[~ranked["linked"] & ~ranked["cross_class"]].tail(24), "rejected_pairs_strongest.jpg", "rejected")
    sheet(cross_matches.head(24), "cross_class_matches.jpg", "cross-class")
    png_jpg = link_table[link_table["kind"] == "jpg-png"].copy()
    png_jpg["png_group"] = np.where(png_jpg["path_a"].str.endswith(".png"), png_jpg["file_group_a"], png_jpg["file_group_b"])
    sheet(png_jpg.sort_values("inliers", ascending=False).drop_duplicates("png_group").head(24), "png_source_matches.jpg", "")
    if not args.keep_cache:
        shutil.rmtree(SIFT_STORE, ignore_errors=True)

    nd = report["near_duplicates"]
    print(f"Hard-negative (cross-class) inlier rates: {hard_neg}")
    print(f"Links: {nd['links']} {nd['links_by_kind']} ({nd['links_from_sequence_rule_only']} via the consecutive-photo rule); "
          f"cross-class matches not merged: {len(nd['cross_class_matches_not_merged'])}")
    print(f"PNG groups whose source leaf is also a JPG: {nd['png_groups_whose_source_leaf_is_also_a_jpg']}/{nd['png_groups']}")
    print(f"Filename groups {nd['filename_groups']} -> source groups {nd['source_groups_after_merge']} "
          f"(largest {nd['largest_source_group_files']} files)")
    for split, s in report["splits"].items():
        print(f"  {split:5s}: {s['files']:5d} files, {s['source_groups']:4d} groups, {s['files_per_class']}")
    print(f"Held-out images with an unlinked weak match (>= {cfg.weak_match_inliers} inliers) to train: "
          f"{report['heldout_weak_matches']}")
    print(f"Leakage checks: {checks}", flush=True)
    if not checks["passed"]:
        raise SystemExit("Leakage checks FAILED - splits.csv was not written.")
    print(f"Wrote {SPLITS_CSV} and {DATA_REPORT_JSON} in {report['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
