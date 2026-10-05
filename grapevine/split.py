"""Stratified, group-aware train/val/test split and the leakage checks run on it."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def group_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per source group with its label, size and whether it holds Keras PNG files."""
    g = frame.groupby("source_group")
    table = pd.DataFrame(
        {
            "label": g["label"].agg(lambda s: s.mode().iat[0]),
            "n_labels": g["label"].nunique(),
            "n_files": g.size(),
            "has_png": g["source_type"].agg(lambda s: (s == "png_keras_aug").any()),
        }
    )
    return table.reset_index()


def assign_splits(groups: pd.DataFrame, fractions: dict[str, float], large_min_files: int, seed: int) -> dict[str, str]:
    """Allocate whole source groups to splits, stratified by class and by group size.

    Large groups (the ~56-file Keras PNG groups) and small groups (JPG photo groups of 1-5
    files) are shuffled and divided separately within each class, so each split receives its
    fair share of both kinds and the per-class file counts stay close to the target fractions.
    """
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    for label in sorted(groups["label"].unique()):
        for large in (True, False):
            sel = groups[(groups["label"] == label) & ((groups["n_files"] >= large_min_files) == large)]
            names = sorted(sel["source_group"])
            if not names:
                continue
            order = [names[k] for k in rng.permutation(len(names))]
            n_test = int(np.floor(len(names) * fractions["test"] + 0.5))
            n_val = int(np.floor(len(names) * fractions["val"] + 0.5))
            for name in order[:n_test]:
                assignment[name] = "test"
            for name in order[n_test : n_test + n_val]:
                assignment[name] = "val"
            for name in order[n_test + n_val :]:
                assignment[name] = "train"
    return assignment


def leakage_checks(frame: pd.DataFrame, links: np.ndarray, audit: dict) -> dict:
    """Verify that no source group, identical file or verified same-leaf pair crosses splits."""
    split = frame["split"].to_numpy()
    checks = {
        "unassigned_files": int(frame["split"].isna().sum()),
        "source_groups_in_multiple_splits": int((frame.groupby("source_group")["split"].nunique() > 1).sum()),
        "filename_groups_in_multiple_splits": int((frame.groupby("file_group")["split"].nunique() > 1).sum()),
        "identical_files_in_multiple_splits": int((frame.groupby("md5")["split"].nunique() > 1).sum()),
        "verified_same_leaf_pairs_across_splits": int((split[links[:, 0]] != split[links[:, 1]]).sum()) if len(links) else 0,
        "audit_heldout_images_with_train_match": int(audit["heldout_images_with_train_match"]),
        "classes_missing_from_a_split": [
            f"{label} not in {s}" for s in SPLITS for label in sorted(frame["label"].unique())
            if not ((frame["split"] == s) & (frame["label"] == label)).any()
        ],
    }
    checks["passed"] = all(v == 0 for k, v in checks.items() if isinstance(v, int)) and not checks["classes_missing_from_a_split"]
    return checks


def split_summary(frame: pd.DataFrame) -> dict:
    out = {}
    for split in SPLITS:
        part = frame[frame["split"] == split]
        out[split] = {
            "files": int(len(part)),
            "source_groups": int(part["source_group"].nunique()),
            "files_per_class": part["label"].value_counts().sort_index().to_dict(),
            "groups_per_class": part.groupby("label")["source_group"].nunique().sort_index().to_dict(),
            "original_photos_per_class": part[part["is_original"]]["label"].value_counts().sort_index().to_dict(),
            "png_files_per_class": part[part["source_type"] == "png_keras_aug"]["label"].value_counts().sort_index().to_dict(),
        }
    return out
