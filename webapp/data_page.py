"""Data page: how the dataset was cleaned and split so that no leaf appears on both sides of the split."""

from __future__ import annotations

import streamlit as st

from . import charts, reports, style

CHECK_NAMES = {
    "unassigned_files": "Every usable file assigned to a split",
    "source_groups_in_multiple_splits": "No leaf group in two splits",
    "filename_groups_in_multiple_splits": "No filename group in two splits",
    "identical_files_in_multiple_splits": "No identical file in two splits",
    "verified_same_leaf_pairs_across_splits": "No verified same-leaf pair across splits",
    "audit_heldout_images_with_train_match": "No held-out photo matching a training leaf",
}


def render() -> None:
    report = reports.data_report()
    style.intro("Every leaf is tested once, on a leaf the model never saw.",
                "The folders mix original photos with flipped, rotated and re-shot copies of the same leaves. "
                "Splitting by file would leak those copies into the test set and inflate the score, so the "
                "photos were grouped by physical leaf first.")
    if not report:
        st.info("The data report is not available in this deployment.")
        return

    m, nd, s = report["manifest"], report["near_duplicates"], report["splits"]
    audits = report["audit_rounds"]
    steps = [
        ("Scanned the four class folders",
         f"{len(m['unreadable_files'])} unreadable files. Two formats: 256 px JPG photos and 224 px PNGs made "
         "by Keras augmentation.", f"{m['files_scanned']:,}", "files"),
        ("Removed byte-identical duplicates",
         f"MD5 hashing found {m['exact_duplicates_removed']} exact copies, none across classes.",
         f"{m['usable_files']:,}", "usable files"),
        ("Found copies of the same leaf",
         "Candidates from image-embedding neighbours and consecutive photo numbers, confirmed by SIFT keypoint "
         f"matching with a RANSAC affine fit (mirror-aware). {nd['candidate_pairs_verified']:,} pairs checked.",
         f"{nd['links']:,}", "same-leaf links"),
        ("Grouped photos by physical leaf",
         f"Linked files merged into leaf groups; the largest holds {nd['largest_source_group_files']} files. "
         "Matches between different classes were never merged.",
         f"{nd['source_groups_after_merge']:,}", "leaves"),
        ("Split by leaf, not by file",
         "70 / 15 / 15 of the leaves in each class went to training, validation and test.",
         f"{s['test']['files']:,}", "test photos"),
        ("Audited the held-out photos",
         f"Each validation and test photo was checked against its {audits[0]['train_neighbours_per_image']} "
         f"nearest training photos. Any match was merged into its leaf group and the split redone, "
         f"{len(audits)} rounds in total.",
         f"{audits[-1]['heldout_images_with_train_match']}", "matches left"),
    ]
    items = "".join(f'<li><h4>{t}</h4><div class="num">{n}<small>{u}</small></div><p>{d}</p></li>'
                    for t, d, n, u in steps)
    style.html(f'<ol class="steps">{items}</ol>')

    left, right = st.columns([3, 2], gap="large")
    with left:
        style.section("Files in each split", "Hover a segment for the count of files and leaves.")
        st.altair_chart(charts.split_composition(report), width="stretch", theme=None)
    with right:
        style.section("Audit rounds", "Photos in validation or test that still matched a training leaf.")
        st.altair_chart(charts.audit_rounds(report), width="stretch", theme=None)

    checks = report.get("leakage_checks") or {}
    style.section("Checks the split had to pass",
                  "The split file is written only if all of these hold. Each count below is from the saved report.")
    rows = "".join(f'<div class="check"><i>{"✓" if checks.get(k) in (0, []) else "✗"}</i>'
                   f"<span>{label} ({checks.get(k)})</span></div>"
                   for k, label in CHECK_NAMES.items() if k in checks)
    style.html(f'<div class="checks">{rows}</div>')

    weak = report.get("heldout_weak_matches", {}).get("test")
    if weak:
        st.caption(f"{weak['with_train_match_ge_6_inliers']} of {weak['images']:,} test photos have a weak match "
                   "(6 or more keypoints) to a training photo. The evaluation also reports scores without them.")
