"""Fast unit checks for the data-handling and model utilities (no dataset or GPU needed).

Run from the project folder:  python -m tests.test_core   (or: python -m pytest tests)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from PIL import Image

from grapevine.dataset import GroupCappedSampler, NeutralizeBackground
from grapevine.config import DedupConfig
from grapevine.dedup import candidate_pairs, link_mask, merge_groups, sequence_pairs
from grapevine.geometric import mutual_ratio_matches
from grapevine.gradcam import GradCAM
from grapevine.manifest import parse_filename
from grapevine.models import build_model, configure_phase, gradcam_target_layer
from grapevine.split import assign_splits, group_table


def test_parse_filename():
    jpg = parse_filename("ESCA", "019afd88-6645-44c4-a5d2-bd583b877a50___FAM_B.Msls 1744_flipLR(1).JPG")
    assert jpg["source_type"] == "jpg_photo" and jpg["aug"] == "flipLR" and jpg["is_copy"]
    assert jpg["file_group"] == "ESCA|jpg|019afd88-6645-44c4-a5d2-bd583b877a50"
    png = parse_filename("Healthy", "_105_1066656.png")
    assert png["source_type"] == "png_keras_aug" and png["file_group"] == "Healthy|png|105" and not png["is_copy"]
    assert parse_filename("Healthy", "random name.jpg")["source_type"] == "unknown"


def test_group_capped_sampler():
    groups = ["a"] * 56 + ["b"] * 2 + ["c"]
    sampler = GroupCappedSampler(groups, cap=16, seed=0)
    assert len(sampler) == 19
    sampler.set_epoch(1)
    first = list(sampler)
    assert sorted(first) == sorted(set(first)) and len(first) == 19
    assert sum(groups[i] == "a" for i in first) == 16
    sampler.set_epoch(1)
    assert list(sampler) == first  # deterministic per epoch
    sampler.set_epoch(2)
    assert list(sampler) != first  # a different subset/order each epoch


def test_assign_splits_is_group_disjoint_and_stratified():
    rows = []
    for label in ("A", "B"):
        rows += [{"source_group": f"{label}|png|{k}", "label": label, "source_type": "png_keras_aug"}
                 for k in range(24) for _ in range(56)]
        rows += [{"source_group": f"{label}|jpg|{k}", "label": label, "source_type": "jpg_photo"}
                 for k in range(400) for _ in range(2)]
    frame = pd.DataFrame(rows)
    assignment = assign_splits(group_table(frame), {"train": 0.7, "val": 0.15, "test": 0.15}, 20, seed=1)
    frame["split"] = frame["source_group"].map(assignment)
    assert (frame.groupby("source_group")["split"].nunique() == 1).all()
    per = frame.drop_duplicates("source_group").groupby(["label", "split"]).size()
    for label in ("A", "B"):
        assert per[(label, "test")] == 4 + 60 and per[(label, "val")] == 4 + 60  # 24*0.15 -> 4, 400*0.15 -> 60


def test_merge_groups_is_transitive():
    groups = ["g1", "g2", "g3", "g4"]
    merged = merge_groups(groups, np.array([[0, 1], [1, 2]]))
    assert merged[0] == merged[1] == merged[2] != merged[3]


def test_candidate_pairs_unique_unordered():
    pairs = candidate_pairs(np.array([[1, 2], [0, 2], [0, 1]]))
    assert pairs.tolist() == [[0, 1], [0, 2], [1, 2]]


def test_sequence_pairs_and_link_rule():
    frame = pd.DataFrame({
        "label": ["A", "A", "A", "A", "B"],
        "photo_tag": ["t", "t", "t", "u", "t"],
        "photo_num": [10, 12, 20, 11, 11],
        "file_group": ["g0", "g1", "g2", "g3", "g4"],
    })
    assert sequence_pairs(frame, 3).tolist() == [[0, 1]]  # same class + tag, numbers 2 apart
    cfg = DedupConfig()
    pairs = np.array([[0, 1], [0, 2], [0, 4], [0, 2]])
    inl = np.array([cfg.min_inliers_sequence, cfg.min_inliers_sequence, 50, cfg.min_inliers])
    # consecutive photos need fewer inliers; cross-class pairs are never linked
    assert link_mask(frame, pairs, inl, cfg).tolist() == [True, False, False, True]


def test_neutralize_background_keeps_leaf():
    img = np.full((64, 64, 3), (190, 185, 180), np.uint8)  # grey-beige paper
    img[16:48, 16:48] = (40, 140, 30)                         # saturated green "leaf"
    img[28:36, 28:36] = (200, 200, 200)                       # pale spot inside the leaf
    out = np.asarray(NeutralizeBackground()(Image.fromarray(img)))
    assert (out[:8] == 128).all()                             # border background painted grey
    assert (out[20, 20] == (40, 140, 30)).all()               # leaf untouched
    assert (out[30, 30] == (200, 200, 200)).all()             # pale interior not connected to the border


def test_mutual_ratio_matches():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (50, 128)).astype(np.float32)
    keep, idx = mutual_ratio_matches(a, (a * a).sum(1), a, (a * a).sum(1), 0.75)
    assert len(keep) == 50 and (idx == keep).all()  # identical descriptor sets match one-to-one


def test_phases_and_gradcam():
    torch.manual_seed(0)
    for name in ("mobilenet_v2", "resnet50", "efficientnet_b0"):
        model = build_model(name, 4, pretrained=False)
        frozen = configure_phase(model, name, "head")
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trainable and all(n.startswith(("fc", "classifier")) for n in trainable), name
        configure_phase(model, name, "finetune")
        assert any(p.requires_grad for p in gradcam_target_layer(model, name).parameters())
        assert frozen
        cam, probs, idx = GradCAM(model, gradcam_target_layer(model, name))(torch.randn(1, 3, 224, 224))
        assert cam.shape == (224, 224) and 0 <= cam.min() and cam.max() <= 1 + 1e-6
        assert abs(probs.sum() - 1) < 1e-4 and 0 <= idx < 4


if __name__ == "__main__":
    for fn_name, fn in sorted(globals().items()):
        if fn_name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {fn_name}")
