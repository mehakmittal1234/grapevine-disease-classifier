"""Central configuration: paths, class names and default hyperparameters.

Every script reads its defaults from here, so a run is fully described by this file plus the
command-line overrides that each run writes to ``artifacts/runs/<model>/config.json``.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# The four class folders live one level above the project folder (the dataset is left untouched).
DATA_ROOT = Path(os.environ.get("GRAPEVINE_DATA_ROOT", PROJECT_ROOT.parent)).resolve()
ARTIFACTS_DIR = Path(os.environ.get("GRAPEVINE_ARTIFACTS", PROJECT_ROOT / "artifacts")).resolve()
DATA_DIR = ARTIFACTS_DIR / "data"
RUNS_DIR = ARTIFACTS_DIR / "runs"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
BEST_MODEL_DIR = ARTIFACTS_DIR / "best_model"

MANIFEST_CSV = DATA_DIR / "manifest.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"
DATA_REPORT_JSON = DATA_DIR / "data_report.json"

CLASS_NAMES: tuple[str, ...] = ("Black Rot", "ESCA", "Healthy", "Leaf Blight")
MODEL_NAMES: tuple[str, ...] = ("mobilenet_v2", "resnet50", "efficientnet_b0")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224
SEED = 42

# Fractions of *source groups* (not files) per class that go to each split.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
# Groups with at least this many files are allocated separately so that every split receives
# its share of the large Keras-augmented PNG groups (~56 files each).
LARGE_GROUP_MIN_FILES = 20


@dataclass
class DedupConfig:
    """Two-stage near-duplicate search (embedding shortlist + SIFT/RANSAC verification)."""

    embed_batch_size: int = 64
    num_workers: int = 2
    # Candidate pairs: each image's k most similar images from other filename groups, plus
    # every pair of JPG photos with the same photographer tag and photo numbers <= window apart.
    shortlist_k: int = 15
    sequence_window: int = 3
    # Geometric verification.
    sift_features: int = 600
    ratio: float = 0.75
    ransac_px: float = 3.0
    # Link rule (same class only). Cross-class candidate pairs are ground-truth *different*
    # leaves; among 10,252 of them, >= 6 inliers occurred in 0.08% (JPG-JPG) / 0.12% (with PNG)
    # and >= 8 in none (artifacts/data/data_report.json). Consecutive photo numbers are a strong
    # same-leaf prior (65% of 6-inlier shortlisted JPG pairs vs 0.74% of random same-class
    # pairs), so those pairs need only 6.
    # Lower thresholds chained whole photo sequences into one group through union-find.
    min_inliers: int = 8
    min_inliers_sequence: int = 6
    # Held-out images whose best match to any training image reaches this many inliers (but
    # misses the link rule) are excluded from the "strict" evaluation view.
    weak_match_inliers: int = 6
    verify_workers: int = 6
    calibration_pairs: int = 1500
    # Post-split audit: every val/test image is checked against its audit_k most similar
    # training images; any pair meeting the link rule is merged and the split is redone.
    audit_k: int = 40


@dataclass
class TrainConfig:
    model: str = "mobilenet_v2"
    image_size: int = IMAGE_SIZE
    batch_size: int = 32
    # Phase 1: train only the new classifier head on top of the frozen ImageNet backbone.
    head_epochs: int = 2
    head_lr: float = 1e-3
    # Phase 2: fine-tune the whole network with a cosine-decayed learning rate.
    finetune_epochs: int = 8
    finetune_lr: float = 2e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    # Early stopping on validation macro-F1 during phase 2.
    patience: int = 3
    # Per epoch, draw at most this many images from each source group (the PNG groups hold
    # ~56 augmented views of a single leaf). ``0`` disables the cap.
    group_cap: int = 16
    amp: bool = False
    num_workers: int = 2
    seed: int = SEED
    # Debug helper: stop each epoch after this many batches (``0`` = full epoch).
    max_batches: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# Settings that differ per architecture. Mixed precision only pays off for ResNet50 on MPS
# (measured ~2x faster); it does not speed up the depthwise convolutions of the other two.
MODEL_OVERRIDES: dict[str, dict] = {
    "mobilenet_v2": {},
    "resnet50": {"amp": True},
    "efficientnet_b0": {},
}


def train_config_for(model: str, **overrides) -> TrainConfig:
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {model!r}; choose from {MODEL_NAMES}")
    values = {"model": model, **MODEL_OVERRIDES[model]}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**values)
