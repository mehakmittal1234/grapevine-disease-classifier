"""Read the pipeline's saved reports for the app. Every figure shown in the app comes from these files."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from grapevine.config import BEST_MODEL_DIR, DATA_REPORT_JSON, MODEL_NAMES, REPORTS_DIR, RUNS_DIR
from grapevine.utils import load_json

MODEL_LABELS = {"efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNetV2", "resnet50": "ResNet50"}
REPO_URL = "https://github.com/mehakmittal1234/grapevine-disease-classifier"


@st.cache_data(show_spinner=False)
def model_card() -> dict | None:
    path = BEST_MODEL_DIR / "model_card.json"
    return load_json(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def data_report() -> dict | None:
    return load_json(DATA_REPORT_JSON) if DATA_REPORT_JSON.exists() else None


@st.cache_data(show_spinner=False)
def background_bias() -> dict | None:
    path = DATA_REPORT_JSON.parent / "background_bias.json"
    return load_json(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def comparison() -> pd.DataFrame | None:
    path = REPORTS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    return df


@st.cache_data(show_spinner=False)
def histories() -> pd.DataFrame | None:
    frames = []
    for name in MODEL_NAMES:
        path = RUNS_DIR / name / "history.csv"
        if path.exists():
            frames.append(pd.read_csv(path).assign(model=name, label=MODEL_LABELS.get(name, name)))
    return pd.concat(frames, ignore_index=True) if frames else None


@st.cache_data(show_spinner=False)
def test_predictions(model: str) -> pd.DataFrame | None:
    path = REPORTS_DIR / model / "test_predictions.csv"
    return pd.read_csv(path, usecols=["label", "pred", "confidence", "is_original"]) if path.exists() else None


def summary(model: str) -> dict | None:
    """Conservative, file-backed headline figures for the served model (None if the card is for another model)."""
    card = model_card()
    if not card or card.get("model") != model:
        return None
    ci = card.get("test_accuracy_exact_95ci_one_image_per_leaf") or {}
    background = (card.get("background_neutralized_test") or {}).get("all") or {}
    splits = (card.get("data") or {}).get("splits") or {}
    return {
        "accuracy_low": (ci.get("interval") or [None])[0],
        "leaves": ci.get("n"),
        "test_images": card["test"]["all"]["n"],
        "train_images": (splits.get("train") or {}).get("files"),
        "background_accuracy": background.get("accuracy"),
        "background_errors": background.get("errors"),
        "background_n": background.get("n"),
        "confidence_median": (card.get("test_confidence") or {}).get("median"),
        "confidence_p05": (card.get("test_confidence") or {}).get("p05"),
        "selection_rule": card.get("selection_rule"),
    }
