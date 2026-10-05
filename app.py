"""Streamlit app: upload a grape-leaf photo and get the predicted disease, confidence and Grad-CAM.

Run from the project folder:

    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from grapevine.config import BEST_MODEL_DIR
from grapevine.predict import Predictor
from grapevine.utils import load_json

# Override with GRAPEVINE_CHECKPOINT=artifacts/runs/<model>/best.pt to serve another trained model.
CHECKPOINT = Path(os.environ.get("GRAPEVINE_CHECKPOINT", BEST_MODEL_DIR / "model.pt"))
MODEL_CARD = BEST_MODEL_DIR / "model_card.json"
LOW_CONFIDENCE = 0.70
CLASS_NOTES = {
    "Black Rot": "Fungal disease (Guignardia bidwellii): round tan-to-brown lesions with dark borders.",
    "ESCA": "Esca / black measles (trunk-disease fungi): 'tiger-stripe' discolouration between the veins.",
    "Leaf Blight": "Isariopsis leaf spot (Pseudocercospora vitis): irregular dark-brown spots that can merge.",
    "Healthy": "No disease symptoms recognised.",
}

st.set_page_config(page_title="Grapevine Leaf Disease Classifier", page_icon="🍇", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> Predictor:
    # CPU keeps the app independent of any training job that may be using the GPU.
    return Predictor(CHECKPOINT, device="cpu")


st.title("Grapevine Leaf Disease Classifier")
st.caption("Classifies a single grape leaf as Black Rot, ESCA, Leaf Blight or Healthy, "
           "and shows a Grad-CAM heatmap of the regions that drove the prediction.")

if not CHECKPOINT.exists():
    st.error(f"No trained model found at `{CHECKPOINT}`. Run the training pipeline first (see README).")
    st.stop()

predictor = get_predictor()
card = load_json(MODEL_CARD) if MODEL_CARD.exists() else None
if card and card["model"] != predictor.model_name:
    card = None  # the model card describes a different checkpoint

with st.sidebar:
    st.header("Model")
    st.write(f"Architecture: **{predictor.model_name}**")
    if card:
        test = card["test"]["all"]
        st.write(f"Selected by: {card['selection_rule']}")
        st.write(f"Held-out test set (n={test['n']} images, leaf-level split):")
        st.write(f"- Accuracy: **{test['accuracy']:.2%}**\n- Macro-F1: **{test['macro_f1']:.4f}**")
        orig = card["test"].get("original_photos")
        if orig:
            st.write(f"- Un-augmented photos only (n={orig['n']}): accuracy **{orig['accuracy']:.2%}**")
    st.divider()
    show_cam = st.toggle("Show Grad-CAM", value=True)
    st.info("Trained on lab-style photos of single leaves on a plain background. "
            "Results on field photos, other crops or other diseases are not validated.")

uploaded = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"])
if uploaded is None:
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except (UnidentifiedImageError, OSError):
    st.error("This file could not be read as an image.")
    st.stop()

with st.spinner("Classifying..."):
    pred = predictor.predict(image, gradcam=show_cam)

left, right = st.columns(2)
left.image(image, caption="Uploaded image", width="stretch")
if show_cam and pred.overlay is not None:
    right.image(pred.overlay, caption=f"Grad-CAM for '{pred.label}' (red = most influential)", width="stretch")

st.subheader(f"Prediction: {pred.label}")
st.metric("Confidence", f"{pred.confidence:.1%}")
conf_stats = (card or {}).get("test_confidence")
if conf_stats:
    st.caption(f"Confidence is the softmax probability of the top class. Training used label smoothing (0.1), so "
               f"even clear cases stay below 100%: on the held-out test set the median was "
               f"{conf_stats['median']:.1%} (5th percentile {conf_stats['p05']:.1%}).")
st.write(CLASS_NOTES.get(pred.label, ""))
if pred.confidence < LOW_CONFIDENCE:
    st.warning(f"Low confidence (< {LOW_CONFIDENCE:.0%}). The image may differ from the training data "
               "(background, lighting, several leaves, other plant) - treat this result with caution.")

probs = pd.DataFrame({"class": list(pred.probabilities), "probability": list(pred.probabilities.values())})
st.bar_chart(probs.set_index("class"), y="probability", horizontal=True)
st.dataframe(probs.sort_values("probability", ascending=False).style.format({"probability": "{:.2%}"}),
             hide_index=True, width="stretch")
st.caption("Decision support only - confirm a diagnosis with an agronomist or plant pathologist.")
