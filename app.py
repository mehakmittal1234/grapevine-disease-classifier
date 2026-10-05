"""Streamlit app: upload a grape-leaf photo and get the predicted disease, confidence and Grad-CAM.

Run from the project folder:

    streamlit run app.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from grapevine.config import BEST_MODEL_DIR
from grapevine.gradcam import overlay_heatmap
from grapevine.predict import Predictor
from grapevine.utils import load_json

# Override with GRAPEVINE_CHECKPOINT=artifacts/runs/<model>/best.pt to serve another trained model.
CHECKPOINT = Path(os.environ.get("GRAPEVINE_CHECKPOINT", BEST_MODEL_DIR / "model.pt"))
MODEL_CARD = BEST_MODEL_DIR / "model_card.json"
REPO_URL = "https://github.com/mehakmittal1234/grapevine-disease-classifier"
LOW_CONFIDENCE = 0.70
MODEL_LABELS = {"efficientnet_b0": "EfficientNet-B0", "mobilenet_v2": "MobileNetV2", "resnet50": "ResNet50"}
CLASSES = {
    "Black Rot": ("#4A2F22", "Fungal disease (Guignardia bidwellii): round tan-to-brown spots with dark "
                             "borders and tiny black fruiting bodies."),
    "ESCA": ("#A3472A", "Esca / black measles, a trunk disease: 'tiger-stripe' yellow-to-red discolouration "
                        "between the veins that dries to brown."),
    "Leaf Blight": ("#8A6D0B", "Isariopsis leaf spot (Pseudocercospora vitis): irregular dark-brown spots "
                               "that merge and dry out the leaf."),
    "Healthy": ("#2E7D32", "Uniformly green leaf without disease symptoms."),
}

st.set_page_config(page_title="Grapevine Leaf Disease Classifier", page_icon="🍇", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1150px;}
.hero {background: linear-gradient(135deg, #24502A 0%, #3F7D3A 55%, #7C8F38 100%); color: #FFFFFF;
       border-radius: 18px; padding: 28px 32px 22px; margin-bottom: 1.6rem;}
.hero-kicker {text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; font-weight: 600; opacity: .85;}
.hero h1 {color: #FFFFFF; font-size: 2.15rem; line-height: 1.15; margin: .25rem 0 .5rem; padding: 0;}
.hero p {color: #E7F0DF; font-size: 1.02rem; max-width: 46rem; margin: 0 0 1rem;}
.chips {display: flex; flex-wrap: wrap; gap: 8px;}
.chip {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.30); color: #FFFFFF;
       border-radius: 999px; padding: 4px 12px; font-size: .85rem;}
.hero-note {color: #D8E6CE; font-size: .78rem; margin-top: .7rem;}
.step {font-weight: 700; color: #1E2A1E; font-size: 1.05rem; margin: .2rem 0 .6rem;}
.step span {color: #3F7D3A;}
.card {background: #FFFFFF; border: 1px solid #DFE7D6; border-radius: 14px; padding: 18px 20px;
       box-shadow: 0 1px 2px rgba(30,42,30,.05); margin-bottom: .9rem;}
.eyebrow {font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; color: #5F6E5A; font-weight: 600;}
.result-row {display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: .4rem 0 .7rem;}
.badge {display: inline-block; color: #FFFFFF; border-radius: 10px; padding: 6px 14px; font-size: 1.35rem; font-weight: 700;}
.conf {text-align: right; line-height: 1.1;}
.conf b {font-size: 1.9rem; color: #1E2A1E;}
.conf span {display: block; font-size: .78rem; color: #5F6E5A;}
.note {color: #33402F; font-size: .95rem; margin: 0;}
.prob-row {display: grid; grid-template-columns: 96px 1fr 56px; align-items: center; gap: 10px; margin: 8px 0;}
.prob-name {font-size: .9rem; color: #33402F;}
.prob-track {background: #ECF1E5; border-radius: 999px; height: 10px; overflow: hidden;}
.prob-fill {height: 100%; border-radius: 999px;}
.prob-val {font-variant-numeric: tabular-nums; font-size: .9rem; text-align: right; color: #33402F;}
.legend-item {display: flex; gap: 12px; align-items: flex-start; margin: 12px 0;}
.dot {width: 12px; height: 12px; border-radius: 50%; margin-top: 6px; flex: none;}
.legend-item b {display: block; color: #1E2A1E;}
.legend-item span {color: #5F6E5A; font-size: .9rem;}
.footer {color: #6B7867; font-size: .8rem; text-align: center; margin: 2rem 0 .5rem;}
@media (max-width: 640px) {.hero {padding: 20px 20px 16px;} .hero h1 {font-size: 1.6rem;} .hero p {font-size: .95rem;} .chip {font-size: .8rem;}}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the model...")
def get_predictor() -> Predictor:
    # CPU keeps the app independent of any training job that may be using the GPU.
    return Predictor(CHECKPOINT, device="cpu")


@st.cache_data(show_spinner=False, max_entries=16)
def classify(image_bytes: bytes) -> dict:
    """Run the model once per uploaded file; reruns (view toggle, slider) reuse the result."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    pred = get_predictor().predict(image, gradcam=True)
    return {"image": image, "label": pred.label, "confidence": pred.confidence,
            "probabilities": pred.probabilities, "cam": pred.cam}


def evaluation_summary(model_name: str) -> dict | None:
    """Conservative, file-backed evaluation figures for the selected model (None if unavailable)."""
    if not MODEL_CARD.exists():
        return None
    card = load_json(MODEL_CARD)
    ci = card.get("test_accuracy_exact_95ci_one_image_per_leaf")
    if card.get("model") != model_name or not ci:
        return None  # the card describes a different checkpoint
    background = (card.get("background_neutralized_test") or {}).get("all", {})
    return {
        "accuracy_low": ci["interval"][0],
        "leaves": ci["n"],
        "test_images": card["test"]["all"]["n"],
        "train_images": ((card.get("data") or {}).get("splits") or {}).get("train", {}).get("files"),
        "background_accuracy": background.get("accuracy"),
        "confidence_median": (card.get("test_confidence") or {}).get("median"),
    }


if not CHECKPOINT.exists():
    st.error(f"No trained model found at `{CHECKPOINT}`. Run the training pipeline first (see README).")
    st.stop()

predictor = get_predictor()
summary = evaluation_summary(predictor.model_name)
model_label = MODEL_LABELS.get(predictor.model_name, predictor.model_name)

chips = [model_label, "4 classes", "Grad-CAM explanations"]
if summary:
    chips.insert(1, f"≥{summary['accuracy_low']:.1%} accuracy on unseen leaves*")
hero_note = (
    f"<div class='hero-note'>*Exact 95% lower bound on {summary['leaves']} held-out leaves (lab photos on a plain "
    "background). Accuracy on field photos has not been measured and is likely lower.</div>" if summary else ""
)
st.markdown(
    f"""
<div class="hero">
  <div class="hero-kicker">Plant disease detection</div>
  <h1>Grapevine Leaf Disease Classifier</h1>
  <p>Upload a photo of a single grape leaf. The model identifies Black Rot, ESCA or Leaf Blight
     (or a healthy leaf) and highlights the regions that drove its decision.</p>
  <div class="chips">{''.join(f'<span class="chip">{c}</span>' for c in chips)}</div>
  {hero_note}
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([5, 6], gap="large")

with left:
    st.markdown('<div class="step"><span>1</span> · Upload a leaf photo</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")
    st.caption("Best results: one leaf per photo, plain background, the leaf filling most of the frame, sharp focus.")

result = None
if uploaded is not None:
    try:
        with st.spinner("Analysing the leaf..."):
            result = classify(uploaded.getvalue())
    except (UnidentifiedImageError, OSError):
        with left:
            st.error("This file could not be read as an image.")

with left:
    if result:
        st.markdown('<div class="step"><span>2</span> · Diagnosis</div>', unsafe_allow_html=True)
        color, note = CLASSES.get(result["label"], ("#3F7D3A", ""))
        st.markdown(
            f"""
<div class="card">
  <div class="eyebrow">Predicted class</div>
  <div class="result-row">
    <span class="badge" style="background:{color}">{result['label']}</span>
    <div class="conf"><b>{result['confidence']:.1%}</b><span>confidence</span></div>
  </div>
  <p class="note">{note}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if result["confidence"] < LOW_CONFIDENCE:
            st.warning(f"Low confidence (below {LOW_CONFIDENCE:.0%}). The photo may differ from the training data "
                       "(background, lighting, several leaves, another plant); treat this result with caution.")
        rows = "".join(
            f'<div class="prob-row"><span class="prob-name">{name}</span><div class="prob-track">'
            f'<div class="prob-fill" style="width:{100 * p:.1f}%;background:{CLASSES.get(name, ("#3F7D3A",))[0]}"></div>'
            f'</div><span class="prob-val">{p:.1%}</span></div>'
            for name, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1])
        )
        st.markdown(f'<div class="card"><div class="eyebrow">Class probabilities</div>{rows}</div>',
                    unsafe_allow_html=True)
        typical = f" (median {summary['confidence_median']:.0%} on the test set)" if summary and summary["confidence_median"] else ""
        st.caption("Confidence is the softmax probability of the top class. The model was trained with label "
                   f"smoothing, so even clear cases stay below 100%{typical}.")

with right:
    if not result:
        st.markdown('<div class="step"><span>2</span> · Diagnosis</div>', unsafe_allow_html=True)
        legend = "".join(
            f'<div class="legend-item"><div class="dot" style="background:{color}"></div>'
            f"<div><b>{name}</b><span>{note}</span></div></div>"
            for name, (color, note) in CLASSES.items()
        )
        st.markdown(f'<div class="card"><div class="eyebrow">What the model recognises</div>{legend}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="step"><span>3</span> · Where the model looked</div>', unsafe_allow_html=True)
        view = st.segmented_control("View", ["Grad-CAM", "Original"], default="Grad-CAM", label_visibility="collapsed")
        if view == "Original":
            st.image(result["image"], caption="Uploaded photo", width="stretch")
        else:
            alpha = st.slider("Heatmap opacity", 0.1, 0.9, 0.45, 0.05)
            st.image(overlay_heatmap(result["image"], result["cam"], alpha), width="stretch",
                     caption=f"Grad-CAM for '{result['label']}': red marks the regions that most influenced the prediction")

with st.expander("About this model and its limits"):
    if summary:
        trained = f"{summary['train_images']:,} " if summary["train_images"] else ""
        background = (f" With the background painted grey (a check against background shortcuts), accuracy "
                      f"was {summary['background_accuracy']:.1%}." if summary["background_accuracy"] else "")
        st.markdown(
            f"- **Model:** {model_label}, fine-tuned from ImageNet on {trained}lab photos of single grape leaves.\n"
            f"- **Evaluation:** {summary['test_images']:,} held-out photos of {summary['leaves']} leaves never seen "
            "in training (duplicates, augmented copies and re-shoots of the same leaf were kept on one side of the "
            f"split). Conservative accuracy: **≥{summary['accuracy_low']:.1%}** (exact 95% lower bound, one photo "
            f"per leaf).{background}\n"
            "- **Limits:** trained on single leaves on a plain background. Field photos, other grape diseases "
            "(e.g. downy or powdery mildew) and other crops are not validated, and accuracy there is likely lower.\n"
            f"- **Details:** data audit, training and full evaluation on [GitHub]({REPO_URL})."
        )
    else:
        st.markdown(f"Trained on lab photos of single grape leaves. Details on [GitHub]({REPO_URL}).")

st.markdown('<div class="footer">Decision support only: confirm a diagnosis with an agronomist or plant '
            "pathologist.</div>", unsafe_allow_html=True)
