"""Diagnose page: upload a leaf, get the predicted disease, confidence and an interactive Grad-CAM view."""

from __future__ import annotations

import base64
import hashlib
import io
import os
from pathlib import Path

import matplotlib
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from grapevine.config import BEST_MODEL_DIR
from grapevine.predict import Predictor

from . import reports, style

# Override with GRAPEVINE_CHECKPOINT=artifacts/runs/<model>/best.pt to serve another trained model.
CHECKPOINT = Path(os.environ.get("GRAPEVINE_CHECKPOINT", BEST_MODEL_DIR / "model.pt"))
LOW_CONFIDENCE = 0.70
VIEW_SIZE = 448  # display size of the square the model sees (it resizes every photo to 224 x 224)


@st.cache_resource(show_spinner="Loading the model...")
def get_predictor() -> Predictor:
    # CPU keeps the app independent of any training job that may be using the GPU.
    return Predictor(CHECKPOINT, device="cpu")


@st.cache_data(show_spinner=False, max_entries=16)
def classify(image_bytes: bytes) -> dict:
    """Run the model once per uploaded file and pre-encode the two layers of the Grad-CAM viewer."""
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    pred = get_predictor().predict(image, gradcam=True)
    photo = image.resize((VIEW_SIZE, VIEW_SIZE), Image.BICUBIC)
    cam = np.asarray(Image.fromarray(np.uint8(pred.cam * 255)).resize((VIEW_SIZE, VIEW_SIZE), Image.BILINEAR)) / 255.0
    heat = Image.fromarray(np.uint8(matplotlib.colormaps["jet"](cam)[..., :3] * 255))
    return {"label": pred.label, "confidence": pred.confidence, "probabilities": pred.probabilities,
            "photo": _data_uri(photo), "heat": _data_uri(heat), "id": hashlib.sha1(image_bytes).hexdigest()[:12]}


def _data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _viewer(result: dict) -> None:
    """Drag-to-compare viewer: Grad-CAM on the left of the handle, the photo on the right.

    Opacity and the split run in the browser, so dragging never re-runs the model."""
    vid = f"cmp-{result['id']}"
    st.html(
        f"""
<div class="cmp" id="{vid}">
  <img src="{result['photo']}" alt="Uploaded leaf as the model sees it">
  <img class="cmp-heat" src="{result['heat']}" alt="Grad-CAM heatmap for {result['label']}">
  <div class="cmp-scan"></div>
  <div class="cmp-handle"></div>
  <span class="cmp-tag l">Grad-CAM</span><span class="cmp-tag r">Photo</span>
  <input class="cmp-split" type="range" min="0" max="100" value="0" aria-label="Drag to compare Grad-CAM and photo">
</div>
<div class="cmp-tools"><label for="{vid}-a">Heatmap strength</label>
  <input id="{vid}-a" class="cmp-alpha" type="range" min="10" max="90" value="55"></div>
<script>
(() => {{
  const root = document.getElementById("{vid}");
  if (!root || root.dataset.ready) return;
  root.dataset.ready = "1";
  const split = root.querySelector(".cmp-split");
  const alpha = document.getElementById("{vid}-a");
  const setSplit = v => {{ split.value = v; root.style.setProperty("--split", v + "%"); }};
  split.addEventListener("input", () => setSplit(split.value));
  alpha.addEventListener("input", () => root.style.setProperty("--alpha", alpha.value / 100));
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (still) {{ setSplit(60); return; }}
  const t0 = performance.now();
  const step = t => {{
    const k = Math.min(1, (t - t0) / 1100);
    setSplit(Math.round(60 * (1 - Math.pow(1 - k, 3))));
    if (k < 1) requestAnimationFrame(step);
  }};
  requestAnimationFrame(step);
  try {{
    if (!sessionStorage.getItem("{vid}")) {{
      sessionStorage.setItem("{vid}", "1");
      root.scrollIntoView({{behavior: "smooth", block: "center"}});
    }}
  }} catch (e) {{}}
}})();
</script>
""",
        unsafe_allow_javascript=True,
    )


def _diagnosis(result: dict, summary: dict | None) -> None:
    color, note = style.CLASSES.get(result["label"], (style.GRAPE, ""))
    bars = "".join(
        f'<div class="prob"><span>{name}</span><div class="prob-track"><div class="prob-fill" '
        f'style="width:{100 * p:.1f}%;background:{style.CLASSES.get(name, (style.GRAPE,))[0]};animation-delay:{.15 + .08 * i:.2f}s">'
        f'</div></div><span class="prob-val">{p:.1%}</span></div>'
        for i, (name, p) in enumerate(sorted(result["probabilities"].items(), key=lambda kv: -kv[1]))
    )
    style.html(
        f"""
<div class="dx">
  <p class="dx-label">Most likely</p>
  <div class="dx-name"><span class="dx-dot" style="background:{color}"></span>{result['label']}</div>
  <div class="dx-conf"><b>{result['confidence']:.1%}</b> confidence</div>
  <p class="dx-note">{note}</p>
  {bars}
</div>
"""
    )
    if result["confidence"] < LOW_CONFIDENCE:
        st.warning(f"Confidence is below {LOW_CONFIDENCE:.0%}. The photo may differ from the training data "
                   "(background, lighting, several leaves, another plant), so treat this result with caution.")
    typical = (f" On the test set the median was {summary['confidence_median']:.0%}."
               if summary and summary.get("confidence_median") else "")
    st.caption("Confidence is the model's probability for its top class. Training used label smoothing, so even "
               f"clear cases stay below 100%.{typical} Drag across the image to compare the heatmap with the photo; "
               "red marks the regions that most influenced the answer.")


def render() -> None:
    if not CHECKPOINT.exists():
        st.error(f"No trained model found at `{CHECKPOINT}`. Run the training pipeline first (see README).")
        st.stop()
    predictor = get_predictor()
    summary = reports.summary(predictor.model_name)
    model_label = reports.MODEL_LABELS.get(predictor.model_name, predictor.model_name)

    left, right = st.columns([7, 5], gap="large")
    with left:
        style.intro("Which disease is on this grape leaf?",
                    f"Upload a photo of one grape leaf. {model_label}, picked from three trained networks, names "
                    "the disease and shows which part of the leaf its answer is based on.")
        uploaded = st.file_uploader("Leaf photo", type=["jpg", "jpeg", "png", "bmp", "webp"],
                                    help="One leaf per photo, filling most of the frame, in sharp focus.")
        if summary and summary.get("accuracy_low"):
            facts = [f"<b>≥{summary['accuracy_low']:.1%}</b> accuracy on {summary['leaves']} unseen leaves*",
                     f"<b>{summary['train_images']:,}</b> training photos" if summary.get("train_images") else "",
                     "<b>4</b> classes"]
            style.html('<div class="facts">' + "".join(f"<span>{f}</span>" for f in facts if f) + "</div>"
                       f'<p class="footnote">*Exact 95% lower bound, one photo per held-out leaf. These are lab photos '
                       "on a plain background; accuracy on field photos has not been measured and is likely lower.</p>")
    with right:
        guide = "".join(
            f'<div class="guide-item"><div class="guide-swatch" style="background:{c}"></div>'
            f"<div><b>{name}</b><span>{note}</span></div></div>"
            for name, (c, note) in style.CLASSES.items()
        )
        style.html(f'<div class="guide"><h3>What it can recognise</h3>{guide}</div>')

    if uploaded is None:
        return
    try:
        with st.spinner("Analysing the leaf..."):
            result = classify(uploaded.getvalue())
    except (UnidentifiedImageError, OSError):
        st.error("This file could not be read as an image. Upload a JPG, PNG, BMP or WebP photo.")
        return

    style.section("Result")
    view, dx = st.columns([6, 5], gap="large")
    with view:
        _viewer(result)
    with dx:
        _diagnosis(result, summary)
