"""Reliability page: conservative accuracy, shortcut checks, confidence and known limits."""

from __future__ import annotations

import streamlit as st

from . import charts, reports, style
from .diagnose import get_predictor


def render() -> None:
    model = get_predictor().model_name
    summary = reports.summary(model)
    bias = reports.background_bias() or {}
    card = reports.model_card() or {}
    style.intro("How far to trust it.",
                "A model can score well by learning the wrong cue, such as the background colour of the photo "
                "booth. These checks measure how much of the result comes from the leaf itself, and quote the "
                "most conservative figure the data supports.")
    if not summary or not summary.get("accuracy_low"):
        st.info("The evaluation report for the served model is not available in this deployment.")
        return

    background_only = (bias.get("test_all") or {}).get("accuracy")
    items = [
        (f"≥{summary['accuracy_low']:.1%}", f"accuracy on unseen leaves: the exact 95% lower bound, counting each of "
                                            f"the {summary['leaves']} held-out leaves once"),
    ]
    if summary.get("background_accuracy"):
        items.append((f"{summary['background_accuracy']:.1%}",
                      f"with the photo background painted grey ({summary['background_errors']} errors in "
                      f"{summary['background_n']:,} test photos)"))
    if background_only:
        items.append((f"{background_only:.0%}", "from background pixels alone, with no model of the leaf at all. "
                                                "Chance is 25%, so the backgrounds carry a class signal."))
    style.figures(items)

    rows = [{"test": "Chance, 4 classes", "accuracy": bias.get("chance_accuracy", .25), "kind": "Baseline",
             "detail": "Always guessing one class"}]
    if background_only:
        rows.append({"test": "Background pixels only", "accuracy": background_only, "kind": "Baseline",
                     "detail": "Simple classifier on the colour of the photo border"})
    if summary.get("background_accuracy"):
        rows.append({"test": "Model, background painted grey", "accuracy": summary["background_accuracy"],
                     "kind": "Model", "detail": f"{summary['background_n']:,} test photos"})
    rows.append({"test": "Model, lower bound on unseen leaves", "accuracy": summary["accuracy_low"], "kind": "Model",
                 "detail": f"Exact 95% bound, {summary['leaves']} leaves"})
    for r in rows:
        r["label"] = f"{r['accuracy']:.1%}"
    style.section("Is it looking at the leaf?",
                  "If the model relied on the background, hiding it would make accuracy fall towards the "
                  "background-only line. It barely moved.")
    st.altair_chart(charts.robustness(rows), width="stretch", theme=None)

    preds = reports.test_predictions(model)
    if preds is not None:
        conf = card.get("test_confidence") or {}
        style.section("How sure it is",
                      f"Top-class confidence on the {len(preds):,} test photos. Label smoothing trains the model "
                      f"towards 92.5% rather than 100% for the right class, so a typical answer scores around {conf.get('median', preds['confidence'].median()):.0%}; "
                      f"5% of photos fall below {conf.get('p05', preds['confidence'].quantile(.05)):.0%} and the "
                      f"lowest is {preds['confidence'].min():.0%}. The app warns below 70%.")
        st.altair_chart(charts.confidence_histogram(preds), width="stretch", theme=None)

    style.section("Known limits")
    st.markdown(
        "- **Lab photos only.** Every training and test photo shows one leaf on a plain background. Field photos "
        "with soil, sky or other leaves have not been tested, and accuracy there is likely lower.\n"
        "- **Four classes only.** Other grape problems (downy or powdery mildew, nutrient deficiency, insect damage) "
        "will be forced into one of the four labels.\n"
        "- **One dataset.** Test photos come from the same collection and capture setup as the training photos. "
        "Photos from a new source are the real test.\n"
        "- **Single training run.** Each architecture was trained once; run-to-run variation was not measured."
    )
