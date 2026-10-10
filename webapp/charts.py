"""Altair charts for the report pages, styled to match ``webapp.style``."""

from __future__ import annotations

import altair as alt
import pandas as pd

from .style import BODY_FONT, CLASSES, COPPER, GRAPE, INK, LINE, MUTED

CLASS_SCALE = alt.Scale(domain=list(CLASSES), range=[c for c, _ in CLASSES.values()])
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_NAMES = {"train": "Training", "val": "Validation", "test": "Test"}


def _style(chart: alt.TopLevelMixin) -> alt.TopLevelMixin:
    return (
        chart.configure(font=BODY_FONT, background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor=MUTED, titleColor=MUTED, domainColor=LINE, tickColor=LINE, gridColor="#E2E6DE",
                        labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
        .configure_legend(labelColor=INK, titleColor=MUTED, labelFontSize=12, titleFontSize=12,
                          titleFontWeight="normal", orient="top", symbolType="circle")
    )


def split_composition(report: dict) -> alt.Chart:
    rows = [
        {"split": SPLIT_NAMES[s], "order": i, "class": cls, "files": n}
        for i, s in enumerate(SPLIT_ORDER)
        for cls, n in report["splits"][s]["files_per_class"].items()
    ]
    groups = {SPLIT_NAMES[s]: report["splits"][s]["source_groups"] for s in SPLIT_ORDER}
    df = pd.DataFrame(rows).assign(leaves=lambda d: d["split"].map(groups))
    chart = alt.Chart(df).mark_bar(cornerRadius=3).encode(
        y=alt.Y("split:N", sort=[SPLIT_NAMES[s] for s in SPLIT_ORDER], title=None, scale=alt.Scale(paddingInner=.35)),
        x=alt.X("files:Q", title="Image files", stack="zero"),
        color=alt.Color("class:N", scale=CLASS_SCALE, title=None),
        order=alt.Order("class:N"),
        tooltip=[alt.Tooltip("split:N", title="Split"), alt.Tooltip("class:N", title="Class"),
                 alt.Tooltip("files:Q", title="Files", format=","), alt.Tooltip("leaves:Q", title="Leaves in split", format=",")],
    ).properties(height=150)
    return _style(chart)


def audit_rounds(report: dict) -> alt.Chart:
    df = pd.DataFrame([
        {"round": f"Round {r['round']}", "matches": r["heldout_images_with_train_match"],
         "checked": r["pairs_checked"]} for r in report["audit_rounds"]
    ])
    base = alt.Chart(df).encode(x=alt.X("round:N", title=None, axis=alt.Axis(labelAngle=0)))
    bars = base.mark_bar(size=46, cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=GRAPE).encode(
        y=alt.Y("matches:Q", title="Photos still matching", axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("round:N", title="Audit"), alt.Tooltip("checked:Q", title="Pairs checked", format=","),
                 alt.Tooltip("matches:Q", title="Photos still matching")],
    )
    labels = base.mark_text(dy=-9, color=INK, fontSize=13, fontWeight=600).encode(y="matches:Q", text="matches:Q")
    return _style((bars + labels).properties(height=230))


def metric_bars(df: pd.DataFrame, column: str, title: str, fmt: str, selected: str, zero: bool = True) -> alt.Chart:
    data = df[["label", "model", column]].rename(columns={column: "value"})
    data["chosen"] = data["model"] == selected
    base = alt.Chart(data).encode(y=alt.Y("label:N", title=None, sort=list(data["label"]), scale=alt.Scale(paddingInner=.35)))
    color = alt.condition("datum.chosen", alt.value(GRAPE), alt.value("#B9C2B4"))
    tooltip = [alt.Tooltip("label:N", title="Model"), alt.Tooltip("value:Q", title=title, format=fmt)]
    if not zero:
        # Differences are small relative to the values, so plot dots on a zoomed axis instead of bars from zero.
        lo, hi = data["value"].min(), data["value"].max()
        pad = (hi - lo) * 0.6 or abs(hi) * 0.01
        x = alt.X("value:Q", title=title, scale=alt.Scale(domain=[lo - pad, hi + pad], zero=False))
        marks = base.mark_circle(size=260, opacity=1).encode(x=x, color=color, tooltip=tooltip)
        labels = base.mark_text(dy=-18, color=INK, fontSize=13).encode(x="value:Q", text=alt.Text("value:Q", format=fmt))
    else:
        marks = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("value:Q", title=title, scale=alt.Scale(zero=True)), color=color, tooltip=tooltip)
        labels = base.mark_text(align="left", dx=6, color=INK, fontSize=13).encode(
            x="value:Q", text=alt.Text("value:Q", format=fmt))
    return _style((marks + labels).properties(height=190))


def training_curves(hist: pd.DataFrame, column: str, title: str, selected_label: str) -> alt.Chart:
    pick = alt.selection_point(fields=["label"], bind="legend")
    colors = alt.Scale(domain=[selected_label] + sorted(set(hist["label"]) - {selected_label}),
                       range=[GRAPE, COPPER, "#9AA593"])
    base = alt.Chart(hist).encode(
        x=alt.X("epoch:O", title="Epoch", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{column}:Q", title=title, scale=alt.Scale(zero=False)),
        color=alt.Color("label:N", scale=colors, title=None),
        opacity=alt.condition(pick, alt.value(1), alt.value(.15)),
    )
    lines = base.mark_line(strokeWidth=2.5, interpolate="monotone")
    points = base.mark_point(filled=True, size=55).encode(
        shape=alt.Shape("phase:N", title="Phase", scale=alt.Scale(domain=["head", "finetune"], range=["circle", "diamond"])),
        tooltip=[alt.Tooltip("label:N", title="Model"), "epoch:O", alt.Tooltip("phase:N", title="Phase"),
                 alt.Tooltip(f"{column}:Q", title=title, format=".4f"),
                 alt.Tooltip("seconds:Q", title="Epoch time (s)", format=".0f")],
    )
    return _style((lines + points).add_params(pick).properties(height=320))


def confidence_histogram(preds: pd.DataFrame) -> alt.Chart:
    chart = alt.Chart(preds).mark_bar(opacity=.9, binSpacing=1).encode(
        x=alt.X("confidence:Q", bin=alt.Bin(step=0.01), title="Model confidence in its answer",
                axis=alt.Axis(format=".0%")),
        y=alt.Y("count():Q", title="Test photos", stack="zero"),
        color=alt.Color("label:N", scale=CLASS_SCALE, title=None),
        tooltip=[alt.Tooltip("label:N", title="True class"), alt.Tooltip("count():Q", title="Photos"),
                 alt.Tooltip("confidence:Q", bin=alt.Bin(step=0.01), title="Confidence", format=".0%")],
    )
    return _style(chart.properties(height=260))


def robustness(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y("test:N", title=None, sort=list(df["test"]), scale=alt.Scale(paddingInner=.3),
                                        axis=alt.Axis(labelLimit=320, labelFontSize=12.5)))
    bars = base.mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        x=alt.X("accuracy:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Model", "Baseline"], range=[GRAPE, "#B9C2B4"]), title=None),
        tooltip=[alt.Tooltip("test:N", title="Test"), alt.Tooltip("accuracy:Q", format=".2%", title="Accuracy"),
                 alt.Tooltip("detail:N", title="Details")],
    )
    labels = base.mark_text(align="right", dx=-8, color="#FFFFFF", fontSize=12.5, fontWeight=600).encode(
        x="accuracy:Q", text=alt.Text("label:N"))
    return _style((bars + labels).properties(height=46 * len(df)))
