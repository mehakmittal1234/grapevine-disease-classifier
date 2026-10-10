"""Models page: the three trained networks side by side, and how the served one was chosen."""

from __future__ import annotations

import streamlit as st

from . import charts, reports, style

METRICS = {
    "Validation log-loss": ("val_log_loss", "Validation log-loss (lower is better)", ".4f"),
    "Parameters": ("params_millions", "Parameters (millions)", ".1f"),
    "File size": ("checkpoint_mb", "Checkpoint size (MB)", ".1f"),
    "CPU time": ("latency_ms_cpu", "CPU time per photo on an M2 (ms)", ".0f"),
    "GPU time": ("latency_ms_gpu", "Apple-GPU time per photo (ms)", ".1f"),
    "Training time": ("train_minutes", "Training time (minutes)", ".0f"),
}
CURVES = {
    "Validation macro-F1": ("val_macro_f1", "Validation macro-F1"),
    "Validation loss": ("val_loss", "Validation loss"),
    "Training loss": ("train_loss", "Training loss"),
    "Training accuracy": ("train_acc", "Training accuracy"),
}


def render() -> None:
    df = reports.comparison()
    style.intro("Three networks, one fair comparison.",
                "MobileNetV2, ResNet50 and EfficientNet-B0 were fine-tuned from ImageNet on the same leaf-grouped "
                "split, with the same augmentation, schedule and early stopping. Only the validation set was used "
                "to pick a winner; the test set was opened once, after the choice.")
    if df is None:
        st.info("The model comparison report is not available in this deployment.")
        return
    chosen = df.loc[df["selected"]].iloc[0] if df["selected"].any() else df.iloc[0]
    card = reports.model_card() or {}

    style.figures([
        (f"{len(df)}", "architectures trained, each in two phases: classifier head first, then partial fine-tuning"),
        (f"{df['train_minutes'].sum():.0f} min", "total training time on an 8 GB Apple M2"),
        (chosen["label"], f"served in this app, {chosen['params_millions']:.1f} M parameters, "
                          f"{chosen['checkpoint_mb']:.0f} MB"),
    ])

    style.section("How the winner was picked")
    rule = card.get("selection_rule", "max validation macro-F1, then min validation log-loss, then fewest parameters")
    style.html(f'<div class="pick">Rule, fixed before training: <b>{rule}</b>. All three tied on validation macro-F1, '
               f"so the decision went to log-loss, which rewards well-calibrated probabilities. "
               f"{chosen['label']} had the lowest ({chosen['val_log_loss']:.4f}).</div>")

    style.section("Compare the models", "Pick a measure. The served model is highlighted.")
    choice = st.segmented_control("Measure", list(METRICS), default="Validation log-loss", label_visibility="collapsed")
    column, title, fmt = METRICS[choice or "Validation log-loss"]
    log_loss = column == "val_log_loss"
    st.altair_chart(charts.metric_bars(df, column, title, fmt, chosen["model"], zero=not log_loss),
                    width="stretch", theme=None)
    if log_loss:
        st.caption("Zoomed axis: the three values differ only in the third decimal place.")

    hist = reports.histories()
    if hist is not None:
        style.section("Training, epoch by epoch",
                      "Circles are the head-only phase, diamonds the fine-tuning phase. Click a legend entry to focus on one model.")
        curve = st.segmented_control("Curve", list(CURVES), default="Validation loss", label_visibility="collapsed")
        column, title = CURVES[curve or "Validation loss"]
        st.altair_chart(charts.training_curves(hist, column, title, chosen["label"]), width="stretch", theme=None)

    style.section("All measurements")
    table = df[["label", "params_millions", "checkpoint_mb", "latency_ms_cpu", "latency_ms_gpu", "train_minutes",
                "best_epoch", "val_log_loss"]]
    st.dataframe(table, hide_index=True, column_config={
        "label": "Model",
        "params_millions": st.column_config.NumberColumn("Parameters (M)", format="%.1f"),
        "checkpoint_mb": st.column_config.NumberColumn("File (MB)", format="%.1f"),
        "latency_ms_cpu": st.column_config.NumberColumn("CPU ms / photo", format="%.0f"),
        "latency_ms_gpu": st.column_config.NumberColumn("GPU ms / photo", format="%.1f"),
        "train_minutes": st.column_config.NumberColumn("Training (min)", format="%.0f"),
        "best_epoch": st.column_config.NumberColumn("Best epoch"),
        "val_log_loss": st.column_config.NumberColumn("Val log-loss", format="%.4f"),
    })
    st.caption("Timings are batch size 1 on the training machine (Apple M2). Per-model test-set reports are on GitHub.")
