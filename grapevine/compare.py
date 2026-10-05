"""Compare the evaluated models and promote the best one to ``artifacts/best_model``.

Usage::

    python -m grapevine.compare

The winner is chosen on the **validation** split only: highest macro-F1, then lowest
validation log-loss, then fewest parameters. Test metrics are reported for every model but
never used for the choice, so the selected model's test score stays an unbiased estimate.
All numbers are read from the JSON files written by ``train`` and ``evaluate``.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone

import pandas as pd

from . import viz
from .config import BEST_MODEL_DIR, DATA_REPORT_JSON, MODEL_NAMES, REPORTS_DIR, RUNS_DIR
from .utils import load_json, project_relative, save_json

SELECTION_RULE = "max validation macro-F1, then min validation log-loss, then fewest parameters"


def model_row(name: str) -> dict | None:
    metrics_path, summary_path = REPORTS_DIR / name / "metrics.json", RUNS_DIR / name / "train_summary.json"
    if not (metrics_path.exists() and summary_path.exists()):
        print(f"[compare] skipping {name}: missing {metrics_path.name if not metrics_path.exists() else summary_path.name}")
        return None
    m, s = load_json(metrics_path), load_json(summary_path)
    val, test, eff = m["val"]["all"], m["test"], m["efficiency"]
    latency_gpu = next((v for k, v in eff.items() if k.startswith("latency_ms_batch1_") and not k.endswith("cpu")), None)
    return {
        "model": name,
        "val_macro_f1": val["macro_f1"],
        "val_accuracy": val["accuracy"],
        "val_log_loss": val["log_loss"],
        "test_accuracy": test["all"]["accuracy"],
        "test_macro_precision": test["all"]["macro_precision"],
        "test_macro_recall": test["all"]["macro_recall"],
        "test_macro_f1": test["all"]["macro_f1"],
        "test_balanced_accuracy": test["all"]["balanced_accuracy"],
        "test_roc_auc_ovr": test["all"]["roc_auc_ovr_macro"],
        "test_mcc": test["all"]["mcc"],
        "test_errors": test["all"]["errors"],
        "test_n": test["all"]["n"],
        "test_original_photos_accuracy": test["original_photos"]["accuracy"],
        "test_original_photos_macro_f1": test["original_photos"]["macro_f1"],
        "test_one_image_per_leaf_macro_f1": test["one_image_per_leaf"]["macro_f1"],
        "test_png_keras_aug_macro_f1": test["png_keras_aug"]["macro_f1"],
        "test_strict_n": test["strict_no_weak_train_match"]["n"],
        "test_strict_accuracy": test["strict_no_weak_train_match"]["accuracy"],
        "test_strict_macro_f1": test["strict_no_weak_train_match"]["macro_f1"],
        "test_one_per_leaf_acc_95ci_low": (m.get("test_accuracy_exact_95ci_one_image_per_leaf") or {}).get("interval", [None])[0],
        "test_background_neutralized_accuracy": (m.get("background_neutralized_test") or {}).get("all", {}).get("accuracy"),
        "params_millions": eff["parameters"] / 1e6,
        "checkpoint_mb": eff["checkpoint_mb"],
        "latency_ms_gpu": latency_gpu,
        "latency_ms_cpu": eff["latency_ms_batch1_cpu"],
        "train_minutes": s["train_seconds"] / 60,
        "best_epoch": s["best_epoch"],
        "epochs_run": s["epochs_run"],
    }


def to_markdown(table: pd.DataFrame) -> str:
    cols = list(table.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in table.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    rows = [r for r in (model_row(n) for n in MODEL_NAMES) if r is not None]
    if not rows:
        raise SystemExit("No evaluated models found - run train and evaluate first.")
    table = pd.DataFrame(rows)
    ranked = table.sort_values(["val_macro_f1", "val_log_loss", "params_millions"], ascending=[False, True, True])
    best = ranked.iloc[0]["model"]
    table["selected"] = table["model"] == best

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    headline = table[["model", "val_macro_f1", "test_accuracy", "test_macro_precision", "test_macro_recall",
                      "test_macro_f1", "test_roc_auc_ovr", "test_original_photos_macro_f1", "test_strict_macro_f1",
                      "test_one_per_leaf_acc_95ci_low", "test_background_neutralized_accuracy", "test_errors", "test_n",
                      "params_millions", "latency_ms_cpu", "train_minutes", "selected"]]
    md = (f"# Model comparison\n\nSelection rule: {SELECTION_RULE}. Selected: **{best}**.\n\n"
          f"{to_markdown(headline)}\n\nFull table: `model_comparison.csv`.\n")
    (REPORTS_DIR / "model_comparison.md").write_text(md)
    viz.comparison_bars(table, ["val_macro_f1", "test_accuracy", "test_macro_f1", "test_strict_macro_f1"],
                        REPORTS_DIR / "model_comparison.png")

    BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNS_DIR / best / "best.pt", BEST_MODEL_DIR / "model.pt")
    metrics = load_json(REPORTS_DIR / best / "metrics.json")
    data_report = load_json(DATA_REPORT_JSON) if DATA_REPORT_JSON.exists() else {}
    card = {
        "model": best,
        "selected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": SELECTION_RULE,
        "checkpoint": project_relative(BEST_MODEL_DIR / "model.pt"),
        "class_names": list(metrics["test"]["all"]["per_class"].keys()),
        "input": "RGB, resized to 224x224, ImageNet mean/std normalisation",
        "validation": metrics["val"],
        "test": metrics["test"],
        "efficiency": metrics["efficiency"],
        "test_confidence": metrics.get("test_confidence"),
        "background_neutralized_test": metrics.get("background_neutralized_test"),
        "test_accuracy_exact_95ci_one_image_per_leaf": metrics.get("test_accuracy_exact_95ci_one_image_per_leaf"),
        "training": load_json(RUNS_DIR / best / "train_summary.json"),
        "data": {
            "splits": data_report.get("splits"),
            "leakage_checks": data_report.get("leakage_checks"),
        },
        "intended_use": "Lab-style photos of single grape leaves on a plain background (PlantVillage-like). "
                        "Not validated on field images, other crops or other diseases.",
    }
    save_json(card, BEST_MODEL_DIR / "model_card.json")
    print(md)
    print(f"Best model ({best}) copied to {BEST_MODEL_DIR / 'model.pt'}")


if __name__ == "__main__":
    main()
