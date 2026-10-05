"""Write ``artifacts/reports/results.md`` from the JSON/CSV outputs of the pipeline.

Usage::

    python -m grapevine.report

Every number in the file is read from ``data_report.json``, ``model_comparison.csv``,
``metrics.json`` and ``train_summary.json``; nothing is typed in by hand.
"""

from __future__ import annotations

import pandas as pd

from .config import ARTIFACTS_DIR, BEST_MODEL_DIR, DATA_REPORT_JSON, MODEL_NAMES, PROJECT_ROOT, REPORTS_DIR, RUNS_DIR
from .utils import load_json

VIEWS = {
    "all": "all test images",
    "original_photos": "un-augmented JPG photos",
    "one_image_per_leaf": "one image per leaf group",
    "png_keras_aug": "Keras-augmented PNGs",
    "strict_no_weak_train_match": "strict (no weak match to train)",
}


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * x:.2f}%"


def _f(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def data_section(r: dict) -> list[str]:
    m, nd, s = r["manifest"], r["near_duplicates"], r["splits"]
    weak = r["heldout_weak_matches"]
    weak_key = next(k for k in weak["test"] if k.startswith("with_"))
    lines = [
        "## Data audit and split",
        "",
        f"- Files scanned: {m['files_scanned']}; unreadable: {len(m['unreadable_files'])}; "
        f"byte-identical duplicates removed: {m['exact_duplicates_removed']}; usable: {m['usable_files']}.",
        f"- Filename groups (UUID / Keras source index): {nd['filename_groups']}; after merging verified "
        f"same-leaf links: **{nd['source_groups_after_merge']} leaf groups** (largest {nd['largest_source_group_files']} files).",
        f"- Verified same-leaf links: {nd['links']} ({', '.join(f'{k}: {v}' for k, v in nd['links_by_kind'].items())}); "
        f"{nd['links_from_sequence_rule_only']} of them via the consecutive-photo rule.",
        f"- Keras PNG groups whose source leaf is also one of the JPG photos: "
        f"**{nd['png_groups_whose_source_leaf_is_also_a_jpg']} of {nd['png_groups']}**.",
        f"- Cross-class matches (reported, not merged): {len(nd['cross_class_matches_not_merged'])}.",
        f"- Leakage checks passed: **{r['leakage_checks']['passed']}** "
        f"(audit rounds: {len(r['audit_rounds'])}; final round checked "
        f"{r['audit_rounds'][-1]['heldout_images_checked']} held-out images against "
        f"{r['audit_rounds'][-1]['train_neighbours_per_image']} training neighbours each).",
        "- Held-out images with an unlinked weak match to train (excluded in the strict view): "
        + "; ".join(f"{k} {v[weak_key]}/{v['images']}" for k, v in weak.items()) + ".",
        "",
        "| split | files | leaf groups | " + " | ".join(sorted(s["train"]["files_per_class"])) + " |",
        "|---|---|---|" + "---|" * len(s["train"]["files_per_class"]),
    ]
    for name, part in s.items():
        lines.append(f"| {name} | {part['files']} | {part['source_groups']} | "
                     + " | ".join(str(part["files_per_class"].get(c, 0)) for c in sorted(s["train"]["files_per_class"])) + " |")
    return lines + [""]


def bias_section(b: dict) -> list[str]:
    t, o = b["test_all"], b["test_original_photos"]
    return [
        "## Capture bias (background only, no CNN)",
        "",
        f"A logistic regression on colour statistics of background-like border pixels, trained on the training "
        f"split, reaches **{_pct(t['accuracy'])}** test accuracy (macro-F1 {_f(t['macro_f1'])}, n={t['n']}; "
        f"un-augmented photos {_pct(o['accuracy'])}, n={o['n']}) against {_pct(b['chance_accuracy'])} chance: "
        "the backgrounds carry class information. See `background_neutralized_test` below for the CNNs.",
        "",
    ]


def comparison_section() -> list[str]:
    table = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    lines = [
        "## Model comparison (held-out test set; selection on validation only)",
        "",
        "| model | val macro-F1 | test accuracy | test macro-P | test macro-R | test macro-F1 | test ROC-AUC (OvR) "
        "| strict test macro-F1 (n) | one-image-per-leaf acc. 95% CI low | background-neutralised test acc. "
        "| errors / n | params (M) | CPU latency (ms) | train time (min) | selected |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.itertuples():
        low = getattr(r, "test_one_per_leaf_acc_95ci_low", None)
        neutral = getattr(r, "test_background_neutralized_accuracy", None)
        lines.append(
            f"| {r.model} | {_f(r.val_macro_f1)} | {_pct(r.test_accuracy)} | {_f(r.test_macro_precision)} | "
            f"{_f(r.test_macro_recall)} | {_f(r.test_macro_f1)} | {_f(r.test_roc_auc_ovr)} | "
            f"{_f(r.test_strict_macro_f1)} ({r.test_strict_n}) | {_pct(None if pd.isna(low) else low)} | "
            f"{_pct(None if pd.isna(neutral) else neutral)} | {r.test_errors} / {r.test_n} | {r.params_millions:.2f} | "
            f"{r.latency_ms_cpu:.1f} | {r.train_minutes:.1f} | {'**yes**' if r.selected else ''} |")
    return lines + [""]


def model_section(name: str) -> list[str]:
    m = load_json(REPORTS_DIR / name / "metrics.json")
    s = load_json(RUNS_DIR / name / "train_summary.json")
    lines = [
        f"### {name}",
        "",
        f"Best epoch {s['best_epoch']} ({s['best_phase']}) of {s['epochs_run']} run; training time {s['train_seconds'] / 60:.1f} min "
        f"on {s['device']} (AMP: {s['amp']}).",
        "",
        "| test view | n | accuracy | balanced acc. | macro-F1 | weighted-F1 | MCC | errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for view, label in VIEWS.items():
        v = m["test"].get(view)
        if v:
            lines.append(f"| {label} | {v['n']} | {_pct(v['accuracy'])} | {_pct(v['balanced_accuracy'])} | "
                         f"{_f(v['macro_f1'])} | {_f(v['weighted_f1'])} | {_f(v['mcc'])} | {v['errors']} |")
    pc = m["test"]["all"]["per_class"]
    lines += ["", "| class (test, all) | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    lines += [f"| {c} | {_f(x['precision'])} | {_f(x['recall'])} | {_f(x['f1'])} | {x['support']} |" for c, x in pc.items()]
    ci = m.get("test_accuracy_exact_95ci_one_image_per_leaf")
    if ci:
        lines += ["", f"One image per leaf: {ci['correct']}/{ci['n']} correct; exact (Clopper-Pearson) 95% interval for "
                      f"accuracy {_pct(ci['interval'][0])} - {_pct(ci['interval'][1])}."]
    bs = m.get("test_bootstrap", {}).get("all")
    if bs:
        lines += [f"Leaf-group bootstrap 95% interval (all test images): accuracy {_pct(bs['accuracy_95ci'][0])} - "
                  f"{_pct(bs['accuracy_95ci'][1])}, macro-F1 {_f(bs['macro_f1_95ci'][0])} - {_f(bs['macro_f1_95ci'][1])}."]
    bn = m.get("background_neutralized_test")
    if bn:
        lines += [f"Background painted grey: test accuracy {_pct(bn['all']['accuracy'])} ({bn['all']['errors']} errors / "
                  f"{bn['all']['n']}); un-augmented photos {_pct(bn['original_photos']['accuracy'])}."]
    sc = m.get("black_corner_shortcut_check")
    if sc:
        lines += ["", f"Black-corner shortcut check on {sc['n']} non-Healthy test photos: predicted-Healthy rate "
                      f"{_pct(sc['unchanged']['predicted_healthy_rate'])} unchanged vs "
                      f"{_pct(sc['rotated_30deg_black_corners']['predicted_healthy_rate'])} after a black-cornered 30-degree "
                      f"rotation (accuracy {_pct(sc['unchanged']['accuracy'])} vs {_pct(sc['rotated_30deg_black_corners']['accuracy'])})."]
    return lines + [""]


def readme_section() -> list[str]:
    """Headline results for README.md (between the RESULTS markers)."""
    card_path = BEST_MODEL_DIR / "model_card.json"
    if not ((REPORTS_DIR / "model_comparison.csv").exists() and card_path.exists()):
        return []
    card = load_json(card_path)
    best = card["model"]
    m = load_json(REPORTS_DIR / best / "metrics.json")
    lines = ["## Results", "",
             "Generated from the output files by `python -m grapevine.report`; full details in "
             "[`artifacts/reports/results.md`](artifacts/reports/results.md).", ""]
    lines += comparison_section()[2:]  # table without its heading
    t = m["test"]["all"]
    ci = m.get("test_accuracy_exact_95ci_one_image_per_leaf")
    lines += [f"**Selected model: {best}** ({card['selection_rule']})."]
    if ci:
        lines += [f"**Headline figure: accuracy >= {_pct(ci['interval'][0])} on unseen leaves**, the exact 95% lower "
                  f"bound counting each of the {ci['n']} held-out leaves once ({ci['correct']}/{ci['n']} correct). A "
                  "finite test set cannot establish perfect accuracy, and field photos were not evaluated."]
    lines += [f"Measured on the held-out split ({t['n']} images): accuracy {_pct(t['accuracy'])}, "
              f"macro-F1 {_f(t['macro_f1'])}, {t['errors']} errors."]
    bn = m.get("background_neutralized_test")
    if bn:
        lines += [f"With the background painted grey the test accuracy is {_pct(bn['all']['accuracy'])} "
                  f"({bn['all']['errors']} errors)."]
    table = pd.read_csv(REPORTS_DIR / "model_comparison.csv")
    fastest = table.loc[table["latency_ms_cpu"].idxmin()]
    chosen = table.loc[table["model"] == best].iloc[0]
    if (table["test_accuracy"] == chosen["test_accuracy"]).all() and fastest["model"] != best:
        lines += [f"All models reach the same test accuracy, so the choice rests on the validation log-loss tie-break "
                  f"({', '.join(f'{r.model} {r.val_log_loss:.4f}' for r in table.itertuples())}). If CPU latency matters "
                  f"more, {fastest['model']} is {chosen['latency_ms_cpu'] / fastest['latency_ms_cpu']:.1f}x faster on CPU "
                  f"({fastest['latency_ms_cpu']:.0f} vs {chosen['latency_ms_cpu']:.0f} ms per image): pass "
                  f"`--checkpoint artifacts/runs/{fastest['model']}/best.pt` to `predict`, or set "
                  f"`GRAPEVINE_CHECKPOINT` for the app."]
    lines += ["", f"![confusion matrix](artifacts/reports/{best}/confusion_matrix_test_all.png)", "",
              "Grad-CAM example grids for the test set are written locally to `artifacts/reports/gradcam/` by "
              "`python -m grapevine.gradcam` (they show dataset photos, so they are not committed); the app "
              "computes Grad-CAM live for every uploaded image.", ""]
    return lines


def update_readme(section: list[str]) -> None:
    # Only the project's default artifacts folder describes the project's own results.
    if not section or ARTIFACTS_DIR != (PROJECT_ROOT / "artifacts").resolve():
        return
    readme = PROJECT_ROOT / "README.md"
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if not readme.exists():
        return
    text = readme.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        readme.write_text(head + start + "\n" + "\n".join(section) + "\n" + end + tail)
        print(f"Updated results in {readme}")


def main() -> None:
    lines = ["# Results", "", "Generated by `python -m grapevine.report` from the pipeline's output files.", ""]
    if DATA_REPORT_JSON.exists():
        lines += data_section(load_json(DATA_REPORT_JSON))
    bias = DATA_REPORT_JSON.parent / "background_bias.json"
    if bias.exists():
        lines += bias_section(load_json(bias))
    if (REPORTS_DIR / "model_comparison.csv").exists():
        lines += comparison_section()
    lines += ["## Per-model test results", ""]
    for name in MODEL_NAMES:
        if (REPORTS_DIR / name / "metrics.json").exists() and (RUNS_DIR / name / "train_summary.json").exists():
            lines += model_section(name)
    card = BEST_MODEL_DIR / "model_card.json"
    if card.exists():
        lines += [f"Selected model: **{load_json(card)['model']}** -> `artifacts/best_model/model.pt`.", ""]
    out = REPORTS_DIR / "results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")
    update_readme(readme_section())


if __name__ == "__main__":
    main()
